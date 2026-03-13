# train_dsr.py
# DSR training script with optional CometML logging.
# Mirrors the structure of the SB3 DQN training script.
#
# Usage:
#   # without comet
#   python3 train_dsr.py -e env_config.yaml -r my_run -s ./logs
#
#   # with comet
#   python3 train_dsr.py -e env_config.yaml -r my_run -s ./logs \
#       -p my_project -a <COMET_API_KEY>

from __future__ import annotations

# CometML MUST be imported before torch
import comet_ml

import argparse
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import gymnasium as gym
import socnavgym                        # noqa: F401  registers SocNavGym-v1
from socnavgym.wrappers import DiscreteActions, ExpertObservations

from rl.DSR import DSRAgent, DSRConfig


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--env_config",    required=True,  help="path to environment config")
ap.add_argument("-r", "--run_name",      required=True,  help="comet run name / checkpoint prefix")
ap.add_argument("-s", "--save_path",     required=True,  help="directory to save checkpoints")
ap.add_argument("-p", "--project_name",  required=False, default=None)
ap.add_argument("-a", "--api_key",       required=False, default=None)
ap.add_argument("-g", "--gpu",           required=False, default="0")
ap.add_argument("--total_steps",         required=False, default=1_000_000, type=int)
ap.add_argument("--eval_freq",           required=False, default=10_000,  type=int)
ap.add_argument("--eval_episodes",       required=False, default=5,       type=int)
ap.add_argument("--checkpoint_freq",     required=False, default=50_000,  type=int)
ap.add_argument("--log_freq",            required=False, default=1_000,   type=int)
args = vars(ap.parse_args())


# ──────────────────────────────────────────────────────────────────────────────
# CometML
# ──────────────────────────────────────────────────────────────────────────────
experiment = None
if args["api_key"] is not None:
    from comet_ml import Experiment
    experiment = Experiment(
        api_key=args["api_key"],
        project_name=args["project_name"],
        parse_args=False,
    )
    experiment.set_name(args["run_name"])
    print(f"[comet] logging to project '{args['project_name']}' as '{args['run_name']}'")
else:
    print("[info] no CometML api key provided – running without logging")


def log_metrics(metrics: dict, step: int):
    if experiment is not None:
        experiment.log_metrics(metrics, step=step)
    else:
        print(f"  step={step}  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))


# ──────────────────────────────────────────────────────────────────────────────
# Environments
# ──────────────────────────────────────────────────────────────────────────────
def make_env(env_config: str):
    env = gym.make("SocNavGym-v1", config=env_config)
    env = DiscreteActions(env)
    env = ExpertObservations(env)
    return env

train_env = make_env(args["env_config"])
eval_env  = make_env(args["env_config"])


# ──────────────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────────────────
device = f"cuda:{args['gpu']}" if torch.cuda.is_available() else "cpu"

config = DSRConfig(
    state_dim            = 5,        # raw input: [gx, gy, theta, hx, hy]
    feat_dim             = 64,       # encoder output size
    action_size          = 7,
    enc_hidden           = 128,
    map_size             = 10.0,
    sr_hidden            = 128,
    learning_rate        = 1e-4,
    gamma                = 0.5,    # reduced: max |psi| ~ 20x instead of 100x
    epsilon_start        = 1.0,
    epsilon_end          = 0.05,
    epsilon_decay_steps  = 50_000,
    target_update_freq   = 5_000,   # fresher targets = less SR drift
    batch_size           = 128,
    start_steps          = 5_000,
    buffer_size          = 100_000,
    train_freq           = 4,
    lambda_sr            = 0.5,     # slower SR growth
    lambda_r             = 1.0,
    grad_clip            = 0.5,     # tighter gradient clipping
    device               = device,
)


agent = DSRAgent(config)

if experiment is not None:
    experiment.log_parameters(vars(config))


# ──────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ──────────────────────────────────────────────────────────────────────────────
save_dir = Path(args["save_path"])
save_dir.mkdir(parents=True, exist_ok=True)

def save_checkpoint(tag: str):
    path = save_dir / f"{args['run_name']}_{tag}.pt"
    torch.save({
        "net_state":   agent.net.state_dict(),
        "tgt_state":   agent.tgt.state_dict(),
        "opt_sr_state": agent.opt_sr.state_dict(),
        "opt_rw_state": agent.opt_rw.state_dict(),
        "total_steps": agent.total_steps,
        "config":      vars(config),
    }, path)
    print(f"[ckpt] saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────
def evaluate(n_episodes: int = 5) -> dict:
    returns, lengths = [], []
    for _ in range(n_episodes):
        ep_len, ep_ret = agent.run_episode(eval_env, eval_mode=True)
        returns.append(ep_ret)
        lengths.append(ep_len)
    return {
        "eval/mean_return": float(np.mean(returns)),
        "eval/std_return":  float(np.std(returns)),
        "eval/mean_length": float(np.mean(lengths)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n[train] device={device}  total_steps={args['total_steps']:,}")

obs, _info    = train_env.reset()
total_steps   = 0
ep_num        = 0
ep_ret        = 0.0
ep_len        = 0
last_loss_sr: Optional[float] = None
last_loss_r:  Optional[float] = None

ret_window = deque(maxlen=100)
len_window = deque(maxlen=100)

best_eval_return: float = -np.inf

while total_steps < args["total_steps"]:

    # ── act ───────────────────────────────────────────────────────────────────
    s = agent.get_state_vec(obs)
    a = agent.sample_action(s, eval_mode=False)
    next_obs, reward, terminated, truncated, info = train_env.step(a)

    # ── store & learn ──────────────────────────────────────────────────────────
    loss = agent.observe_and_learn(obs, a, reward, next_obs, terminated, truncated)
    if loss is not None:
        last_loss_sr, last_loss_r = loss

    ep_ret      += float(reward)
    ep_len      += 1
    total_steps += 1
    obs          = next_obs

    # ── end of episode ─────────────────────────────────────────────────────────
    if terminated or truncated:
        ep_num += 1
        ret_window.append(ep_ret)
        len_window.append(ep_len)

        log_metrics(
            {"episode/return": ep_ret, "episode/length": ep_len},
            step=ep_num,
        )

        obs, _info = train_env.reset()
        ep_ret     = 0.0
        ep_len     = 0

    # ── periodic rollout metrics ───────────────────────────────────────────────
    if total_steps % args["log_freq"] == 0 and len(ret_window) > 0:
        metrics = {
            "rollout/ep_rew_mean": float(np.mean(ret_window)),
            "rollout/ep_len_mean": float(np.mean(len_window)),
            "rollout/epsilon":     agent.epsilon,
            "rollout/buffer_size": len(agent.rb),
        }
        if last_loss_sr is not None:
            metrics["train/loss_sr"] = last_loss_sr
            metrics["train/loss_r"]  = last_loss_r
        log_metrics(metrics, step=total_steps)

    # ── eval ──────────────────────────────────────────────────────────────────
    if total_steps % args["eval_freq"] == 0:
        eval_metrics = evaluate(args["eval_episodes"])
        log_metrics(eval_metrics, step=total_steps)
        print(
            f"[eval] step={total_steps:>8,}  "
            f"mean_return={eval_metrics['eval/mean_return']:+.3f}  "
            f"std={eval_metrics['eval/std_return']:.3f}  "
            f"eps={agent.epsilon:.3f}"
        )
        if eval_metrics["eval/mean_return"] > best_eval_return:
            best_eval_return = eval_metrics["eval/mean_return"]
            save_checkpoint("best")
            if experiment is not None:
                experiment.log_metric("eval/best_return", best_eval_return, step=total_steps)

    # ── checkpoint ────────────────────────────────────────────────────────────
    if total_steps % args["checkpoint_freq"] == 0:
        save_checkpoint(f"step{total_steps}")


# ──────────────────────────────────────────────────────────────────────────────
# Final save
# ──────────────────────────────────────────────────────────────────────────────
save_checkpoint("final")

if experiment is not None:
    experiment.end()

print("\n[done] training finished.")