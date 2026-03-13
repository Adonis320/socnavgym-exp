# test_cartpole.py
# Validates DSR agent on CartPole-v1.
# If episode returns climb from ~20 to ~400+ the learning pipeline is working.
#
# Usage:
#   python3 test_cartpole.py
#   python3 test_cartpole.py -a <COMET_API_KEY> -p my-project -r cartpole-dsr

from __future__ import annotations

try:
    from comet_ml import Experiment as CometExperiment
except ImportError:
    CometExperiment = None

import argparse
import types
from collections import deque
from typing import Optional

import numpy as np
import torch
import gymnasium as gym

from rl.DSR import DSRAgent, DSRConfig


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("-r", "--run_name",     required=False, default="cartpole-dsr")
ap.add_argument("-p", "--project_name", required=False, default=None)
ap.add_argument("-a", "--api_key",      required=False, default=None)
ap.add_argument("--total_steps",        required=False, default=100_000, type=int)
ap.add_argument("--log_freq",           required=False, default=1_000,   type=int)
ap.add_argument("--eval_freq",          required=False, default=5_000,   type=int)
ap.add_argument("--eval_episodes",      required=False, default=10,      type=int)
args = vars(ap.parse_args())


# ──────────────────────────────────────────────────────────────────────────────
# CometML
# ──────────────────────────────────────────────────────────────────────────────
experiment = None
if args["api_key"] is not None:
    if CometExperiment is None:
        raise ImportError("pip install comet_ml")
    experiment = CometExperiment(
        api_key=args["api_key"],
        project_name=args["project_name"],
        parse_args=False,
    )
    experiment.set_name(args["run_name"])
    print(f"[comet] project='{args['project_name']}' run='{args['run_name']}'")
else:
    print("[info] no CometML key — logging to stdout")


def log_metrics(metrics: dict, step: int):
    if experiment is not None:
        experiment.log_metrics(metrics, step=step)
    else:
        print(f"  step={step:>8}  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))


# ──────────────────────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────────────────────
train_env = gym.make("CartPole-v1")
eval_env  = gym.make("CartPole-v1")


# ──────────────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────────────────
config = DSRConfig(
    state_dim            = 4,        # CartPole: [cart_pos, cart_vel, pole_angle, pole_vel]
    feat_dim             = 32,
    action_size          = 2,
    map_size             = 1.0,      # CartPole values already small
    sr_hidden            = 128,
    learning_rate        = 1e-4,
    gamma                = 0.99,
    epsilon_start        = 1.0,
    epsilon_end          = 0.05,
    epsilon_decay_steps  = 10_000,   # 10% of 100k total steps
    target_update_freq   = 500,
    batch_size           = 32,
    start_steps          = 1_000,
    buffer_size          = 50_000,
    train_freq           = 4,
    lambda_sr            = 1.0,
    lambda_r             = 1.0,
    grad_clip            = 1.0,
    device               = "cpu",
)

agent = DSRAgent(config)

if experiment is not None:
    experiment.log_parameters(vars(config))

# ── Patch get_state_vec for flat CartPole obs ─────────────────────────────────
def cartpole_state_vec(self, obs):
    if isinstance(obs, (tuple, list)) and not isinstance(obs[0], float):
        obs = obs[0]
    return np.asarray(obs, dtype=np.float32).flatten()

agent.get_state_vec = types.MethodType(cartpole_state_vec, agent)

if experiment is not None:
    experiment.log_parameters(vars(config))

# ── Patch get_state_vec for flat CartPole obs ─────────────────────────────────
# CartPole returns a flat np.ndarray [cart_pos, cart_vel, pole_angle, pole_vel]
# The default get_state_vec expects a dict — this patch handles the flat array.
def cartpole_state_vec(self, obs):
    if isinstance(obs, (tuple, list)) and not isinstance(obs[0], float):
        obs = obs[0]
    return np.asarray(obs, dtype=np.float32).flatten()

agent.get_state_vec = types.MethodType(cartpole_state_vec, agent)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────
def evaluate(n_episodes: int = 10) -> dict:
    returns, lengths = [], []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        ep_ret, ep_len = 0.0, 0
        while True:
            s = agent.get_state_vec(obs)
            a = agent.sample_action(s, eval_mode=True)
            obs, reward, terminated, truncated, _ = eval_env.step(a)
            ep_ret += float(reward)
            ep_len += 1
            if terminated or truncated:
                break
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
print(f"\n[train] CartPole-v1  total_steps={args['total_steps']:,}")
print(f"        Expect returns to climb from ~20 → ~400 if agent is learning\n")

obs, _      = train_env.reset()
total_steps = 0
ep_num      = 0
ep_ret      = 0.0
ep_len      = 0
last_loss_sr: Optional[float] = None
last_loss_r:  Optional[float] = None

ret_window = deque(maxlen=100)
len_window = deque(maxlen=100)

best_eval_return: float = -np.inf

while total_steps < args["total_steps"]:

    s = agent.get_state_vec(obs)
    a = agent.sample_action(s, eval_mode=False)
    next_obs, reward, terminated, truncated, _ = train_env.step(a)

    loss = agent.observe_and_learn(obs, a, reward, next_obs, terminated, truncated)
    if loss is not None:
        last_loss_sr, last_loss_r = loss

    ep_ret      += float(reward)
    ep_len      += 1
    total_steps += 1
    obs          = next_obs

    if terminated or truncated:
        ep_num += 1
        ret_window.append(ep_ret)
        len_window.append(ep_len)

        log_metrics(
            {"episode/return": ep_ret, "episode/length": ep_len},
            step=ep_num,
        )

        obs, _ = train_env.reset()
        ep_ret = 0.0
        ep_len = 0

    if total_steps % args["log_freq"] == 0 and len(ret_window) > 0:
        metrics = {
            "rollout/ep_rew_mean": float(np.mean(ret_window)),
            "rollout/ep_len_mean": float(np.mean(len_window)),
            "rollout/buffer_size": len(agent.rb),
            "rollout/epsilon":     agent.epsilon,
        }
        if last_loss_sr is not None:
            metrics["train/loss_sr"] = last_loss_sr
            metrics["train/loss_r"]  = last_loss_r
        log_metrics(metrics, step=total_steps)

    if total_steps % args["eval_freq"] == 0:
        eval_metrics = evaluate(args["eval_episodes"])
        log_metrics(eval_metrics, step=total_steps)
        print(
            f"[eval]  step={total_steps:>8,}  "
            f"mean_return={eval_metrics['eval/mean_return']:>6.1f}  "
            f"std={eval_metrics['eval/std_return']:.1f}  "
            f"eps={agent.epsilon:.3f}"
        )
        if eval_metrics["eval/mean_return"] > best_eval_return:
            best_eval_return = eval_metrics["eval/mean_return"]
            print(f"[best]  new best return: {best_eval_return:.1f}")
            if experiment is not None:
                experiment.log_metric("eval/best_return", best_eval_return, step=total_steps)

if experiment is not None:
    experiment.end()