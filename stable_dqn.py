import gymnasium as gym
import socnavgym
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import argparse
import numpy as np
from collections import deque

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--env_config", help="path to environment config", required=True)
ap.add_argument("-r", "--run_name", help="name of comet_ml run", required=True)
ap.add_argument("-s", "--save_path", help="path to save the model", required=True)
ap.add_argument("-p", "--project_name", required=False, default=None)
ap.add_argument("-a", "--api_key", required=False, default=None)
ap.add_argument("-d", "--use_deep_net", required=False, default=False)
ap.add_argument("-g", "--gpu", required=False, default="0")
args = vars(ap.parse_args())

experiment = None
if args["api_key"] is not None:
    from comet_ml import Experiment
    experiment = Experiment(api_key=args["api_key"], project_name=args["project_name"], parse_args=False)
    experiment.set_name(args["run_name"])
    print(f"[comet] logging to project '{args['project_name']}' as '{args['run_name']}'")
else:
    print("[info] no CometML api key – running without logging")


def log_metrics(metrics: dict, step: int):
    if experiment is not None:
        experiment.log_metrics(metrics, step=step)
    else:
        print(f"  step={step}  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))


class CometCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.ep_num = 0
        self.ep_ret = 0.0
        self.ep_len = 0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        dones   = self.locals.get("dones", [])

        reward = rewards[0] if hasattr(rewards, '__len__') else float(rewards)
        done   = dones[0]   if hasattr(dones,   '__len__') else bool(dones)

        self.ep_ret += float(reward)
        self.ep_len += 1

        if done:
            self.ep_num += 1
            log_metrics(
                {"episode/return": self.ep_ret, "episode/length": self.ep_len},
                step=self.ep_num,
            )
            self.ep_ret = 0.0
            self.ep_len = 0

        return True


from socnavgym.wrappers import DiscreteActions, ExpertObservations

env = gym.make("SocNavGym-v1", config=args["env_config"])
env = DiscreteActions(env)
env = ExpertObservations(env)

net_arch = [512, 256, 256, 256, 128, 128, 64] if args["use_deep_net"] else [512, 256, 128, 64]
policy_kwargs = {"net_arch": net_arch}

device = f"cuda:{args['gpu']}" if torch.cuda.is_available() else "cpu"
model = DQN("MultiInputPolicy", env, verbose=0, policy_kwargs=policy_kwargs, device=device)

model.learn(total_timesteps=100_000, callback=CometCallback())
model.save(args["save_path"])

if experiment is not None:
    experiment.end()