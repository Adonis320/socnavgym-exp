from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor

class EpisodeLogCallback(BaseCallback):
    def __init__(self, comet_cb=None):
        super().__init__()
        self.comet_cb = comet_cb
        self.episode_idx = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is None:
                continue  # Monitor/VecMonitor not active

            self.episode_idx += 1
            ep_r = float(ep["r"])
            ep_l = int(ep["l"])

            # Log to SB3 logger
            self.logger.record("rollout/ep_rew", ep_r)
            self.logger.record("rollout/ep_len", ep_l)
            self.logger.record("rollout/episode", self.episode_idx)

            # Optional: also send to Comet if your callback exposes an experiment handle
            if self.comet_cb is not None and hasattr(self.comet_cb, "experiment"):
                self.comet_cb.experiment.log_metric("ep_rew", ep_r, step=self.episode_idx)
                self.comet_cb.experiment.log_metric("ep_len", ep_l, step=self.episode_idx)

        return True