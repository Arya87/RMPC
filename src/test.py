import warnings

warnings.filterwarnings('ignore')
import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
import torch
import numpy as np
import gym

gym.logger.set_level(40)
import time
import random
from pathlib import Path
from cfg import parse_cfg
from env import make_env
from algorithm.tdmpc import TDMPC
from algorithm.helper import Episode, ReplayBuffer
import logger

torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__ = 'cfgs', 'logs'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(env, agent, num_episodes, step, env_step, video):
    """Evaluate a trained agent and optionally save a video."""
    episode_rewards = []
    for i in range(num_episodes):
        obs, done, ep_reward, t = env.reset(), False, 0, 0
        if video: video.init(env, enabled=(i == 0))
        while not done:
            action = agent.plan(obs, eval_mode=True, step=step, t0=t == 0)
            obs, reward, done, _ = env.step(action.cpu().numpy())
            ep_reward += reward
            if video: video.record(env)
            t += 1
        episode_rewards.append(ep_reward)
        if video: video.save(env_step)
    return np.nanmean(episode_rewards)


def test(cfg):
    """Training script for TD-MPC. Requires a CUDA-enabled device."""
    assert torch.cuda.is_available()
    set_seed(cfg.seed)
    work_dir = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
    env, agent, buffer = make_env(cfg), TDMPC(cfg), ReplayBuffer(cfg)
    agent.load('/home/zhangfeng/liyaohui/TDMPC_DrQ/logs/humanoid-run/state/aug_Q,z,r_3M/1/models/model.pt')

    # Run training
    L = logger.Logger(work_dir, cfg)
    episode_idx, start_time = 0, time.time()

    for step in range(0, cfg.train_steps + cfg.episode_length, cfg.episode_length):

        obs = env.reset()
        episode = Episode(cfg, obs)




        # Log training episode
        episode_idx += 1
        env_step = int(step * cfg.action_repeat)
        common_metrics = {
            'episode': episode_idx,
            'step': step,
            'env_step': env_step,
            'total_time': time.time() - start_time,
            'episode_reward': episode.cumulative_reward}

        # Evaluate agent periodically
        common_metrics['episode_reward'] = evaluate(env, agent, cfg.eval_episodes, step, env_step, L.video)
        L.log(common_metrics, category='eval')

    L.finish(agent)
    print('Training completed successfully')


if __name__ == '__main__':
    test(parse_cfg(Path().cwd() / __CONFIG__))
