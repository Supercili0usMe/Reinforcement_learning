#!/usr/bin/env python3
import gymnasium as gym
import ale_py
import time
import argparse
import numpy as np

import torch

from lib import wrappers
from lib import dqn_model

import collections

# Регистрируем Atari окружения
gym.register_envs(ale_py)

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
FPS = 25


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-e", "--env", default=DEFAULT_ENV_NAME,
                        help="Environment name to use, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("-r", "--record", help="Directory to store video recording")
    parser.add_argument("--no-visualize", default=True, action='store_false', dest='visualize',
                        help="Disable visualization of the game play")
    args = parser.parse_args()

    # Определяем render_mode
    if args.record:
        render_mode = "rgb_array"  # Для записи видео нужен rgb_array
    elif args.visualize:
        render_mode = "human"
    else:
        render_mode = None
    
    env = wrappers.make_env(args.env, render_mode=render_mode)
    
    # Для записи видео используем RecordVideo вместо устаревшего Monitor
    if args.record:
        env = gym.wrappers.RecordVideo(env, args.record)
    
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
    net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage, weights_only=True))

    # Новый API gymnasium: reset() возвращает (obs, info)
    state, _ = env.reset()
    total_reward = 0.0
    c = collections.Counter()

    while True:
        start_ts = time.time()
        state_v = torch.tensor(np.asarray([state]), dtype=torch.float32)
        q_vals = net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)
        c[action] += 1
        
        # Новый API gymnasium: step() возвращает (obs, reward, terminated, truncated, info)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
            
        if args.visualize:
            delta = 1/FPS - (time.time() - start_ts)
            if delta > 0:
                time.sleep(delta)
                
    print("Total reward: %.2f" % total_reward)
    print("Action counts:", c)
    env.close()
