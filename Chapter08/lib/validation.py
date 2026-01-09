import numpy as np

import torch

from lib import environ


def validation_run(env, net, episodes=100, device="cpu", epsilon=0.02, comission=0.1):
    stats = {
        'episode_reward': [],
        'episode_steps': [],
        'order_profits': [],
        'order_steps': [],
    }

    for episode in range(episodes):
        result = env.reset()
        if isinstance(result, tuple):
            obs, _ = result
        else:
            obs = result

        total_reward = 0.0
        position = None
        position_steps = None
        episode_steps = 0

        while True:
            obs_v = torch.tensor(np.array([obs]), dtype=torch.float32).to(device)
            out_v = net(obs_v)

            action_idx = out_v.max(dim=1)[1].item()
            if np.random.random() < epsilon:
                action_idx = env.action_space.sample()
            action = environ.Actions(action_idx)

            close_price = env._state._cur_close()

            if action == environ.Actions.Buy and position is None:
                position = close_price
                position_steps = 0
            elif action == environ.Actions.Close and position is not None:
                profit = close_price - position - (close_price + position) * comission / 100
                profit = 100.0 * profit / position
                stats['order_profits'].append(profit)
                stats['order_steps'].append(position_steps)
                position = None
                position_steps = None

            # gymnasium API: obs, reward, terminated, truncated, info
            result = env.step(action_idx)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, reward, done, info = result
            
            total_reward += reward
            episode_steps += 1
            if position_steps is not None:
                position_steps += 1
            if done:
                if position is not None:
                    profit = close_price - position - (close_price + position) * comission / 100
                    profit = 100.0 * profit / position
                    stats['order_profits'].append(profit)
                    stats['order_steps'].append(position_steps)
                break

        stats['episode_reward'].append(total_reward)
        stats['episode_steps'].append(episode_steps)

    return { key: np.mean(vals) for key, vals in stats.items() }
