import gym, warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    env = gym.make('CartPole-v0')
    total_reward = 0.0
    total_step = 0
    obs = env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        total_step += 1
        if done: break
    
    print(f"Эпизод закончен за {total_step} шагов, с суммарной наградой {total_reward}")

# Отключить все предупреждения в python scripts === python -W ignore foo.py
# d:/Project/Reinforcement_learning/.venv/Scripts/python.exe -W ignore d:/Project/Reinforcement_learning/Chapter02/02_cartpole_random.py