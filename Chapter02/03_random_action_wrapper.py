import gym, random

class RandomActionWrapper(gym.ActionWrapper):
    def __init__(self, env, epsilon=0.7):
        super(RandomActionWrapper, self).__init__(env)
        self.epsilon = epsilon

    def action(self, action):
        if random.random() < self.epsilon:
            print("random!")
            return self.env.action_space.sample()
        return action
    
if __name__ == "__main__":
    env = RandomActionWrapper(gym.make("CartPole-v0"))

    obs = env.reset()
    total_reward = 0.0

    while True:
        obs, reward, done, _, _ = env.step(0)
        total_reward += reward
        if done:
            break
    
    print(f"Общая награда: {total_reward}")