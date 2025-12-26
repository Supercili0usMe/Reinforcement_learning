import random

class Environment:
    def __init__(self):
        """Инициализируем внутреннее состояние среды"""
        self.steps_left = 10
    
    def get_observation(self):
        '''Возвращаем текущее наблюдение среды агенту'''
        return [0.0, 0.0, 0.0, 0.0]
    
    def get_actions(self):
        '''Позволяем агенту запросить набор действий, которые он может выполнить'''
        return [0, 1]
    
    def is_done(self):
        '''Сигнализируем агенту об окончании эпизода'''
        return self.steps_left == 0
    
    def action(self, action):
        '''Основной функциональный элемент среды: идентифицирует действие агента и возвращает вознаграждение за это действие'''
        if self.is_done():
            raise Exception("Игра окончена")
        self.steps_left -= 1
        return random.random()
    

class Agent:
    def __init__(self):
        '''Суммируем награду агента в течении всего эпизода'''
        self.total_reward = 0.0

    def step(self, env):
        '''Агент наблюдает за средой, принимает решения о выборе действий,
        выполняет действия в среде, получает награду за текущий шаг'''
        current_obs = env.get_observation()
        action = env.get_actions()
        reward = env.action(random.choice(action))
        self.total_reward += reward
    
if __name__ == "__main__":
    env = Environment()
    agent = Agent()

    while not env.is_done():
        agent.step(env)
    
    print(f"Общая награда: {agent.total_reward:.4f}")
