"""
Совместимый с gymnasium заменитель ptan для Python 3.12+
Реализует основные классы, используемые в проекте.
"""
import numpy as np
import torch
from collections import namedtuple, deque
from typing import List, Optional, Tuple, Any


# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'last_state'])


class EpsilonGreedyActionSelector:
    """Epsilon-greedy action selector"""
    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon

    def __call__(self, q_values: np.ndarray) -> np.ndarray:
        """Select actions based on Q-values with epsilon-greedy strategy"""
        batch_size = q_values.shape[0]
        actions = np.argmax(q_values, axis=1)
        
        # Apply epsilon-greedy
        mask = np.random.random(batch_size) < self.epsilon
        random_actions = np.random.randint(0, q_values.shape[1], size=batch_size)
        actions[mask] = random_actions[mask]
        
        return actions


class TargetNet:
    """Target network wrapper for DQN"""
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.target_model = self._create_target()

    def _create_target(self) -> torch.nn.Module:
        """Create a copy of the model for target network"""
        import copy
        target = copy.deepcopy(self.model)
        for param in target.parameters():
            param.requires_grad = False
        return target

    def sync(self):
        """Synchronize target network with main network"""
        self.target_model.load_state_dict(self.model.state_dict())


class DQNAgent:
    """DQN Agent that interacts with the environment"""
    def __init__(self, model: torch.nn.Module, action_selector: EpsilonGreedyActionSelector, 
                 device: str = "cpu"):
        self.model = model
        self.action_selector = action_selector
        self.device = device

    @torch.no_grad()
    def __call__(self, states: List[np.ndarray]) -> Tuple[np.ndarray, List[None]]:
        """
        Select actions for given states
        Returns: (actions, agent_states) where agent_states is always None for DQN
        """
        states_v = torch.tensor(np.array(states, dtype=np.float32)).to(self.device)
        q_values = self.model(states_v).cpu().numpy()
        actions = self.action_selector(q_values)
        return actions, [None] * len(states)


class ExperienceSourceFirstLast:
    """
    Experience source that provides (state, action, reward, last_state) tuples.
    Uses n-step returns.
    """
    def __init__(self, env, agent: DQNAgent, gamma: float, steps_count: int = 1):
        self.env = env
        self.agent = agent
        self.gamma = gamma
        self.steps_count = steps_count
        
        self._rewards_steps = []
        self._reset()

    def _reset(self):
        """Reset environment and internal state"""
        result = self.env.reset()
        if isinstance(result, tuple):
            self._state, _ = result
        else:
            self._state = result
        self._history = deque(maxlen=self.steps_count)
        self._cur_reward = 0.0
        self._cur_steps = 0

    def pop_rewards_steps(self) -> List[Tuple[float, int]]:
        """Pop accumulated rewards and steps"""
        res = self._rewards_steps
        self._rewards_steps = []
        return res

    def __iter__(self):
        return self

    def __next__(self) -> Experience:
        """Generate next experience"""
        while True:
            # Select action
            actions, _ = self.agent([self._state])
            action = actions[0]
            
            # Step environment
            result = self.env.step(action)
            if len(result) == 5:
                next_state, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                next_state, reward, done, info = result
            
            self._cur_reward += reward
            self._cur_steps += 1
            
            # Store in history for n-step
            self._history.append((self._state, action, reward))
            
            if len(self._history) == self.steps_count:
                # Calculate n-step reward
                total_reward = 0.0
                for idx, (s, a, r) in enumerate(self._history):
                    total_reward += r * (self.gamma ** idx)
                
                first_state = self._history[0][0]
                first_action = self._history[0][1]
                
                if done:
                    last_state = None
                else:
                    last_state = next_state
                
                exp = Experience(first_state, first_action, total_reward, last_state)
                
                if done:
                    self._rewards_steps.append((self._cur_reward, self._cur_steps))
                    self._reset()
                else:
                    self._state = next_state
                
                return exp
            
            if done:
                # Episode ended before we filled the history
                if len(self._history) > 0:
                    total_reward = 0.0
                    for idx, (s, a, r) in enumerate(self._history):
                        total_reward += r * (self.gamma ** idx)
                    
                    first_state = self._history[0][0]
                    first_action = self._history[0][1]
                    
                    exp = Experience(first_state, first_action, total_reward, None)
                    self._rewards_steps.append((self._cur_reward, self._cur_steps))
                    self._reset()
                    return exp
                else:
                    self._rewards_steps.append((self._cur_reward, self._cur_steps))
                    self._reset()
            else:
                self._state = next_state


class ExperienceReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, experience_source: ExperienceSourceFirstLast, buffer_size: int):
        self.experience_source = experience_source
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self._experience_iter = iter(experience_source)

    def __len__(self) -> int:
        return len(self.buffer)

    def populate(self, count: int):
        """Populate buffer with given number of experiences"""
        for _ in range(count):
            try:
                exp = next(self._experience_iter)
                self.buffer.append(exp)
            except StopIteration:
                self._experience_iter = iter(self.experience_source)

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch from buffer"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]

