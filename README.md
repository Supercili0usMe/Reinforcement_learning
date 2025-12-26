Для запуска Tensorboard:
```
tensorboard --logdir runs
```

---
Старый `gym` заменен на `gymnasium`. Поэтому код потребует некоторых изменений:
```python
# Было:
import gym
env = gym.make("PongNoFrameskip-v4")

# Стало:
import gymnasium as gym
env = gym.make("ALE/Pong-v5")
```