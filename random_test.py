import time
import gymnasium as gym
import gymnasium_env

env = gym.make("gymnasium_env/GridWorld-v0", size=4, render_mode="human")
s, _ = env.reset()

for i in range(50):
    a = env.action_space.sample()
    s, r, term, trunc, _ = env.step(a)
    time.sleep(0.2)
    if term or trunc:
        break

env.close()
