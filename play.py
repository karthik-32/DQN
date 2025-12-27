import time
import numpy as np
import gymnasium as gym
import gymnasium_env

import torch
import torch.nn as nn


def one_hot(state: int, n_states: int) -> np.ndarray:
    v = np.zeros(n_states, dtype=np.float32)
    v[state] = 1.0
    return v


class DQN(nn.Module):
    def __init__(self, n_states: int, n_actions: int, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def main():
    size = 15
    model_file = "double_dqn_grid_15.pt"

    env = gym.make("gymnasium_env/GridWorld-v0", size=size, render_mode="human")
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    model = DQN(n_states, n_actions, hidden=128)
    model.load_state_dict(torch.load(model_file, map_location="cpu"))
    model.eval()

    state, _ = env.reset()
    print("GridWorld opened. Starting in 2 seconds...")
    time.sleep(2)

    terminated = truncated = False
    steps = 0
    max_steps = size * size * 3

    while not (terminated or truncated) and steps < max_steps:
        steps += 1
        s_vec = one_hot(state, n_states)

        with torch.no_grad():
            qs = model(torch.from_numpy(s_vec).unsqueeze(0))

        # tie-break among equal max actions
        q_np = qs.numpy().reshape(-1)
        max_q = q_np.max()
        best_actions = np.flatnonzero(q_np == max_q)
        action = int(np.random.choice(best_actions))

        state, reward, terminated, truncated, _ = env.step(action)
        time.sleep(0.15)

    print("Done. Close the window to exit.")
    while True:
        env.render()
        time.sleep(0.05)


if __name__ == "__main__":
    main()
