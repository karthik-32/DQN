import time
import numpy as np
import gymnasium as gym
import gymnasium_env
import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, n_states: int, n_actions: int, emb_dim=64, hidden=256):
        super().__init__()
        self.emb = nn.Embedding(n_states, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, s_idx: torch.Tensor):
        x = self.emb(s_idx)
        return self.net(x)


def main():
    size = 30
    model_file = "double_dqn_grid_30.pt"

    env = gym.make(
        "gymnasium_env/GridWorld-v0",
        size=size,
        render_mode="human",
        max_steps=size * size * 6,  # ✅ enough steps
    )

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    model = DQN(n_states, n_actions, emb_dim=64, hidden=256)
    model.load_state_dict(torch.load(model_file, map_location="cpu"))
    model.eval()

    state, _ = env.reset()
    time.sleep(1)

    terminated = truncated = False
    steps = 0
    no_move_streak = 0

    while not (terminated or truncated):
        steps += 1

        with torch.no_grad():
            s_t = torch.tensor([state], dtype=torch.long)
            q = model(s_t)[0].numpy()

        best_actions = np.flatnonzero(q == q.max())
        action = int(np.random.choice(best_actions))

        next_state, reward, terminated, truncated, _ = env.step(action)

        if next_state == state:
            no_move_streak += 1
        else:
            no_move_streak = 0

        state = next_state
        time.sleep(0.06)

        if no_move_streak >= 40:
            print("⚠️ Agent stuck (no movement for 40 steps). Ending run.")
            break

    print(f"Steps={steps} | terminated={terminated} | truncated={truncated}")
    print("Close the window to exit.")

    while True:
        env.render()
        time.sleep(0.05)


if __name__ == "__main__":
    main()
