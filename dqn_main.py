import random
import numpy as np
import gymnasium as gym
import gymnasium_env

from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, done = zip(*batch)
        return (
            np.array(s, dtype=np.int64),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(s2, dtype=np.int64),
            np.array(done, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


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


def train_double_dqn(
    episodes=8000,              # ✅ 2000 is often too low for 30x30; start 8000
    size=30,
    model_path="double_dqn_grid_30.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    max_steps = size * size * 6  # ✅ 5400 steps
    env = gym.make("gymnasium_env/GridWorld-v0", size=size, render_mode=None, max_steps=max_steps)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    policy_net = DQN(n_states, n_actions, emb_dim=64, hidden=256).to(device)
    target_net = DQN(n_states, n_actions, emb_dim=64, hidden=256).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=8e-4)
    loss_fn = nn.SmoothL1Loss()

    buffer = ReplayBuffer(capacity=200_000)

    gamma = 0.99
    batch_size = 256
    warmup = 3000

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.9995

    target_update_steps = 500
    step_count = 0

    success_window = deque(maxlen=100)

    print("🏋️ Double DQN training started...", flush=True)

    for ep in range(episodes):
        state, _ = env.reset()
        terminated = truncated = False
        ep_reward = 0.0

        while not (terminated or truncated):
            step_count += 1

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s_t = torch.tensor([state], dtype=torch.long, device=device)
                    q = policy_net(s_t)[0]
                    action = int(torch.argmax(q).item())

            new_state, reward, terminated, truncated, _ = env.step(action)
            done = float(terminated or truncated)

            buffer.push(state, action, reward, new_state, done)

            state = new_state
            ep_reward += reward

            if len(buffer) >= max(warmup, batch_size):
                s_b, a_b, r_b, s2_b, done_b = buffer.sample(batch_size)

                s_b = torch.from_numpy(s_b).to(device)
                a_b = torch.from_numpy(a_b).to(device).unsqueeze(1)
                r_b = torch.from_numpy(r_b).to(device)
                s2_b = torch.from_numpy(s2_b).to(device)
                done_b = torch.from_numpy(done_b).to(device)

                q_sa = policy_net(s_b).gather(1, a_b).squeeze(1)

                with torch.no_grad():
                    next_actions = policy_net(s2_b).argmax(dim=1, keepdim=True)
                    next_q = target_net(s2_b).gather(1, next_actions).squeeze(1)
                    target = r_b + gamma * next_q * (1.0 - done_b)

                loss = loss_fn(q_sa, target)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 5.0)
                optimizer.step()

            if step_count % target_update_steps == 0:
                target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        success_window.append(1 if terminated else 0)

        if (ep + 1) % 50 == 0:
            sr = sum(success_window) / len(success_window)
            print(
                f"Episode {ep+1}/{episodes} | reward: {ep_reward:.1f} | eps: {epsilon:.3f} | success(100): {sr:.2f}",
                flush=True
            )

    env.close()
    torch.save(policy_net.state_dict(), model_path)
    print(f"✅ Training finished. Saved model to: {model_path}", flush=True)
    print("Now run: python play.py", flush=True)


if __name__ == "__main__":
    train_double_dqn()
