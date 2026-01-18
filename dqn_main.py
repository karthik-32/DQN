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
            np.stack(s).astype(np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.stack(s2).astype(np.float32),
            np.array(done, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def flatten_obs(obs: np.ndarray) -> np.ndarray:
    return obs.reshape(-1).astype(np.float32)


def train_fast_dqn(
    episodes=8000,
    size=30,
    model_path="fast_dqn_dynamic_30.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    # ✅ BIGGEST SPEED WIN: cap max steps
    max_steps = 600  # was 1800; keeps training fast
    env = gym.make("gymnasium_env/GridWorld-v0", size=size, render_mode=None, max_steps=max_steps)

    obs0, _ = env.reset()
    if not hasattr(obs0, "shape"):
        raise RuntimeError("Env returned int. You are not using the Box-observation env.")

    obs_dim = obs0.size
    n_actions = env.action_space.n

    policy_net = DQN(obs_dim, n_actions, hidden=128).to(device)
    target_net = DQN(obs_dim, n_actions, hidden=128).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    loss_fn = nn.SmoothL1Loss()

    buffer = ReplayBuffer(capacity=60_000)

    gamma = 0.99
    batch_size = 64
    warmup = 800

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.9993

    target_update_steps = 500
    step_count = 0

    train_every = 8  # ✅ fewer updates = faster

    success_window = deque(maxlen=100)

    print("🏋️ FAST DQN training (dynamic obstacles) started...", flush=True)

    for ep in range(episodes):
        obs, _ = env.reset()
        s_vec = flatten_obs(obs)

        terminated = truncated = False
        ep_reward = 0.0

        while not (terminated or truncated):
            step_count += 1

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s_t = torch.from_numpy(s_vec).unsqueeze(0).to(device)
                    q = policy_net(s_t)[0]
                    action = int(torch.argmax(q).item())

            obs2, reward, terminated, truncated, _ = env.step(action)
            s2_vec = flatten_obs(obs2)
            done = float(terminated or truncated)

            buffer.push(s_vec, action, reward, s2_vec, done)
            s_vec = s2_vec
            ep_reward += reward

            # ✅ train only every N steps
            if (step_count % train_every == 0) and len(buffer) >= max(warmup, batch_size):
                s_b, a_b, r_b, s2_b, done_b = buffer.sample(batch_size)

                s_b = torch.from_numpy(s_b).to(device)
                a_b = torch.from_numpy(a_b).to(device).unsqueeze(1)
                r_b = torch.from_numpy(r_b).to(device)
                s2_b = torch.from_numpy(s2_b).to(device)
                done_b = torch.from_numpy(done_b).to(device)

                q_sa = policy_net(s_b).gather(1, a_b).squeeze(1)

                # ✅ FAST DQN target (NOT double dqn)
                with torch.no_grad():
                    max_next_q = target_net(s2_b).max(dim=1).values
                    target = r_b + gamma * max_next_q * (1.0 - done_b)

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
                f"Ep {ep+1}/{episodes} | reward {ep_reward:.1f} | eps {epsilon:.3f} | success(100) {sr:.2f}",
                flush=True
            )

    env.close()
    torch.save(policy_net.state_dict(), model_path)
    print(f"✅ Training finished. Saved model to: {model_path}", flush=True)
    print("Now run: python play.py", flush=True)


if __name__ == "__main__":
    train_fast_dqn()
