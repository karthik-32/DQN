import random
import numpy as np
import gymnasium as gym
import gymnasium_env

from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


def one_hot(state: int, n_states: int) -> np.ndarray:
    v = np.zeros(n_states, dtype=np.float32)
    v[state] = 1.0
    return v


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, done = zip(*batch)
        return (
            np.stack(s).astype(np.float32),    # ✅ stack = fast
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.stack(s2).astype(np.float32),   # ✅ stack = fast
            np.array(done, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


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


def train_double_dqn(
    episodes=8000,
    size=15,
    model_path="double_dqn_grid_15.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    env = gym.make("gymnasium_env/GridWorld-v0", size=size, render_mode=None)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    policy_net = DQN(n_states, n_actions, hidden=128).to(device)
    target_net = DQN(n_states, n_actions, hidden=128).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    loss_fn = nn.SmoothL1Loss()

    buffer = ReplayBuffer(capacity=100_000)

    gamma = 0.99
    batch_size = 64
    warmup = 1000

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.999

    max_steps_per_episode = size * size * 3  # 675
    target_update_steps = 300
    step_count = 0

    print("🏋️ Double DQN training started...", flush=True)

    for ep in range(episodes):
        state, _ = env.reset()
        s_vec = one_hot(state, n_states)

        terminated = truncated = False
        steps = 0
        ep_reward = 0.0

        while not (terminated or truncated) and steps < max_steps_per_episode:
            steps += 1
            step_count += 1

            # epsilon-greedy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    # ✅ FAST: from_numpy instead of torch.tensor([s_vec])
                    s_t = torch.from_numpy(s_vec).unsqueeze(0).to(device)
                    qs = policy_net(s_t)
                    action = int(torch.argmax(qs, dim=1).item())

            new_state, reward, terminated, truncated, _ = env.step(action)
            s2_vec = one_hot(new_state, n_states)
            done = float(terminated or truncated)

            buffer.push(s_vec, action, reward, s2_vec, done)

            s_vec = s2_vec
            ep_reward += reward

            # Learn
            if len(buffer) >= max(warmup, batch_size):
                s_b, a_b, r_b, s2_b, done_b = buffer.sample(batch_size)

                # ✅ FAST batch conversion
                s_b = torch.from_numpy(s_b).to(device)
                a_b = torch.from_numpy(a_b).to(device).unsqueeze(1)
                r_b = torch.from_numpy(r_b).to(device)
                s2_b = torch.from_numpy(s2_b).to(device)
                done_b = torch.from_numpy(done_b).to(device)

                # Q(s,a) from policy net
                q_sa = policy_net(s_b).gather(1, a_b).squeeze(1)

                # Double DQN target
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

        # print progress more often so you KNOW it's training
        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{episodes} | reward: {ep_reward:.3f} | epsilon: {epsilon:.3f}", flush=True)

    env.close()

    torch.save(policy_net.state_dict(), model_path)
    print(f"✅ Training finished. Saved model to: {model_path}", flush=True)
    print("Now run: python play.py", flush=True)


if __name__ == "__main__":
    train_double_dqn(
        episodes=8000,
        size=15,
        model_path="double_dqn_grid_15.pt",
    )
