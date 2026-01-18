import os
os.environ["SDL_VIDEO_CENTERED"] = "1"
os.environ["SDL_HINT_VIDEO_MINIMIZE_ON_FOCUS_LOSS"] = "0"

import gymnasium as gym
from gymnasium import spaces
import numpy as np

try:
    import pygame
except ImportError:
    pygame = None


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, size=30, render_mode=None, max_steps=None):
        super().__init__()
        self.size = int(size)
        self.render_mode = render_mode

        # Observation: (3, size, size) float32
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3, self.size, self.size),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)

        self.start_pos = (1, 0)
        self.goal_pos = (self.size - 1, self.size - 1)

        self.steps = 0
        self.max_steps = max_steps if max_steps is not None else (self.size * self.size)

        self.static_obstacles = set(self._static_obstacles_fixed())
        self.static_obstacles.discard(self.start_pos)
        self.static_obstacles.discard(self.goal_pos)

        self.dynamic_specs = self._init_dynamic_obstacles()
        self.dynamic_positions = [sp["pos"] for sp in self.dynamic_specs]

        self.agent_pos = self.start_pos

        # pygame
        self.window_size = 750
        self.screen = None
        self.clock = None

    # ---------------- helpers ----------------
    def _in_bounds(self, r, c):
        return 0 <= r < self.size and 0 <= c < self.size

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _dynamic_cells_set(self):
        return set(self.dynamic_positions)

    def _get_obs(self):
        obs = np.zeros((3, self.size, self.size), dtype=np.float32)

        ar, ac = self.agent_pos
        gr, gc = self.goal_pos

        obs[0, ar, ac] = 1.0
        obs[1, gr, gc] = 1.0

        for (r, c) in self.static_obstacles:
            obs[2, r, c] = 1.0
        for (r, c) in self.dynamic_positions:
            obs[2, r, c] = 1.0

        return obs

    def _static_obstacles_fixed(self):
        obs = []

        for r in range(3, 6):
            for c in range(7, 12):
                obs.append((r, c))
        for r in range(3, 6):
            for c in range(18, 23):
                obs.append((r, c))

        obs += [(10, 3), (11, 3), (11, 4)]
        obs += [(9, 4), (15, 25)]
        obs += [(22, 13), (22, 14), (23, 13), (23, 14)]

        for r in range(11, 15):
            obs += [(r, 9), (r, 10)]
            obs += [(r, 20), (r, 21)]

        for r in range(18, 23):
            obs += [(r, 9), (r, 10)]
            obs += [(r, 20), (r, 21)]

        return obs

    def _init_dynamic_obstacles(self):
        # 10 deterministic patrol obstacles
        specs = [
            {"pos": (2, 2),  "axis": "h", "dir": +1, "min": 2,  "max": 12},
            {"pos": (6, 24), "axis": "h", "dir": -1, "min": 16, "max": 27},
            {"pos": (8, 6),  "axis": "h", "dir": +1, "min": 4,  "max": 14},
            {"pos": (16, 5), "axis": "h", "dir": +1, "min": 3,  "max": 11},
            {"pos": (24, 18),"axis": "h", "dir": -1, "min": 14, "max": 25},

            {"pos": (4, 15), "axis": "v", "dir": +1, "min": 4,  "max": 12},
            {"pos": (12, 26),"axis": "v", "dir": +1, "min": 10, "max": 22},
            {"pos": (17, 2), "axis": "v", "dir": -1, "min": 8,  "max": 18},
            {"pos": (20, 12),"axis": "v", "dir": +1, "min": 16, "max": 28},
            {"pos": (10, 17),"axis": "v", "dir": +1, "min": 7,  "max": 19},
        ]

        cleaned = []
        for sp in specs:
            if sp["pos"] in self.static_obstacles or sp["pos"] == self.start_pos or sp["pos"] == self.goal_pos:
                r, c = sp["pos"]
                sp["pos"] = (r, min(self.size - 2, c + 1))
            cleaned.append(sp)
        return cleaned

    def _move_dynamic_obstacles(self):
        occupied = set(self.dynamic_positions)
        new_positions = list(self.dynamic_positions)

        for i, spec in enumerate(self.dynamic_specs):
            r, c = self.dynamic_positions[i]
            axis = spec["axis"]
            d = spec["dir"]

            if axis == "h":
                nc = c + d
                if nc < spec["min"] or nc > spec["max"]:
                    spec["dir"] *= -1
                    d = spec["dir"]
                    nc = c + d
                cand = (r, nc)
            else:
                nr = r + d
                if nr < spec["min"] or nr > spec["max"]:
                    spec["dir"] *= -1
                    d = spec["dir"]
                    nr = r + d
                cand = (nr, c)

            # bounce off static obstacles/bounds
            if (not self._in_bounds(cand[0], cand[1])) or (cand in self.static_obstacles):
                spec["dir"] *= -1
                d = spec["dir"]
                cand = (r, c + d) if axis == "h" else (r + d, c)

            if (not self._in_bounds(cand[0], cand[1])) or (cand in self.static_obstacles):
                cand = (r, c)

            if cand == self.agent_pos:
                cand = (r, c)

            if cand in occupied and cand != (r, c):
                cand = (r, c)

            occupied.remove((r, c))
            occupied.add(cand)
            new_positions[i] = cand

        self.dynamic_positions = new_positions

    # ---------------- gym API ----------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.agent_pos = self.start_pos

        self.dynamic_specs = self._init_dynamic_obstacles()
        self.dynamic_positions = [sp["pos"] for sp in self.dynamic_specs]

        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()
        return obs, {}

    def step(self, action):
        self.steps += 1

        # move dynamics first
        self._move_dynamic_obstacles()

        r, c = self.agent_pos
        prev_dist = self._manhattan(self.agent_pos, self.goal_pos)

        nr, nc = r, c
        if action == 0:
            nr = max(r - 1, 0)
        elif action == 1:
            nr = min(r + 1, self.size - 1)
        elif action == 2:
            nc = max(c - 1, 0)
        elif action == 3:
            nc = min(c + 1, self.size - 1)

        cand = (nr, nc)
        hit_static = cand in self.static_obstacles
        hit_dynamic = cand in self._dynamic_cells_set()

        if hit_static or hit_dynamic:
            cand = (r, c)

        self.agent_pos = cand
        new_dist = self._manhattan(self.agent_pos, self.goal_pos)

        terminated = (self.agent_pos == self.goal_pos)
        truncated = (self.steps >= self.max_steps)

        # reward shaping
        if terminated:
            reward = 150.0
        else:
            reward = -0.1
            if hit_static:
                reward -= 2.0
            if hit_dynamic:
                reward -= 3.0

            if new_dist < prev_dist:
                reward += 0.35
            elif new_dist > prev_dist:
                reward -= 0.25

        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, {}

    # ---------------- render ----------------
    def render(self):
        if pygame is None:
            raise ImportError("pygame not installed. Install: pip install pygame-ce")

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption(
                "GridWorld 30x30 (Orange=Start, Blue=Agent, Green=Goal, Black=Static, Gray=Dynamic)"
            )
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.screen.fill((255, 255, 255))
        cell = self.window_size // self.size

        for (rr, cc) in self.static_obstacles:
            pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(cc * cell, rr * cell, cell, cell))

        for (rr, cc) in self.dynamic_positions:
            pygame.draw.rect(self.screen, (90, 90, 90), pygame.Rect(cc * cell, rr * cell, cell, cell))

        sr, sc = self.start_pos
        pygame.draw.rect(self.screen, (255, 165, 0), pygame.Rect(sc * cell, sr * cell, cell, cell))

        gr, gc = self.goal_pos
        pygame.draw.rect(self.screen, (0, 200, 0), pygame.Rect(gc * cell, gr * cell, cell, cell))

        ar, ac = self.agent_pos
        pygame.draw.rect(self.screen, (0, 0, 255), pygame.Rect(ac * cell, ar * cell, cell, cell))

        for i in range(self.size + 1):
            pygame.draw.line(self.screen, (70, 70, 70), (0, i * cell), (self.window_size, i * cell), 1)
            pygame.draw.line(self.screen, (70, 70, 70), (i * cell, 0), (i * cell, self.window_size), 1)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if pygame is not None and self.screen is not None:
            pygame.display.quit()
            pygame.quit()
        self.screen = None
        self.clock = None
