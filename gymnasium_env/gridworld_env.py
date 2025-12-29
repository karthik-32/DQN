import os
os.environ["SDL_VIDEO_CENTERED"] = "1"
os.environ["SDL_HINT_VIDEO_MINIMIZE_ON_FOCUS_LOSS"] = "0"

import gymnasium as gym
from gymnasium import spaces

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

        self.observation_space = spaces.Discrete(self.size * self.size)
        self.action_space = spaces.Discrete(4)

        self.start_pos = (1, 0)
        self.goal_pos = (self.size - 1, self.size - 1)

        self.start_state = self._to_state(self.start_pos)
        self.goal_state = self._to_state(self.goal_pos)

        self.state = self.start_state
        self.steps = 0

        # max_steps is passed from training/play
        self.max_steps = max_steps if max_steps is not None else (self.size * self.size * 2)

        self.obstacles = set(self._obstacles_fixed())
        self.obstacles.discard(self.start_pos)
        self.obstacles.discard(self.goal_pos)

        self.window_size = 750
        self.screen = None
        self.clock = None

    def _to_state(self, pos):
        r, c = pos
        return r * self.size + c

    def _to_pos(self, state):
        return divmod(state, self.size)

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _obstacles_fixed(self):
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.start_state
        self.steps = 0
        if self.render_mode == "human":
            self.render()
        return self.state, {}

    def step(self, action):
        self.steps += 1

        r, c = self._to_pos(self.state)
        prev_pos = (r, c)
        prev_dist = self._manhattan(prev_pos, self.goal_pos)

        nr, nc = r, c
        if action == 0:
            nr = max(r - 1, 0)
        elif action == 1:
            nr = min(r + 1, self.size - 1)
        elif action == 2:
            nc = max(c - 1, 0)
        elif action == 3:
            nc = min(c + 1, self.size - 1)

        hit_obstacle = (nr, nc) in self.obstacles
        if hit_obstacle:
            nr, nc = r, c

        new_pos = (nr, nc)
        new_dist = self._manhattan(new_pos, self.goal_pos)
        self.state = self._to_state(new_pos)

        terminated = (self.state == self.goal_state)
        truncated = (self.steps >= self.max_steps)

        if terminated:
            reward = 100.0
        else:
            reward = -0.1
            if hit_obstacle:
                reward -= 1.5

            if new_dist < prev_dist:
                reward += 0.3
            elif new_dist > prev_dist:
                reward -= 0.2

        if self.render_mode == "human":
            self.render()

        return self.state, reward, terminated, truncated, {}

    def render(self):
        if pygame is None:
            raise ImportError("pygame not installed. Install: pip install pygame-ce")

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("GridWorld 30x30 (Orange=Start, Blue=Agent, Green=Goal, Black=Obstacles)")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.screen.fill((255, 255, 255))
        cell = self.window_size // self.size

        for (orow, ocol) in self.obstacles:
            pygame.draw.rect(
                self.screen, (0, 0, 0),
                pygame.Rect(ocol * cell, orow * cell, cell, cell),
            )

        sr, sc = self.start_pos
        pygame.draw.rect(self.screen, (255, 165, 0), pygame.Rect(sc * cell, sr * cell, cell, cell))

        gr, gc = self.goal_pos
        pygame.draw.rect(self.screen, (0, 200, 0), pygame.Rect(gc * cell, gr * cell, cell, cell))

        ar, ac = self._to_pos(self.state)
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
