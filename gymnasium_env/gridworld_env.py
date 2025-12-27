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

    def __init__(self, size=15, render_mode=None):
        super().__init__()
        self.size = int(size)
        self.render_mode = render_mode

        self.observation_space = spaces.Discrete(self.size * self.size)
        self.action_space = spaces.Discrete(4)

        self.start_pos = (0, 0)
        self.goal_pos = (self.size - 1, self.size - 1)

        self.start_state = self._to_state(self.start_pos)
        self.goal_state = self._to_state(self.goal_pos)
        self.state = self.start_state

        self.obstacles = self._make_static_obstacles()
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

    def _make_static_obstacles(self):
        obs = set()

        # vertical walls (col, gaps)
        walls = [
            (3, [2, 10]),
            (7, [5, 12]),
            (11, [3, 9]),
        ]
        for col, gaps in walls:
            if col >= self.size:
                continue
            for row in range(self.size):
                if row in gaps:
                    continue
                obs.add((row, col))

        # horizontal walls (row, gaps)
        h_walls = [
            (5, [1, 8, 13]),
            (9, [4, 10]),
        ]
        for row, gaps in h_walls:
            if row >= self.size:
                continue
            for col in range(self.size):
                if col in gaps:
                    continue
                obs.add((row, col))

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.start_state
        if self.render_mode == "human":
            self.render()
        return self.state, {}

    def step(self, action):
        r, c = self._to_pos(self.state)

        nr, nc = r, c
        if action == 0:      # up
            nr = max(r - 1, 0)
        elif action == 1:    # down
            nr = min(r + 1, self.size - 1)
        elif action == 2:    # left
            nc = max(c - 1, 0)
        elif action == 3:    # right
            nc = min(c + 1, self.size - 1)

        hit_obstacle = (nr, nc) in self.obstacles
        if hit_obstacle:
            nr, nc = r, c

        self.state = self._to_state((nr, nc))

        terminated = (self.state == self.goal_state)
        truncated = False

        # shortest-path shaping
        if terminated:
            reward = 10.0
        elif hit_obstacle:
            reward = -1.0
        else:
            reward = -0.05

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
            pygame.display.set_caption(
                "GridWorld 15x15 (Orange=Start, Blue=Agent, Green=Goal, Black=Obstacles)"
            )
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.screen.fill((255, 255, 255))
        cell = self.window_size // self.size

        # obstacles
        for (orow, ocol) in self.obstacles:
            pygame.draw.rect(
                self.screen, (0, 0, 0),
                pygame.Rect(ocol * cell, orow * cell, cell, cell),
            )

        # start
        sr, sc = self.start_pos
        pygame.draw.rect(
            self.screen, (255, 165, 0),
            pygame.Rect(sc * cell, sr * cell, cell, cell),
        )

        # goal
        gr, gc = self.goal_pos
        pygame.draw.rect(
            self.screen, (0, 200, 0),
            pygame.Rect(gc * cell, gr * cell, cell, cell),
        )

        # agent
        ar, ac = self._to_pos(self.state)
        pygame.draw.rect(
            self.screen, (0, 0, 255),
            pygame.Rect(ac * cell, ar * cell, cell, cell),
        )

        # grid lines
        for i in range(self.size + 1):
            pygame.draw.line(self.screen, (50, 50, 50), (0, i * cell), (self.window_size, i * cell), 1)
            pygame.draw.line(self.screen, (50, 50, 50), (i * cell, 0), (i * cell, self.window_size), 1)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if pygame is not None and self.screen is not None:
            pygame.display.quit()
            pygame.quit()
        self.screen = None
        self.clock = None
