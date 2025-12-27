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
        self.action_space = spaces.Discrete(4)  # 0 up,1 down,2 left,3 right

        # Start & Goal (you can change if you want)
        self.start_pos = (1, 0)                 # matches the top-left-ish start in your screenshot
        self.goal_pos = (self.size - 1, self.size - 1)  # bottom-right

        self.start_state = self._to_state(self.start_pos)
        self.goal_state = self._to_state(self.goal_pos)

        self.state = self.start_state
        self.steps = 0
        self.max_steps = max_steps if max_steps is not None else (self.size * self.size * 3)

        # Obstacles EXACTLY like your screenshot (black cells)
        self.obstacles = set(self._obstacles_from_screenshot())

        # Ensure start/goal not blocked
        self.obstacles.discard(self.start_pos)
        self.obstacles.discard(self.goal_pos)

        # Pygame
        self.window_size = 750
        self.screen = None
        self.clock = None

    # ------------------- helpers -------------------
    def _to_state(self, pos):
        r, c = pos
        return r * self.size + c

    def _to_pos(self, state):
        return divmod(state, self.size)

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _obstacles_from_screenshot(self):
        """
        30x30 obstacle mask extracted from the image you shared.
        Coordinates are (row, col).
        """
        obs = []

        # Two big top blocks (5x3 each)
        # rows 3..5, cols 7..11
        for r in range(3, 6):
            for c in range(7, 12):
                obs.append((r, c))
        # rows 3..5, cols 18..22
        for r in range(3, 6):
            for c in range(18, 23):
                obs.append((r, c))

        # Small left cluster around row 10..11
        obs += [(10, 3), (11, 3), (11, 4)]
        # Small block row 10 col 9..10
        obs += [(10, 9), (10, 10)]
        # Small block row 10 col 18..19
        obs += [(10, 18), (10, 19)]

        # Vertical pillars (like the screenshot)
        # left pillar rows 11..14 cols 9..10 (actually in screenshot it appears as 2-wide in places)
        for r in range(11, 15):
            obs.append((r, 9))
            obs.append((r, 10))

        # right-ish pillar rows 11..14 cols 20..21
        for r in range(11, 15):
            obs.append((r, 20))
            obs.append((r, 21))

        # More pillars lower (rows 18..22)
        for r in range(18, 23):
            obs.append((r, 9))
            obs.append((r, 10))
            obs.append((r, 20))
            obs.append((r, 21))

        # Bottom middle small block rows 22..23 cols 13..14 (from screenshot)
        obs += [(22, 13), (22, 14), (23, 13), (23, 14)]

        # Single dot obstacle around row 15 col 25 (seen in screenshot)
        obs += [(15, 25)]

        # A few extra small blocks to match the screenshot middle scatter
        # row 10 has two small squares around col 12..13 (grey in your screenshot, but you asked only BLACK; keeping them out)
        # If you WANT them black, uncomment:
        # obs += [(4, 13), (4, 14)]

        # Also one single near bottom middle-left in screenshot:
        obs += [(9, 4)]

        # Some extra blocks that appear as a small bar around row 22 col 12..13 already included
        # A single block around row 23 col 15 not needed for black mask.

        return obs

    # ------------------- gym API -------------------
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
            nr, nc = r, c  # stay in place

        new_pos = (nr, nc)
        new_dist = self._manhattan(new_pos, self.goal_pos)

        self.state = self._to_state(new_pos)

        terminated = (self.state == self.goal_state)
        truncated = (self.steps >= self.max_steps)

        # Rewards: encourage minimum-distance path
        if terminated:
            reward = 30.0
        else:
            reward = -0.05  # step cost = shorter path
            if hit_obstacle:
                reward -= 1.0

            # distance shaping to learn faster
            if new_dist < prev_dist:
                reward += 0.15
            elif new_dist > prev_dist:
                reward -= 0.10

        if self.render_mode == "human":
            self.render()

        return self.state, reward, terminated, truncated, {}

    # ------------------- render -------------------
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
