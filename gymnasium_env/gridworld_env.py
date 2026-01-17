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

        self.max_steps = max_steps if max_steps is not None else (self.size * self.size * 2)

        # static obstacles (black)
        self.static_obstacles = set(self._static_obstacles_fixed())
        self.static_obstacles.discard(self.start_pos)
        self.static_obstacles.discard(self.goal_pos)

        # dynamic obstacles (dark gray)
        # Each obstacle has: position, direction, axis, min_bound, max_bound
        # deterministic patrol patterns (same every run)
        self.dynamic_specs = self._init_dynamic_obstacles()
        self.dynamic_positions = [spec["pos"] for spec in self.dynamic_specs]

        # Pygame
        self.window_size = 750
        self.screen = None
        self.clock = None

    # ---------------- helpers ----------------
    def _to_state(self, pos):
        r, c = pos
        return r * self.size + c

    def _to_pos(self, state):
        return divmod(state, self.size)

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _in_bounds(self, r, c):
        return 0 <= r < self.size and 0 <= c < self.size

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
        """
        10 dynamic obstacles moving in organized patterns.
        Each obstacle patrols along a row or column segment and bounces.
        Deterministic: same start positions and movement each run.
        """
        # chosen to avoid static obstacles + start/goal area
        specs = [
            # Horizontal patrols: axis="h", position moves in col
            {"pos": (2, 2),  "axis": "h", "dir": +1, "min": 2,  "max": 12},
            {"pos": (6, 24), "axis": "h", "dir": -1, "min": 16, "max": 27},
            {"pos": (8, 6),  "axis": "h", "dir": +1, "min": 4,  "max": 14},
            {"pos": (16, 5), "axis": "h", "dir": +1, "min": 3,  "max": 11},
            {"pos": (24, 18),"axis": "h", "dir": -1, "min": 14, "max": 25},

            # Vertical patrols: axis="v", position moves in row
            {"pos": (4, 15), "axis": "v", "dir": +1, "min": 4,  "max": 12},
            {"pos": (12, 26),"axis": "v", "dir": +1, "min": 10, "max": 22},
            {"pos": (17, 2), "axis": "v", "dir": -1, "min": 8,  "max": 18},
            {"pos": (20, 12),"axis": "v", "dir": +1, "min": 16, "max": 28},
            {"pos": (10, 17),"axis": "v", "dir": +1, "min": 7,  "max": 19},
        ]

        # Make sure none start inside static obstacles or start/goal
        cleaned = []
        for sp in specs:
            if sp["pos"] in self.static_obstacles:
                # shift one cell right if blocked
                r, c = sp["pos"]
                sp["pos"] = (r, min(self.size - 1, c + 1))
            if sp["pos"] == self.start_pos:
                sp["pos"] = (sp["pos"][0], sp["pos"][1] + 1)
            if sp["pos"] == self.goal_pos:
                sp["pos"] = (sp["pos"][0] - 1, sp["pos"][1])
            cleaned.append(sp)

        return cleaned

    def _dynamic_cells_set(self):
        return set(self.dynamic_positions)

    def _move_dynamic_obstacles(self):
        """
        Move each dynamic obstacle one step along its axis.
        Bounce at bounds or when next cell is blocked by static obstacles.
        Also prevents two dynamic obstacles from occupying same cell by simple rule:
        - if move would collide, that obstacle waits for this step.
        """
        occupied = set(self.dynamic_positions)  # current occupied
        new_positions = list(self.dynamic_positions)

        for i, spec in enumerate(self.dynamic_specs):
            r, c = self.dynamic_positions[i]
            axis = spec["axis"]
            d = spec["dir"]

            nr, nc = r, c
            if axis == "h":
                nc = c + d
                # bounce on bounds of patrol
                if nc < spec["min"] or nc > spec["max"]:
                    spec["dir"] *= -1
                    d = spec["dir"]
                    nc = c + d
            else:  # "v"
                nr = r + d
                if nr < spec["min"] or nr > spec["max"]:
                    spec["dir"] *= -1
                    d = spec["dir"]
                    nr = r + d

            candidate = (nr, nc)

            # if candidate hits static obstacle, bounce and try opposite direction once
            if candidate in self.static_obstacles or not self._in_bounds(nr, nc):
                spec["dir"] *= -1
                d = spec["dir"]
                if axis == "h":
                    candidate = (r, c + d)
                else:
                    candidate = (r + d, c)

            # if still invalid, stay
            if candidate in self.static_obstacles or not self._in_bounds(candidate[0], candidate[1]):
                candidate = (r, c)

            # prevent entering agent cell (makes it fair)
            agent_pos = self._to_pos(self.state)
            if candidate == agent_pos:
                candidate = (r, c)

            # prevent collisions among dynamic obstacles
            if candidate in occupied and candidate != (r, c):
                candidate = (r, c)  # wait

            # update sets
            occupied.remove((r, c))
            occupied.add(candidate)
            new_positions[i] = candidate

        self.dynamic_positions = new_positions

    # ---------------- gym API ----------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.start_state
        self.steps = 0

        # reset dynamic obstacles to deterministic initial positions
        self.dynamic_specs = self._init_dynamic_obstacles()
        self.dynamic_positions = [spec["pos"] for spec in self.dynamic_specs]

        if self.render_mode == "human":
            self.render()
        return self.state, {}

    def step(self, action):
        self.steps += 1

        # Move dynamic obstacles FIRST each step (consistent)
        self._move_dynamic_obstacles()

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

        candidate = (nr, nc)

        hit_static = candidate in self.static_obstacles
        hit_dynamic = candidate in self._dynamic_cells_set()

        if hit_static or hit_dynamic:
            candidate = (r, c)  # blocked

        new_pos = candidate
        new_dist = self._manhattan(new_pos, self.goal_pos)

        self.state = self._to_state(new_pos)

        terminated = (self.state == self.goal_state)
        truncated = (self.steps >= self.max_steps)

        # rewards
        if terminated:
            reward = 100.0
        else:
            reward = -0.1
            if hit_static:
                reward -= 1.5
            if hit_dynamic:
                reward -= 2.0  # stronger penalty for dynamic obstacle collision attempt

            if new_dist < prev_dist:
                reward += 0.3
            elif new_dist > prev_dist:
                reward -= 0.2

        if self.render_mode == "human":
            self.render()

        return self.state, reward, terminated, truncated, {}

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

        # static obstacles (black)
        for (orow, ocol) in self.static_obstacles:
            pygame.draw.rect(
                self.screen, (0, 0, 0),
                pygame.Rect(ocol * cell, orow * cell, cell, cell),
            )

        # dynamic obstacles (dark gray)
        for (orow, ocol) in self.dynamic_positions:
            pygame.draw.rect(
                self.screen, (90, 90, 90),
                pygame.Rect(ocol * cell, orow * cell, cell, cell),
            )

        # start (orange)
        sr, sc = self.start_pos
        pygame.draw.rect(self.screen, (255, 165, 0), pygame.Rect(sc * cell, sr * cell, cell, cell))

        # goal (green)
        gr, gc = self.goal_pos
        pygame.draw.rect(self.screen, (0, 200, 0), pygame.Rect(gc * cell, gr * cell, cell, cell))

        # agent (blue)
        ar, ac = self._to_pos(self.state)
        pygame.draw.rect(self.screen, (0, 0, 255), pygame.Rect(ac * cell, ar * cell, cell, cell))

        # grid lines
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
