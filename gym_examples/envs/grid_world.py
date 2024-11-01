import random
import gym
from gym import spaces
import pygame
import numpy as np
from maze_project import config_maze as c


CUSTOM_MAZE = np.asarray([[ 0., -1.,  0.,  0.,  0.,  0.,  0.,],
                          [ 0., -1., -1., -1., -1., -1.,  0.,],
                          [ 0.,  0.,  0.,  0.,  0., -1.,  0.,],
                          [ 0., -1., -1., -1.,  0., -1.,  0.,],
                          [ 0., -1.,  0.,  0.,  0., -1.,  0.,],
                          [-1., -1., -1., -1.,  0., -1.,  0.,],
                          [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,]])

CUSTOM_WALL_LIST = list(np.asarray(np.where(CUSTOM_MAZE==-1)).T.tolist())

CUSTOM_MAZE_5 = np.asarray([[ 0.,  0.,  0.,  0.,  2.],
                            [-1., -1.,  0., -1.,  -2],
                            [ -2,  2,  0., -1., -1.],
                            [-1., -1.,  0.,  0,   0.],
                            [ 0.,  0.,  2, -1.,  0.]])

CUSTOM_WALL_LIST_5 = list(np.asarray(np.where(CUSTOM_MAZE_5==-1)).T.tolist())
CUSTOM_COIN_LIST_5 = list(np.asarray(np.where(CUSTOM_MAZE_5==2)).T.tolist())
CUSTOM_TRAP_LIST_5 = list(np.asarray(np.where(CUSTOM_MAZE_5==-2)).T.tolist())

def create_maze(size=c.SIZE):
    '''Build random maze within grid world. algorithm works with uneven number of cells.'''

    maze = np.zeros((size, size))
    cell_list = []
    maze_list = []
    sample_walls = []

    for i in range(size):
        for j in range(size):

            if (i + 1)%2 == 0 or (j + 1)%2 == 0:
                maze[i][j] = -2

            else:
                cell_list.append([i, j])

    init_cell_position = random.randrange(len(cell_list))
    init_cell = cell_list[init_cell_position]
    maze_list.append(init_cell)
    del cell_list[init_cell_position]

    if init_cell[0] > 0:
        sample_walls.append([init_cell[0] - 1, init_cell[1]])
        maze[init_cell[0] - 1][init_cell[1]] = -1

    if init_cell[1] > 0:
        sample_walls.append([init_cell[0], init_cell[1] - 1])
        maze[init_cell[0]][init_cell[1] - 1] = -1

    if init_cell[0]  < c.SIZE - 1:
        sample_walls.append([init_cell[0] + 1, init_cell[1]])
        maze[init_cell[0] + 1][init_cell[1]] = -1

    if init_cell[1] < c.SIZE -1:
        sample_walls.append([init_cell[0], init_cell[1] + 1])
        maze[init_cell[0]][init_cell[1] + 1] = -1

    # print(maze)

    while len(sample_walls) > 0:
        new_cell_list_position = random.randrange(len(sample_walls))
        loc_1 = [sample_walls[new_cell_list_position][0] + 1,
                    sample_walls[new_cell_list_position][1]]
        loc_2 = [sample_walls[new_cell_list_position][0] - 1,
                    sample_walls[new_cell_list_position][1]]
        loc_3 = [sample_walls[new_cell_list_position][0],
                    sample_walls[new_cell_list_position][1] - 1]
        loc_4 = [sample_walls[new_cell_list_position][0],
                    sample_walls[new_cell_list_position][1] + 1]

        if loc_1 in cell_list:
            new_maze_cell_position = cell_list.index(loc_1)
        elif loc_2 in cell_list:
            new_maze_cell_position = cell_list.index(loc_2)
        elif loc_3 in cell_list:
            new_maze_cell_position = cell_list.index(loc_3)
        elif loc_4 in cell_list:
            new_maze_cell_position = cell_list.index(loc_4)
        else:
            new_maze_cell_position = None

        if new_maze_cell_position is not None:
            wall_to_cell_x = sample_walls[new_cell_list_position][0]
            wall_to_cell_y = sample_walls[new_cell_list_position][1]
            cell_to_cell_x = cell_list[new_maze_cell_position][0]
            cell_to_cell_y = cell_list[new_maze_cell_position][1]
            maze[wall_to_cell_x][wall_to_cell_y] = 0
            maze_list.append([sample_walls[new_cell_list_position]])
            maze_list.append(cell_list[new_maze_cell_position])

            if wall_to_cell_x < size - 1:
                if maze[wall_to_cell_x + 1][wall_to_cell_y] == -2:
                    sample_walls.append([wall_to_cell_x + 1, wall_to_cell_y])
                    maze[wall_to_cell_x + 1][wall_to_cell_y] = -1

            if wall_to_cell_x > 0:
                if maze[wall_to_cell_x - 1][wall_to_cell_y] == -2:
                    sample_walls.append([wall_to_cell_x - 1, wall_to_cell_y])
                    maze[wall_to_cell_x - 1][wall_to_cell_y] = -1

            if wall_to_cell_y < size - 1:
                if maze[wall_to_cell_x][wall_to_cell_y + 1] == -2:
                    sample_walls.append([wall_to_cell_x, wall_to_cell_y + 1])
                    maze[wall_to_cell_x][wall_to_cell_y + 1] = -1

            if wall_to_cell_y > 0:
                if maze[wall_to_cell_x][wall_to_cell_y - 1] == -2:
                    sample_walls.append([wall_to_cell_x, wall_to_cell_y - 1])
                    maze[wall_to_cell_x][wall_to_cell_y - 1] = -1

            if cell_to_cell_x < size -1:
                if maze[cell_to_cell_x + 1][cell_to_cell_y] == -2:
                    sample_walls.append([cell_to_cell_x + 1, cell_to_cell_y])
                    maze[cell_to_cell_x + 1][cell_to_cell_y] = -1

            if cell_to_cell_x > 0:
                if maze[cell_to_cell_x - 1][cell_to_cell_y] == -2:
                    sample_walls.append([cell_to_cell_x - 1, cell_to_cell_y])
                    maze[cell_to_cell_x - 1][cell_to_cell_y] = -1

            if cell_to_cell_y < size - 1:
                if maze[cell_to_cell_x][cell_to_cell_y + 1] == -2:
                    sample_walls.append([cell_to_cell_x, cell_to_cell_y + 1])
                    maze[cell_to_cell_x][cell_to_cell_y + 1] = -1

            if cell_to_cell_y > 0:
                if maze[cell_to_cell_x][cell_to_cell_y - 1] == -2:
                    sample_walls.append([cell_to_cell_x, cell_to_cell_y - 1])
                    maze[cell_to_cell_x][cell_to_cell_y - 1] = -1

            del cell_list[new_maze_cell_position]

        del sample_walls[new_cell_list_position]
        # print(maze)

    return maze


def create_environment(game, size=c.SIZE):

    obj_list = []

    for i in range(0, c.SIZE):
        for j in range(0, c.SIZE):
            obj_list.append(np.asarray([i, j]).tolist())

    obj_list.remove(np.asarray([0, 0]).tolist())
    obj_list.remove(np.asarray([c.SIZE-1, c.SIZE-1]).tolist())

    if game[2] == 1:
        maze = create_maze(size)
        walls = list(np.asarray(np.where(maze==-1)).T.tolist())
        for wall in walls:
            obj_list.remove(wall)

    else:
        walls = []

    if game[0] == 1:
        coins = random.sample(obj_list, 3)

        for coin in coins:
            obj_list.remove(coin)

    else:
        coins = []


    if game[1] == 1:
        traps = random.sample(obj_list, 2)

        for trap in traps:
            obj_list.remove(trap)

    else:
        traps = []

    return walls, coins, traps

WALL_LIST, COIN_LIST, TRAP_LIST = create_environment([1, 1, 1])

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, game=[1, 0, 0], render_mode=None, size=c.SIZE, wall_list=WALL_LIST, coin_list=COIN_LIST, new_maze=True):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "walls": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "coins": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "traps": spaces.Box(0, size - 1, shape=(2,), dtype=int)
            }
        )

        if new_maze:
            self.wall_list, self.coin_list, self.trap_list = create_environment(game, size=size)
        else:
            self.wall_list = CUSTOM_WALL_LIST_5
            self.coin_list = CUSTOM_COIN_LIST_5.copy() 
            self.trap_list = CUSTOM_TRAP_LIST_5

        # print(self.maze)

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location,
                "target": self._target_location,
                "walls": self.wall_list,
                "coins": self.coin_list,
                "traps": self.trap_list}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)


        self._target_location = np.asarray([self.size - 1, self.size - 1])
        # Choose the agent's location uniformly at random within the maze
        self._agent_location = np.asarray([0, 0]) # self.np_random.integers(0, self.size, size=2, dtype=int)
        # self.coin_list = CUSTOM_COIN_LIST_5.copy()

        while self._agent_location.tolist() in self.wall_list or np.array_equal(self._target_location, self._agent_location):
            self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location or the walls
        # self._target_location = self._agent_location
        # while np.array_equal(self._target_location, self._agent_location) or self._target_location.tolist() in self.wall_list:
        #     self._target_location = self.np_random.integers(
        #         0, self.size, size=2, dtype=int
        #     )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        prev_location = self._agent_location
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        if self._agent_location.tolist() in self.wall_list:
            self._agent_location = self._agent_location - direction

        # An episode is done iff the agent has reached the target
        succeded = np.array_equal(self._agent_location, self._target_location)
        failed = True if self._agent_location.tolist() in self.trap_list else False
        scale = 1  # Binary sparse rewards

        if succeded:
            reward = scale * 1

        elif failed:
            reward = -1 * scale

        elif np.array_equal(self._agent_location, prev_location):
            reward = -0.1 * scale

        elif self._agent_location.tolist() in self.coin_list:
            self.coin_list.remove(self._agent_location.tolist())
            reward = 0.1 * scale

        else:
            reward = -0.01

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        if succeded or failed:
            terminated = True

        else:
            terminated = False

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        for coin in self.coin_list:
            coin = np.asarray(coin)
            pygame.draw.circle(
                canvas,
                (255, 255, 0),
                (coin + 0.5) * pix_square_size,
                pix_square_size / 3
            )

        for trap in self.trap_list:
            trap = np.asarray(trap)
            pygame.draw.rect(canvas,
                            (255, 0, 0),
                            pygame.Rect(
                                trap * pix_square_size,
                                (pix_square_size, pix_square_size),
            )
        )

        for wall in self.wall_list:
            wall = np.asarray(wall)
            pygame.draw.rect(canvas,
                             (0, 0, 0),
                             pygame.Rect(
                                pix_square_size * wall,
                                (pix_square_size, pix_square_size),
            )
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
