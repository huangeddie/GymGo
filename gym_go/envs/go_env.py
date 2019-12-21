from enum import Enum

import gym
import numpy as np

from gym_go import govars, rendering
from gym_go.gogame import GoGame


class RewardMethod(Enum):
    """
    REAL: 0 = game is ongoing, 1 = black won, -1 = game tied or white won
    HEURISTIC: If game is ongoing, the reward is the area difference between black and white.
    Otherwise the game has ended, and if black has more area, the reward is BOARD_SIZE**2, otherwise it's -BOARD_SIZE**2
    """
    REAL = 'real'
    HEURISTIC = 'heuristic'


class GoEnv(gym.Env):
    metadata = {'render.modes': ['terminal', 'human']}
    gogame = GoGame()
    govars = govars

    def __init__(self, size, reward_method='real'):
        '''
        @param reward_method: either 'heuristic' or 'real'
        heuristic: gives # black pieces - # white pieces.
        real: gives 0 for in-game move, 1 for winning, -1 for losing,
            0 for draw, all from black player's perspective
        '''
        self.size = size
        self.state = GoGame.get_init_board(size)
        self.reward_method = RewardMethod(reward_method)
        self.observation_space = gym.spaces.Box(0, govars.NUM_CHNLS, shape=(govars.NUM_CHNLS, size, size))
        self.action_space = gym.spaces.Discrete(GoGame.get_action_size(self.state))
        self.group_map = [set(), set()]
        self.done = False

    def reset(self):
        '''
        Reset state, go_board, curr_player, prev_player_passed,
        done, return state
        '''
        self.state = GoGame.get_init_board(self.size)
        self.group_map = [set(), set()]
        self.done = False
        return np.copy(self.state)

    def step(self, action):
        '''
        Assumes the correct player is making a move. Black goes first.
        return observation, reward, done, info
        '''
        assert not self.done
        if isinstance(action, tuple) or isinstance(action, list) or isinstance(action, np.ndarray):
            assert 0 <= action[0] < self.size
            assert 0 <= action[1] < self.size
            action = self.size * action[0] + action[1]
        elif action is None:
            action = self.size ** 2

        actions = np.array([action])
        states, group_maps = GoGame.get_batch_next_states(self.state, actions, self.group_map)
        self.state, self.group_map = states[0], group_maps[0]
        self.done = GoGame.get_game_ended(self.state)
        return np.copy(self.state), self.get_reward(), self.done, self.get_info()

    def game_ended(self):
        return self.done

    def turn(self):
        return GoGame.get_turn(self.state)

    def prev_player_passed(self):
        return GoGame.get_prev_player_passed(self.state)

    def get_valid_moves(self):
        return GoGame.get_valid_moves(self.state)

    def action_2d_to_1d(self, action_2d):
        if action_2d is None:
            action_1d = self.size ** 2
        else:
            action_1d = action_2d[0] * self.size + action_2d[1]
        return action_1d

    def uniform_random_action(self):
        valid_moves = self.get_valid_moves()
        valid_move_idcs = np.argwhere(valid_moves).flatten()
        return np.random.choice(valid_move_idcs)

    def get_info(self):
        """
        :return: Debugging info for the state
        """
        return {
            'prev_player_passed': GoGame.get_prev_player_passed(self.state),
            'turn': 'b' if GoGame.get_turn(self.state) == GoEnv.govars.BLACK else 'w',
            'game_ended': GoGame.get_game_ended(self.state)
        }

    def get_state(self):
        """
        :return: copy of state
        """
        return np.copy(self.state)

    def get_canonical_state(self):
        """
        :return: canonical copy of state
        """
        return GoGame.get_canonical_form(self.state)


    def get_children(self, canonical=False):
        """
        :return: Same as get_children, but in canonical form
        """
        return GoGame.get_children(self.state, self.group_map, canonical)

    def get_winning(self):
        """
        :return: Who's currently winning in BLACK's perspective, regardless if the game is over
        """
        return GoGame.get_winning(self.state)

    def get_winner(self):
        """
        Get's the winner in BLACK's perspective
        :return:
        """

        if self.game_ended():
            return self.get_winning()
        else:
            return 0

    def get_reward(self):
        '''
        Return reward based on reward_method.
        heuristic: black total area - white total area
        real: 0 for in-game move, 1 for winning, 0 for losing,
            0.5 for draw, from black player's perspective.
            Winning and losing based on the Area rule
            Also known as Trump Taylor Scoring
        Area rule definition: https://en.wikipedia.org/wiki/Rules_of_Go#End
        '''
        if self.reward_method == RewardMethod.REAL:
            return self.get_winner()

        elif self.reward_method == RewardMethod.HEURISTIC:
            black_area, white_area = GoGame.get_areas(self.state)
            area_difference = black_area - white_area
            if self.game_ended():
                return (1 if area_difference > 0 else -1) * self.size ** 2
            return area_difference
        else:
            raise Exception("Unknown Reward Method")

    def __str__(self):
        return GoGame.str(self.state)

    def close(self):
        if hasattr(self, 'window'):
            assert hasattr(self, 'pyglet')
            self.window.close()
            self.pyglet.app.exit()

    def render(self, mode='terminal'):
        if mode == 'terminal':
            print(self.__str__())
        elif mode == 'human':
            import pyglet
            from pyglet.window import mouse
            from pyglet.window import key

            screen = pyglet.window.get_platform().get_default_display().get_default_screen()
            window_width = min(screen.width, screen.height) * 2 / 3
            window_height = window_width * 1.2
            window = pyglet.window.Window(window_width, window_height)

            self.window = window
            self.pyglet = pyglet
            self.user_action = None

            # Set Cursor
            cursor = window.get_system_mouse_cursor(window.CURSOR_CROSSHAIR)
            window.set_mouse_cursor(cursor)

            # Outlines
            lower_grid_coord = window_width * 0.075
            board_size = window_width * 0.85
            upper_grid_coord = board_size + lower_grid_coord
            delta = board_size / (self.size - 1)
            piece_r = delta / 3.3  # radius

            @window.event
            def on_draw():
                pyglet.gl.glClearColor(0.7, 0.5, 0.3, 1)
                window.clear()

                pyglet.gl.glLineWidth(3)
                batch = pyglet.graphics.Batch()

                # draw the grid and labels
                rendering.draw_grid(batch, delta, self.size, lower_grid_coord, upper_grid_coord)

                # info on top of the board
                rendering.draw_info(batch, window_width, window_height, upper_grid_coord, self.state)

                # Inform user what they can do
                rendering.draw_command_labels(batch, window_width, window_height)

                rendering.draw_title(batch, window_width, window_height)

                batch.draw()

                # draw the pieces
                rendering.draw_pieces(batch, lower_grid_coord, delta, piece_r, self.size, self.state)

            @window.event
            def on_mouse_press(x, y, button, modifiers):
                if button == mouse.LEFT:
                    grid_x = (x - lower_grid_coord)
                    grid_y = (y - lower_grid_coord)
                    x_coord = round(grid_x / delta)
                    y_coord = round(grid_y / delta)
                    try:
                        self.window.close()
                        pyglet.app.exit()
                        self.user_action = (x_coord, y_coord)
                    except:
                        pass

            @window.event
            def on_key_press(symbol, modifiers):
                if symbol == key.P:
                    self.window.close()
                    pyglet.app.exit()
                    self.user_action = None
                elif symbol == key.R:
                    self.reset()
                    self.window.close()
                    pyglet.app.exit()
                elif symbol == key.E:
                    self.window.close()
                    pyglet.app.exit()
                    self.user_action = -1

            pyglet.app.run()

            return self.user_action
