import gym
import numpy as np
from enum import Enum
from gym_go.gogame import GoGame
from gym_go import govars

class RewardMethod(Enum):
    """
    REAL: 0 = game is ongoing, 1 = black won, -1 = game tied or white won
    HEURISTIC: If game is ongoing, the reward is the area difference between black and white.
    Otherwise the game has ended, and if black has more area, the reward is BOARD_SIZE**2, otherwise it's -BOARD_SIZE**2
    """
    REAL = 'real'
    HEURISTIC = 'heuristic'


class GoEnv(gym.Env):
    metadata = {'render.modes': ['terminal']}
    gogame = GoGame()
    govars = govars

    def __init__(self, size, reward_method='real', black_first=True):
        '''
        @param reward_method: either 'heuristic' or 'real' 
        heuristic: gives # black pieces - # white pieces. 
        real: gives 0 for in-game move, 1 for winning, -1 for losing, 
            0 for draw, all from black player's perspective
        '''
        self.size = size
        self.state = GoEnv.gogame.get_init_board(size, black_first)
        self.reward_method = RewardMethod(reward_method)

    def reset(self, black_first=True):
        '''
        Reset state, go_board, curr_player, prev_player_passed,
        done, return state
        '''
        self.state = GoEnv.gogame.get_init_board(self.size, black_first)
        return np.copy(self.state)

    @property
    def prev_player_passed(self):
        return GoEnv.gogame.get_prev_player_passed(self.state)

    @property
    def turn(self):
        return GoEnv.gogame.get_turn(self.state)

    @property
    def game_ended(self):
        return GoEnv.gogame.get_game_ended(self.state)

    def action_2d_to_1d(self, action_2d):
        if action_2d is None:
            action_1d = self.size ** 2
        else:
            action_1d = action_2d[0] * self.size + action_2d[1]
        return action_1d

    def get_valid_moves(self):
        return GoGame.get_valid_moves(self.state)

    def uniform_random_action(self):
        valid_moves = self.get_valid_moves()
        valid_move_idcs = np.argwhere(valid_moves > 0)
        return np.random.choice(valid_move_idcs)

    def step(self, action):
        ''' 
        Assumes the correct player is making a move. Black goes first.
        return observation, reward, done, info 
        '''
        if action is None:
            action = self.size ** 2
        elif isinstance(action, tuple) or isinstance(action, list) or isinstance(action, np.ndarray):
            action = action[0] * self.size + action[1]
        self.state = GoEnv.gogame.get_next_state(self.state, action)
        return np.copy(self.state), self.get_reward(), GoEnv.gogame.get_game_ended(self.state), self.get_info()

    def get_info(self):
        """
        :return: Debugging info for the state
        """
        black_area, white_area = GoEnv.gogame.get_areas(self.state)
        return {
            'prev_player_passed': GoEnv.gogame.get_prev_player_passed(self.state),
            'turn': 'b' if GoEnv.gogame.get_turn(self.state) == GoEnv.govars.BLACK else 'w',
            'area': {
                'w': white_area,
                'b': black_area,
            }
        }

    def get_canonical_state(self):
        return GoEnv.gogame.get_canonical_form(self.state, self.turn)

    def get_state(self):
        """
        Returns deep copy of state
        """
        return np.copy(self.state)

    def get_winner(self):
        """
        Get's the winner in BLACK's perspective
        :return:
        """
        black_area, white_area = GoEnv.gogame.get_areas(self.state)
        area_difference = black_area - white_area

        if self.game_ended:
            if area_difference > 0:
                return 1
            else:
                return -1
        else:
            return 0

    def get_reward(self):
        '''
        Return reward based on reward_method.
        heuristic: black total area - white total area
        real: 0 for in-game move, 1 for winning, -1 for losing, 
            0 for draw, from black player's perspective.
            Winning and losing based on the Area rule
        Area rule definition: https://en.wikipedia.org/wiki/Rules_of_Go#End
        '''
        black_area, white_area = GoEnv.gogame.get_areas(self.state)
        area_difference = black_area - white_area

        if self.reward_method == RewardMethod.REAL:
            return self.get_winner()

        elif self.reward_method == RewardMethod.HEURISTIC:
            if self.game_ended:
                return (1 if area_difference > 0 else -1) * self.size ** 2
            return area_difference
        else:
            raise Exception("Unknown Reward Method")

    @property
    def action_space(self):
        return GoEnv.gogame.get_action_size(self.state)

    def __str__(self):
        return GoGame.str(self.state)

    def render(self, mode='terminal'):
        if mode == 'terminal':
            print(self.__str__())
        elif mode == 'human':
            import pyglet
            from pyglet.window import mouse

            window = pyglet.window.Window()

            @window.event
            def on_key_press(symbol, modifiers):
                print('A key was pressed')

            @window.event
            def on_draw():
                window.clear()

                batch = pyglet.graphics.Batch()

                vertex_list = batch.add(2, pyglet.gl.GL_POINTS, None,
                                        ('v2i', (10, 15, 30, 35)),
                                        ('c3B', (0, 0, 255, 0, 255, 0))
                                        )

                batch.draw()

            @window.event
            def on_mouse_press(x, y, button, modifiers):
                if button == mouse.LEFT:
                    print('The left mouse button was pressed.')

            pyglet.app.run()
            raise Exception("Unknown mode")
