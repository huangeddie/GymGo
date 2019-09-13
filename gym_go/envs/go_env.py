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

    def step(self, action):
        ''' 
        Assumes the correct player is making a move. Black goes first.
        return observation, reward, done, info 
        '''
        if isinstance(action, tuple) or isinstance(action, list) or isinstance(action, np.ndarray):
            if action is None:
                action = self.size**2
            else:
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
                return (1 if area_difference > 0 else -1) * self.size**2
            return area_difference
        else:
            raise Exception("Unknown Reward Method")

    @property
    def action_space(self):
        return GoEnv.gogame.get_action_size(self.state)

    def __str__(self):
        board_str = ' '

        for i in range(self.size):
            board_str += '   {}'.format(i)
        board_str += '\n  '
        board_str += '----' * self.size + '-'
        board_str += '\n'
        for i in range(self.size):
            board_str += '{} |'.format(i)
            for j in range(self.size):
                if self.state[0][i, j] == 1:
                    board_str += ' B'
                elif self.state[1][i, j] == 1:
                    board_str += ' W'
                elif self.state[2][i, j] == 1:
                    board_str += ' .'
                else:
                    board_str += '  '

                board_str += ' |'

            board_str += '\n  '
            board_str += '----' * self.size + '-'
            board_str += '\n'
        info = self.get_info()
        board_str += '\tTurn: {}, Last Turn Passed: {}, Game Over: {}\n'.format('b' if self.turn == 0 else 'w', self.prev_player_passed,
                                                                                self.game_ended)
        board_str += '\tBlack Area: {}, White Area: {}, Reward: {}\n'.format(info['area']['b'], info['area']['w'],
                                                                             self.get_reward())
        return board_str

    def render(self, mode='terminal'):
        if mode == 'terminal':
            print(self.__str__())
        else:
            raise Exception("Unknown mode")