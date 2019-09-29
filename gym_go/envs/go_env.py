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

            screen = pyglet.window.get_platform().get_default_display().get_default_screen()
            window_width = min(screen.width, screen.height) / 2
            window_height = window_width * 1.2
            window = pyglet.window.Window(window_width, window_height)
            lower_coord = window_width * 0.15
            board_size = window_width * 0.7
            delta = board_size / (self.size - 1)
            piece_r = delta / 3.3 # radius

            def draw_grid():
                label_offset = window_width * 0.08
                upper_coord = board_size + lower_coord
                left_coord = lower_coord
                right_coord = lower_coord
                ver_list = []
                color_list = []
                num_vert = 0
                batch = pyglet.graphics.Batch()
                for i in range(self.size):
                    # horizontal
                    ver_list.extend((lower_coord, left_coord,
                        upper_coord, right_coord))
                    # vertical
                    ver_list.extend((left_coord, lower_coord,
                        right_coord, upper_coord))
                    color_list.extend([0] * 12) # black
                    # label on the left
                    pyglet.text.Label(str(i),
                          font_name='Courier', font_size=18,
                          x=lower_coord-label_offset, y=left_coord,
                          anchor_x='center', anchor_y='center',
                          color=(0,0,0,255), batch=batch, dpi=110)
                    # label on the bottom
                    pyglet.text.Label(str(i),
                          font_name='Courier', font_size=18,
                          x=left_coord, y=lower_coord-label_offset,
                          anchor_x='center', anchor_y='center',
                          color=(0,0,0,255), batch=batch, dpi=110)
                    left_coord += delta
                    right_coord += delta
                    num_vert += 4
                batch.add(num_vert, pyglet.gl.GL_LINES, None,
                    ('v2f/static', ver_list), ('c3B/static', color_list))
                batch.draw()

            def draw_circle(x, y, color, radius):
                num_sides = 50
                verts = [x, y]
                colors = [color] * 3
                for i in range(num_sides + 1):
                    verts.append(x + radius * np.cos(i * np.pi * 2 / num_sides))
                    verts.append(y + radius * np.sin(i * np.pi * 2 / num_sides))
                    colors.extend([color] * 3)
                pyglet.graphics.draw(len(verts) // 2, pyglet.gl.GL_TRIANGLE_FAN,
                    ('v2f', verts), ('c3B', colors))

            def draw_info():
                info_offset = window_height * 0.1
                info = self.get_info()
                batch = pyglet.graphics.Batch()
                player = 'Black' if info['turn'] == 'b' else 'White'
                passed = 'passed' if info['prev_player_passed'] else 'did not pass'
                turn_label = pyglet.text.Label("{}'s turn, previous player {}".format(player, passed),
                      font_name='Georgia', font_size=12,
                      x=window_width/2, y=window_height-info_offset,
                      anchor_x='center', anchor_y='center',
                      color=(0,0,0,255), batch=batch, dpi=110)
                pyglet.text.Label("black area: {}, white area: {}, game over: {}".format(info['area']['b'], info['area']['w'], self.game_ended),
                      font_name='Georgia', font_size=12,
                      x=window_width/2, y=window_height-info_offset-turn_label.content_height,
                      anchor_x='center', anchor_y='center',
                      color=(0,0,0,255), batch=batch, dpi=110)
                batch.draw()

            @window.event
            def on_draw():
                pyglet.gl.glClearColor(242/255, 197/255, 119/255, 1)
                window.clear()

                pyglet.gl.glLineWidth(3)
                # draw the grid and labels
                draw_grid()

                # draw the pieces
                for i in range(self.size):
                    for j in range(self.size):
                        # black piece
                        if self.state[0][i, j] == 1:
                            draw_circle(lower_coord + i * delta, lower_coord + j * delta,
                                0, piece_r) # 0 for black

                        # white piece
                        if self.state[1][i, j] == 1:
                            draw_circle(lower_coord + i * delta, lower_coord + j * delta,
                                225, piece_r) # 255 for white

                # info on top of the board
                draw_info()

            @window.event
            def on_mouse_press(x, y, button, modifiers):
                if button == mouse.LEFT:
                    x_coord = round((x - lower_coord) / delta)
                    y_coord = round((y - lower_coord) / delta)
                    self.step((x_coord, y_coord))
                    return (x_coord, y_coord)

            pyglet.app.run()
