import numpy as np
import itertools
from gym_go.govars import BLACK, WHITE, INVD_CHNL, PASS_CHNL, DONE_CHNL
from gym_go import state_utils
from sklearn import preprocessing

"""
The state of the game is a numpy array
* Are values are either 0 or 1

* Shape [6, SIZE, SIZE]

0 - Black pieces
1 - White pieces
2 - Turn (0 - black, 1 - white)
3 - Invalid moves (including ko-protection)
4 - Previous move was a pass
5 - Game over
"""


class GoGame:

    @staticmethod
    def get_init_board(size, black_first=True):
        # return initial board (numpy board)
        state = np.zeros((6, size, size))
        if not black_first:
            state_utils.set_turn(state)
        return state

    @staticmethod
    def get_next_state(state, action):
        """
        Does not change the given state
        :param state:
        :param action:
        :return: The next state
        """

        # check if game is already over
        if GoGame.get_game_ended(state) != 0:
            raise Exception('Attempt to step at {} after game is over'.format(action))

        state = np.copy(state)

        # if the current player passes
        if action == GoGame.get_action_size(state) - 1:
            # if two consecutive passes, game is over
            if GoGame.get_prev_player_passed(state):
                state_utils.set_game_ended(state)
            else:
                state_utils.set_prev_player_passed(state)

            # Update invalid channel
            state_utils.reset_invalid_moves(state)
            state_utils.add_invalid_moves(state)

            # Switch turn
            state_utils.set_turn(state)

            # Return event
            return state

        player = state_utils.get_turn(state)
        m, n = state_utils.get_board_size(state)

        # convert the move to 2d
        action = (action // m, action % n)

        # Check move is valid
        if not state_utils.is_within_bounds(state, action):
            raise Exception("{} Not Within bounds".format(action))
        elif state[INVD_CHNL][action] > 0:
            raise Exception("Invalid Move", action, state)

        state_utils.reset_invalid_moves(state)

        # Get all adjacent groups
        _, opponent_groups = state_utils.get_adjacent_groups(state, action)

        # Go through opponent groups
        killed_single_piece = None
        empty_adjacents_before_kill = state_utils.get_adjacent_locations(state, action)
        for group in opponent_groups:
            empty_adjacents_before_kill = empty_adjacents_before_kill - group.locations
            if len(group.liberties) <= 1:
                assert action in group.liberties

                # Remove group in board
                for loc in group.locations:
                    # TODO: Hardcoded other player. Make more generic
                    state[1 - player][loc] = 0

                # Metric for ko-protection
                if len(group.locations) <= 1:
                    if killed_single_piece is not None:
                        killed_single_piece = None
                    else:
                        killed_single_piece = group.locations.pop()

        # If group was one piece, and location is surrounded by opponents,
        # activate ko protection
        if killed_single_piece is not None and len(empty_adjacents_before_kill) <= 0:
            state[INVD_CHNL][killed_single_piece] = 1

        # Add the piece!
        state[player][action] = 1

        # Update illegal moves
        state_utils.add_invalid_moves(state)

        # This move was not a pass
        state_utils.set_prev_player_passed(state, 0)

        # Switch turn
        state_utils.set_turn(state)

        return state

    @staticmethod
    def get_action_size(state):
        # return number of actions
        m, n = state_utils.get_board_size(state)
        return m * n + 1

    @staticmethod
    def get_prev_player_passed(state):
        m, n = state_utils.get_board_size(state)
        return np.count_nonzero(state[PASS_CHNL] == 1) == m * n

    @staticmethod
    def get_game_ended(state):
        """
        :param state:
        :return: 0/1 = game not ended / game ended respectively
        """
        m, n = state_utils.get_board_size(state)
        return int(np.count_nonzero(state[DONE_CHNL] == 1) == m * n)

    @staticmethod
    def get_turn(state):
        """
        :param state:
        :return: Who's turn it is (BLACK/WHITE)
        """
        return state_utils.get_turn(state)

    @staticmethod
    def get_valid_moves(state):
        # return a fixed size binary vector
        return np.append(1 - state[INVD_CHNL].flatten(), 1)

    @staticmethod
    def action_2d_to_1d(action_2d, state):
        size = state.shape[1]
        if action_2d is None:
            action_1d = size ** 2
        else:
            action_1d = action_2d[0] * size + action_2d[1]
        return action_1d

    @staticmethod
    def get_areas(state):
        '''
        Return black area, white area
        Use DFS helper to find territory.
        '''

        m, n = state_utils.get_board_size(state)
        visited = np.zeros((m, n), dtype=np.bool)
        black_area = 0
        white_area = 0

        # loop through each intersection on board
        for r, c in itertools.product(range(m), range(n)):
            # count pieces towards area
            if state[BLACK][r, c] > 0:
                black_area += 1
            elif state[WHITE][r, c] > 0:
                white_area += 1

            # do DFS on unvisited territory
            elif not visited[r, c]:
                player, area = state_utils.explore_territory(state, (r, c), visited)

                # add area to corresponding player
                if player == BLACK:  # BLACK
                    black_area += area
                elif player == WHITE:  # WHITE
                    white_area += area

        return black_area, white_area

    @staticmethod
    def get_canonical_form(state, player):
        """
        The returned state is a seperate copy of the given state
        :param state:
        :param player:
        :return:
        """
        state = np.copy(state)
        if player == BLACK:
            return state
        else:
            assert player == WHITE
            num_channels = state.shape[0]
            channels = np.arange(num_channels)
            channels[BLACK] = WHITE
            channels[WHITE] = BLACK
            state = state[channels]
            state_utils.set_turn(state)
            return state

    @staticmethod
    def random_symmetry(chunk):
        """
        Returns a random symmetry of the chunk
        :param chunk:
        :return:
        """
        orientation = np.random.randint(0,8)

        if (orientation >> 0) % 2:
            # Horizontal flip
            chunk = np.flip(chunk, 2)
        if (orientation >> 1) % 2:
            # Vertical flip
            chunk = np.flip(chunk, 1)
        if (orientation >> 2) % 2:
            # Rotate 90 degrees
            chunk = np.rot90(chunk, axes=(1, 2))

        return chunk

    @staticmethod
    def get_symmetries(chunk):
        """
        :param chunk: A (C, BOARD_SIZE, BOARD_SIZE) numpy array, where C is any number
        :return: All 8 orientations that are symmetrical in a Go game over the 2nd and 3rd axes
        (i.e. rotations, flipping and combos of them)
        """
        symmetries = []

        for i in range(8):
            x = chunk
            if (i >> 0) % 2:
                # Horizontal flip
                x = np.flip(x, 2)
            if (i >> 1) % 2:
                # Vertical flip
                x = np.flip(x, 1)
            if (i >> 2) % 2:
                # Rotation 90 degrees
                x = np.rot90(x, axes=(1, 2))
            symmetries.append(x)

        return symmetries

    @staticmethod
    def random_weighted_action(move_weights):
        """
        Assumes all invalid moves have weight 0
        Action is 1D
        Expected shape is (NUM OF MOVES, )
        """
        move_weights = preprocessing.normalize(move_weights[np.newaxis], norm='l1')
        return np.random.choice(np.arange(len(move_weights[0])), p=move_weights[0])

    @staticmethod
    def random_action(state):
        """
        Assumed to be (6, BOARD_SIZE, BOARD_SIZE)
        Action is 1D
        """
        invalid_moves = state[INVD_CHNL].flatten()
        invalid_moves = np.append(invalid_moves, 0)
        move_weights = 1 - invalid_moves

        return GoGame.random_weighted_action(move_weights)

    @staticmethod
    def str(state):
        board_str = ' '

        size = state.shape[1]
        for i in range(size):
            board_str += '   {}'.format(i)
        board_str += '\n  '
        board_str += '----' * size + '-'
        board_str += '\n'
        for i in range(size):
            board_str += '{} |'.format(i)
            for j in range(size):
                if state[0][i, j] == 1:
                    board_str += ' B'
                elif state[1][i, j] == 1:
                    board_str += ' W'
                elif state[2][i, j] == 1:
                    board_str += ' .'
                else:
                    board_str += '  '

                board_str += ' |'

            board_str += '\n  '
            board_str += '----' * size + '-'
            board_str += '\n'

        black_area, white_area = GoGame.get_areas(state)
        game_ended = GoGame.get_game_ended(state)
        prev_player_passed = GoGame.get_prev_player_passed(state)
        turn = GoGame.get_turn(state)
        board_str += '\tTurn: {}, Last Turn Passed: {}, Game Over: {}\n'.format('B' if turn == 0 else 'W',
                                                                                prev_player_passed,
                                                                                game_ended)
        board_str += '\tBlack Area: {}, White Area: {}\n'.format(black_area, white_area)
        return board_str