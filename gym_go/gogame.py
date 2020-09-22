import numpy as np
from scipy import ndimage
from sklearn import preprocessing

from gym_go import state_utils, govars

"""
The state of the game is a numpy array
* Are values are either 0 or 1

* Shape [NUM_CHNLS, SIZE, SIZE]

0 - Black pieces
1 - White pieces
2 - Turn (0 - black, 1 - white)
3 - Invalid moves (including ko-protection)
4 - Previous move was a pass
5 - Game over
"""


class GoGame:

    @staticmethod
    def get_init_board(size):
        # return initial board (numpy board)
        state = np.zeros((govars.NUM_CHNLS, size, size))
        return state

    @staticmethod
    def get_next_state(state, action1d, canonical=False):
        """
        Does not change the given state
        """

        # Deep copy the state to modify
        state = np.copy(state)

        # Initialize basic variables
        board_shape = state.shape[1:]
        pass_idx = np.prod(board_shape)
        action2d = action1d // board_shape[0], action1d % board_shape[1]
        passed = action1d == pass_idx

        player = state_utils.get_turn(state)
        previously_passed = GoGame.get_prev_player_passed(state)

        ko_protect = None

        # Pass?
        if passed:
            # We passed
            state[govars.PASS_CHNL] = 1
            if previously_passed:
                # Game ended
                state[govars.DONE_CHNL] = 1
        else:
            # We did not pass
            state[govars.PASS_CHNL] = 0

            # Check move is valid
            assert state[govars.INVD_CHNL, action2d[0], action2d[1]] == 0, ("Invalid move", action2d)

            # Add piece
            state[player, action2d[0], action2d[1]] = 1

            # Get adjacent location and check whether the piece will be surrounded by any piece
            adj_locs, surrounded = state_utils.get_adj_data(state, action2d)

            # Update groups
            killed_groups = GoGame.update_groups(state, adj_locs, player)

            # If only killed one group, and that one group was one piece, and piece set is surrounded by opponents,
            # activate ko protection
            if len(killed_groups) == 1 and surrounded:
                killed_group = killed_groups[0]
                if len(killed_group) == 1:
                    ko_protect = killed_group[0]

        # Update illegal moves
        state[govars.INVD_CHNL] = state_utils.get_invalid_moves(state, player, ko_protect)

        # Switch turn
        state_utils.set_turn(state)

        if canonical:
            GoGame.set_canonical_form(state, 1 - player)

        return state

    @staticmethod
    def update_groups(state, adj_locs, player):
        opponent = 1 - player
        killed_groups = []

        all_pieces = np.sum(state[[govars.BLACK, govars.WHITE]], axis=0)
        empties = 1 - all_pieces

        all_opp_groups, _ = ndimage.measurements.label(state[opponent])

        # Go through opponent groups
        all_adj_labels = all_opp_groups[adj_locs[:, 0], adj_locs[:, 1]]
        all_adj_labels = np.unique(all_adj_labels)
        for opp_group_idx in all_adj_labels[np.nonzero(all_adj_labels)]:
            opp_group = all_opp_groups == opp_group_idx
            liberties = empties * ndimage.binary_dilation(opp_group)
            if np.sum(liberties) <= 0:
                # Killed group
                opp_group_locs = np.argwhere(opp_group)
                killed_groups.append(opp_group_locs)

                state[opponent] *= 1 - opp_group

        return killed_groups

    @staticmethod
    def get_children(state, canonical=False, padded=True):
        valid_moves = GoGame.get_valid_moves(state)
        n = len(valid_moves)
        valid_move_idcs = np.argwhere(valid_moves).flatten()
        children = []
        for move in valid_move_idcs:
            child = GoGame.get_next_state(state, move, canonical)
            children.append(child)

        if padded:
            padded_children = np.zeros((n, *state.shape))
            padded_children[valid_move_idcs] = children
            children = padded_children
        return children

    @staticmethod
    def get_action_size(state=None, board_size: int = None):
        # return number of actions
        if state is not None:
            m, n = state_utils.get_board_size(state)
        elif board_size is not None:
            m, n = board_size, board_size
        else:
            raise RuntimeError('No argument passed')
        return m * n + 1

    @staticmethod
    def get_prev_player_passed(state):
        m, n = state_utils.get_board_size(state)
        return np.count_nonzero(state[govars.PASS_CHNL] == 1) == m * n

    @staticmethod
    def get_game_ended(state):
        """
        :param state:
        :return: 0/1 = game not ended / game ended respectively
        """
        m, n = state_utils.get_board_size(state)
        return int(np.count_nonzero(state[govars.DONE_CHNL] == 1) == m * n)

    @staticmethod
    def get_winning(state, komi=0):
        black_area, white_area = GoGame.get_areas(state)
        area_difference = black_area - white_area
        komi_correction = area_difference - komi

        if komi_correction > 0:
            return 1
        elif komi_correction == 0:
            return 0
        else:
            assert komi_correction < 0
            return -1

    @staticmethod
    def get_turn(state):
        """
        :param state:
        :return: Who's turn it is (govars.BLACK/govars.WHITE)
        """
        return state_utils.get_turn(state)

    @staticmethod
    def get_valid_moves(state):
        # return a fixed size binary vector
        if GoGame.get_game_ended(state):
            return np.zeros(GoGame.get_action_size(state))
        return np.append(1 - state[govars.INVD_CHNL].flatten(), 1)

    @staticmethod
    def action_2d_to_1d(action_2d, state):
        size = state.shape[1]
        if action_2d is None:
            action_1d = size ** 2
        else:
            action_1d = action_2d[0] * size + action_2d[1]
        return action_1d

    @staticmethod
    def get_liberties(state: np.ndarray):
        return state_utils.get_liberties(state)

    @staticmethod
    def get_num_liberties(state: np.ndarray):
        return state_utils.get_num_liberties(state)

    @staticmethod
    def get_areas(state):
        '''
        Return black area, white area
        '''

        all_pieces = np.sum(state[[govars.BLACK, govars.WHITE]], axis=0)
        empties = 1 - all_pieces

        empty_labels, num_empty_areas = ndimage.measurements.label(empties)

        black_area, white_area = np.sum(state[govars.BLACK]), np.sum(state[govars.WHITE])
        for label in range(1, num_empty_areas + 1):
            empty_area = empty_labels == label
            neighbors = ndimage.binary_dilation(empty_area)
            black_claim = False
            white_claim = False
            if (state[govars.BLACK] * neighbors > 0).any():
                black_claim = True
            if (state[govars.WHITE] * neighbors > 0).any():
                white_claim = True
            if black_claim and not white_claim:
                black_area += np.sum(empty_area)
            elif white_claim and not black_claim:
                white_area += np.sum(empty_area)

        return black_area, white_area

    @staticmethod
    def get_canonical_form(state):
        """
        The returned state is a shallow copy of the given state
        :param state:
        :param player:
        :return:
        """

        player = GoGame.get_turn(state)
        if player == govars.BLACK:
            return state
        else:
            assert player == govars.WHITE
            num_channels = state.shape[0]
            channels = np.arange(num_channels)
            channels[govars.BLACK] = govars.WHITE
            channels[govars.WHITE] = govars.BLACK
            can_state = state[channels]
            state_utils.set_turn(can_state)
            return can_state

    @staticmethod
    def set_canonical_form(state, player):
        """
        Assumes the turn of all states is player
        The returned state is a seperate copy of the given state
        :param state:
        :param player:
        :return:
        """

        if player == govars.WHITE:
            state[[govars.BLACK, govars.WHITE]] = state[[govars.WHITE, govars.BLACK]]
            state_utils.set_turn(state)

    @staticmethod
    def random_symmetry(chunk):
        """
        Returns a random symmetry of the chunk
        :param chunk: A (C, BOARD_SIZE, BOARD_SIZE) numpy array, where C is any number
        :return:
        """
        orientation = np.random.randint(0, 8)

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
        Assumed to be (NUM_CHNLS, BOARD_SIZE, BOARD_SIZE)
        Action is 1D
        """
        invalid_moves = state[govars.INVD_CHNL].flatten()
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
                if state[0, i, j] == 1:
                    board_str += ' B'
                elif state[1, i, j] == 1:
                    board_str += ' W'
                elif state[2, i, j] == 1:
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
