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
    def get_next_state(state, action1d, group_map=None, canonical=False):
        action_wrap = np.array([action1d])
        next_states_wrap, next_gmps_wrap = GoGame.get_batch_next_states(state, action_wrap, group_map, canonical)
        return next_states_wrap[0], next_gmps_wrap[0]

    @staticmethod
    def get_batch_next_states(state, batch_action1d, group_map=None, canonical=False):
        """
        Does not change the given state
        """
        if group_map is None:
            group_map = state_utils.get_group_map(state)

        m, n = state_utils.get_board_size(state)
        pass_idcs = np.where(batch_action1d == m * n)
        non_pass_idcs = np.where(batch_action1d != m * n)

        batch_size = len(batch_action1d)
        board_shape = state.shape[1:]
        batch_action2d = np.empty((batch_size, 2), dtype=np.int)
        batch_action2d[:, 0] = batch_action1d // n
        batch_action2d[:, 1] = batch_action1d % n
        batch_action2d[np.where(batch_action1d == m * n)] = 0


        player = state_utils.get_turn(state)
        opponent = 1 - player
        previously_passed = GoGame.get_prev_player_passed(state)

        batch_states = np.tile(state, (batch_size, 1, 1, 1))

        # Check move is valid
        assert (batch_states[non_pass_idcs, govars.INVD_CHNL, batch_action2d[non_pass_idcs, 0], batch_action2d[non_pass_idcs, 1]] == 0).all()

        batch_group_maps = [[group_map[0].copy(), group_map[1].copy()] for _ in range(batch_size)]
        batch_single_kill = [None for _ in range(batch_size)]
        batch_killed_groups = [set() for _ in range(batch_size)]

        batch_adj_locs, batch_surrounded = state_utils.get_batch_adj_data(state, batch_action2d)


        if previously_passed:
            batch_states[pass_idcs, govars.DONE_CHNL] = 1
        else:
            batch_states[pass_idcs, govars.PASS_CHNL] = 1

        # Non passes
        batch_states[non_pass_idcs, govars.PASS_CHNL] = 0

        # Add pieces
        batch_states[non_pass_idcs, player, batch_action2d[non_pass_idcs, 0], batch_action2d[non_pass_idcs, 1]] = 1

        batch_data = enumerate(zip(batch_action1d, batch_action2d, batch_states, batch_group_maps, batch_adj_locs,
                                   batch_killed_groups))

        for i, (action_1d, action_2d, state, group_map, adj_locs, killed_groups) in batch_data:
            # if the current player passes
            if action_1d == m * n:
                continue

            action_2d = tuple(action_2d)

            # Get all adjacent information
            adj_own_groups, adj_opp_groups = state_utils.get_adjacent_groups(group_map, adj_locs, player)

            # Go through opponent groups
            for group in adj_opp_groups:
                assert action_2d in group.liberties, (action_2d, player, group, state)
                if len(group.liberties) <= 1:
                    # Killed group
                    killed_groups.add(group)

                    # Remove group in board and group map
                    for loc in group.locations:
                        state[opponent, loc[0], loc[1]] = 0
                    group_map[opponent].remove(group)

                    # Metric for ko-protection
                    if len(group.locations) <= 1 and batch_single_kill[i] is None:
                        batch_single_kill[i] = next(iter(group.locations))

            adj_opp_groups.difference_update(killed_groups)

            # Update surviving adjacent opponent groups by removing liberties by the new action
            for opp_group in adj_opp_groups:
                assert action_2d in opp_group.liberties, (action_2d, opp_group, adj_opp_groups)

                # New group copy
                group_map[opponent].remove(opp_group)
                opp_group = opp_group.copy()
                group_map[opponent].add(opp_group)

                opp_group.liberties.remove(action_2d)

            # Update adjacent own groups that are merged with the action
            if len(adj_own_groups) > 0:
                merged_group = adj_own_groups.pop()
                group_map[player].remove(merged_group)
                merged_group = merged_group.copy()
            else:
                merged_group = govars.Group()

            group_map[player].add(merged_group)

            # Locations from action and adjacent groups
            merged_group.locations.add(action_2d)

            for own_group in adj_own_groups:
                merged_group.locations.update(own_group.locations)
                merged_group.liberties.update(own_group.liberties)
                group_map[player].remove(own_group)

            # Liberties from action
            for adj_loc in adj_locs:
                if np.count_nonzero(state[[govars.BLACK, govars.WHITE], adj_loc[0], adj_loc[1]]) == 0:
                    merged_group.liberties.add(adj_loc)

            if action_2d in merged_group.liberties:
                merged_group.liberties.remove(action_2d)

            # More work to do if we killed
            if len(killed_groups) > 0:
                killed_map = np.zeros(board_shape)
                for group in killed_groups:
                    for loc in group.locations:
                        killed_map[loc] = 1
                # Update own groups adjacent to opponent groups that we just killed
                killed_liberties = ndimage.binary_dilation(killed_map)
                affected_idcs = set(zip(*np.nonzero(state[player] * killed_liberties)))
                groups_to_update = set()
                for group in group_map[player]:
                    if not affected_idcs.isdisjoint(group.locations):
                        groups_to_update.add(group)

                all_pieces = np.sum(state[[govars.BLACK, govars.WHITE]], axis=0)
                empties = (1 - all_pieces)
                for group in groups_to_update:
                    group_matrix = np.zeros(board_shape)
                    for loc in group.locations:
                        group_matrix[loc] = 1

                    additional_liberties = ndimage.binary_dilation(group_matrix) * empties * killed_map
                    additional_liberties = set(zip(*np.where(additional_liberties)))

                    group_map[player].remove(group)
                    group = group.copy()
                    group_map[player].add(group)

                    group.liberties.update(additional_liberties)

        # Update illegal moves
        batch_states[:, govars.INVD_CHNL] = state_utils.get_batch_invalid_moves(batch_states, batch_group_maps, player)

        # If group was one piece, and location is surrounded by opponents,
        # activate ko protection
        for i, (single_kill, killed_groups, surrounded) in enumerate(zip(batch_single_kill, batch_killed_groups,
                                                                         batch_surrounded)):
            if single_kill is not None and len(killed_groups) == 1 and surrounded:
                state[govars.INVD_CHNL, single_kill[0], single_kill[1]] = 1

        # Switch turn
        state_utils.batch_set_turn(batch_states)

        if canonical:
            GoGame.set_batch_canonical_form(batch_states, opponent)

        return batch_states, batch_group_maps

    @staticmethod
    def get_children(state, group_map=None, canonical=False):
        if group_map is None:
            group_map = state_utils.get_group_map(state)

        valid_moves = GoGame.get_valid_moves(state)
        valid_move_idcs = np.argwhere(valid_moves).flatten()
        children, child_group_maps = GoGame.get_batch_next_states(state, valid_move_idcs, group_map, canonical)
        return children, child_group_maps

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
    def get_winning(state):
        black_area, white_area = GoGame.get_areas(state)
        area_difference = black_area - white_area

        if area_difference > 0:
            return 1
        elif area_difference == 0:
            return 0
        else:
            assert area_difference < 0
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
        Use DFS helper to find territory.
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
    def set_batch_canonical_form(batch_states, player):
        """
        Assumes the turn of all states is player
        The returned state is a seperate copy of the given state
        :param batch_states:
        :param player:
        :return:
        """

        if player == govars.WHITE:
            batch_states[:, [govars.BLACK, govars.WHITE]] = batch_states[:, [govars.WHITE, govars.BLACK]]
            state_utils.batch_set_turn(batch_states)

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
