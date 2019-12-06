import itertools

import numpy as np
from gym_go import state_utils
from gym_go.govars import BLACK, WHITE, INVD_CHNL, PASS_CHNL, DONE_CHNL, Group
from scipy import ndimage
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
    def get_children(state, group_map=None):
        if group_map is None:
            group_map = state_utils.get_all_groups(state)

        children = []
        children_groupmaps = []

        valid_moves = GoGame.get_valid_moves(state)
        valid_move_idcs = np.argwhere(valid_moves > 0).flatten()
        for move in valid_move_idcs:
            next_state, child_groupmap = GoGame.get_next_state(state, move, group_map)
            children.append(next_state)
            children_groupmaps.append(child_groupmap)
        return children, children_groupmaps

    @staticmethod
    def get_canonical_children(state, group_map=None):
        children, child_group_maps = GoGame.get_children(state, group_map)
        canonical_children = list(map(lambda child: GoGame.get_canonical_form(child), children))
        return canonical_children, child_group_maps

    @staticmethod
    def get_next_state(state, action, group_map=None, inplace=False):
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

        if group_map is None:
            group_map = state_utils.get_all_groups(state)

        # if the current player passes
        if action == GoGame.get_action_size(state) - 1:
            # if two consecutive passes, game is over
            if GoGame.get_prev_player_passed(state):
                state_utils.set_game_ended(state)
            else:
                state_utils.set_prev_player_passed(state)

            # Update invalid channel
            state_utils.set_invalid_moves(state, group_map)

            # Switch turn
            state_utils.set_turn(state)

            # Return event
            return state, group_map

        player = state_utils.get_turn(state)
        opponent = 1 - player
        m, n = state_utils.get_board_size(state)

        # convert the move to 2d
        action = (action // m, action % n)

        # Check move is valid
        if not state_utils.is_within_bounds(state, action):
            raise Exception("{} Not Within bounds".format(action))
        elif state[INVD_CHNL, action[0], action[1]] > 0:
            raise Exception("Invalid Move", action, state)

        # Get all adjacent information
        adjacent_locations = state_utils.get_adjacent_locations(state, action)
        adj_own_groups, adj_opp_groups = state_utils.get_adjacent_groups(state, group_map, adjacent_locations, player)

        if not inplace:
            # Start new group map
            group_map = np.copy(group_map)

        # Go through opponent groups
        killed_groups = set()
        single_kill = None
        empty_adjacents_before_kill = adjacent_locations.copy()
        for group in adj_opp_groups:
            assert action in group.liberties, (action, group, state[[BLACK, WHITE]])
            empty_adjacents_before_kill.difference_update(group.locations)
            if len(group.liberties) <= 1:
                # Killed group
                killed_groups.add(group)

                # Remove group in board and group map
                for loc in group.locations:
                    group_map[loc] = None
                    state[opponent, loc[0], loc[1]] = 0

                # Metric for ko-protection
                if len(group.locations) <= 1:
                    if single_kill is not None:
                        single_kill = None
                    else:
                        single_kill = next(iter(group.locations))
        adj_opp_groups.difference_update(killed_groups)

        # Add the piece!
        state[player, action[0], action[1]] = 1

        # Update surviving adjacent opponent groups by removing liberties by the new action
        for opp_group in adj_opp_groups:
            assert action in opp_group.liberties, (action, opp_group, adj_opp_groups)

            if not inplace:
                # New group copy
                opp_group = opp_group.copy()
                for loc in opp_group.locations:
                    group_map[loc] = opp_group

            opp_group.liberties.remove(action)

        # Update adjacent own groups that are merged with the action
        merged_group = Group()
        merged_group.locations.add(action)
        for adj_loc in adjacent_locations:
            if np.count_nonzero(state[[BLACK, WHITE], adj_loc[0], adj_loc[1]]) == 0:
                merged_group.liberties.add(adj_loc)

        for own_group in adj_own_groups:
            merged_group.locations.update(own_group.locations)
            merged_group.liberties.update(own_group.liberties)
        if action in merged_group.liberties:
            merged_group.liberties.remove(action)

        for loc in merged_group.locations:
            group_map[loc] = merged_group

        if len(killed_groups) > 0:
            killed_map = np.zeros(state.shape[1:])
            for group in killed_groups:
                for loc in group.locations:
                    killed_map[loc] = 1
            # Update own groups adjacent to opponent groups that we just killed
            killed_liberties = ndimage.binary_dilation(killed_map)
            affected_group_matrix = state[player] * killed_liberties
            groups_to_update = set(group_map[np.nonzero(affected_group_matrix)])
            all_pieces = np.sum(state[[BLACK, WHITE]], axis=0)
            empties = (1 - all_pieces)
            for group in groups_to_update:
                group_matrix = group_map == group
                additional_liberties = ndimage.binary_dilation(group_matrix) * empties * killed_map
                additional_liberties = np.argwhere(additional_liberties)

                if not inplace:
                    group = group.copy()
                    for loc in group.locations:
                        group_map[loc] = group

                for liberty in additional_liberties:
                    group.liberties.add(tuple(liberty))


        # Update illegal moves
        state_utils.set_invalid_moves(state, group_map)

        # If group was one piece, and location is surrounded by opponents,
        # activate ko protection
        if single_kill is not None and len(empty_adjacents_before_kill) <= 0:
            state[INVD_CHNL, single_kill[0], single_kill[1]] = 1

        # This move was not a pass
        state_utils.set_prev_player_passed(state, 0)

        # Switch turn
        state_utils.set_turn(state)

        return state, group_map

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
        :return: Who's turn it is (BLACK/WHITE)
        """
        return state_utils.get_turn(state)

    @staticmethod
    def get_valid_moves(state):
        # return a fixed size binary vector
        if GoGame.get_game_ended(state):
            return np.zeros(GoGame.get_action_size(state))
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

        m, n = state_utils.get_board_size(state)
        visited = set()
        black_area = 0
        white_area = 0

        # loop through each intersection on board
        for loc in itertools.product(range(m), range(n)):
            # count pieces towards area
            if state[BLACK, loc[0], loc[1]] > 0:
                black_area += 1
            elif state[WHITE, loc[0], loc[1]] > 0:
                white_area += 1

            # do DFS on unvisited territory
            elif loc not in visited:
                player, area = state_utils.explore_territory(state, loc, visited)

                # add area to corresponding player
                if player == BLACK:  # BLACK
                    black_area += area
                elif player == WHITE:  # WHITE
                    white_area += area

        return black_area, white_area

    @staticmethod
    def get_canonical_form(state):
        """
        The returned state is a seperate copy of the given state
        :param state:
        :param player:
        :return:
        """
        state = np.copy(state)

        player = GoGame.get_turn(state)
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
