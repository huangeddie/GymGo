import numpy as np
from scipy import ndimage
from scipy.ndimage import measurements

from gym_go import govars

group_struct = np.array([[[0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0]],
                         [[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]],
                         [[0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0]]])

surround_struct = np.array([[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]])

neighbor_deltas = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])


def compute_invalid_moves(state, player, ko_protect=None):
    """
    Updates invalid moves in the OPPONENT's perspective
    1.) Opponent cannot move at a location
        i.) If it's occupied
        i.) If it's protected by ko
    2.) Opponent can move at a location
        i.) If it can kill
    3.) Opponent cannot move at a location
        i.) If it's adjacent to one of their groups with only one liberty and
            not adjacent to other groups with more than one liberty and is completely surrounded
        ii.) If it's surrounded by our pieces and all of those corresponding groups
            move more than one liberty
    """

    # All pieces and empty spaces
    all_pieces = np.sum(state[[govars.BLACK, govars.WHITE]], axis=0)
    empties = 1 - all_pieces

    # Setup invalid and valid arrays
    possible_invalid_array = np.zeros(state.shape[1:])
    definite_valids_array = np.zeros(state.shape[1:])

    # Get all groups
    all_own_groups, num_own_groups = measurements.label(state[player])
    all_opp_groups, num_opp_groups = measurements.label(state[1 - player])
    expanded_own_groups = np.zeros((num_own_groups, *state.shape[1:]))
    expanded_opp_groups = np.zeros((num_opp_groups, *state.shape[1:]))

    # Expand the groups such that each group is in its own channel
    for i in range(num_own_groups):
        expanded_own_groups[i] = all_own_groups == (i + 1)

    for i in range(num_opp_groups):
        expanded_opp_groups[i] = all_opp_groups == (i + 1)

    # Get all liberties in the expanded form
    all_own_liberties = empties[np.newaxis] * ndimage.binary_dilation(expanded_own_groups, surround_struct[np.newaxis])
    all_opp_liberties = empties[np.newaxis] * ndimage.binary_dilation(expanded_opp_groups, surround_struct[np.newaxis])

    own_liberty_counts = np.sum(all_own_liberties, axis=(1, 2))
    opp_liberty_counts = np.sum(all_opp_liberties, axis=(1, 2))

    # Possible invalids are on single liberties of opponent groups and on multi-liberties of own groups
    # Definite valids are on single liberties of own groups, multi-liberties of opponent groups
    # or you are not surrounded
    possible_invalid_array += np.sum(all_own_liberties[own_liberty_counts > 1], axis=0)
    possible_invalid_array += np.sum(all_opp_liberties[opp_liberty_counts == 1], axis=0)

    definite_valids_array += np.sum(all_own_liberties[own_liberty_counts == 1], axis=0)
    definite_valids_array += np.sum(all_opp_liberties[opp_liberty_counts > 1], axis=0)

    # All invalid moves are occupied spaces + (possible invalids minus the definite valids and it's surrounded)
    surrounded = ndimage.convolve(all_pieces, surround_struct, mode='constant', cval=1) == 4
    invalid_moves = all_pieces + possible_invalid_array * (definite_valids_array == 0) * surrounded

    # Ko-protection
    if ko_protect is not None:
        invalid_moves[ko_protect[0], ko_protect[1]] = 1
    return invalid_moves > 0


def batch_compute_invalid_moves(batch_state, batch_player, batch_ko_protect):
    """
    Updates invalid moves in the OPPONENT's perspective
    1.) Opponent cannot move at a location
        i.) If it's occupied
        i.) If it's protected by ko
    2.) Opponent can move at a location
        i.) If it can kill
    3.) Opponent cannot move at a location
        i.) If it's adjacent to one of their groups with only one liberty and
            not adjacent to other groups with more than one liberty and is completely surrounded
        ii.) If it's surrounded by our pieces and all of those corresponding groups
            move more than one liberty
    """
    batch_idcs = np.arange(len(batch_state))

    # All pieces and empty spaces
    batch_all_pieces = np.sum(batch_state[:, [govars.BLACK, govars.WHITE]], axis=1)
    batch_empties = 1 - batch_all_pieces

    # Setup invalid and valid arrays
    batch_possible_invalid_array = np.zeros(batch_state.shape[:1] + batch_state.shape[2:])
    batch_definite_valids_array = np.zeros(batch_state.shape[:1] + batch_state.shape[2:])

    # Get all groups
    batch_all_own_groups, _ = measurements.label(batch_state[batch_idcs, batch_player], group_struct)
    batch_all_opp_groups, _ = measurements.label(batch_state[batch_idcs, 1 - batch_player], group_struct)

    batch_data = enumerate(zip(batch_all_own_groups, batch_all_opp_groups, batch_empties))
    for i, (all_own_groups, all_opp_groups, empties) in batch_data:
        own_labels = np.unique(all_own_groups)
        opp_labels = np.unique(all_opp_groups)
        own_labels = own_labels[np.nonzero(own_labels)]
        opp_labels = opp_labels[np.nonzero(opp_labels)]
        expanded_own_groups = np.zeros((len(own_labels), *all_own_groups.shape))
        expanded_opp_groups = np.zeros((len(opp_labels), *all_opp_groups.shape))

        # Expand the groups such that each group is in its own channel
        for j, label in enumerate(own_labels):
            expanded_own_groups[j] = all_own_groups == label

        for j, label in enumerate(opp_labels):
            expanded_opp_groups[j] = all_opp_groups == label

        # Get all liberties in the expanded form
        all_own_liberties = empties[np.newaxis] * ndimage.binary_dilation(expanded_own_groups,
                                                                          surround_struct[np.newaxis])
        all_opp_liberties = empties[np.newaxis] * ndimage.binary_dilation(expanded_opp_groups,
                                                                          surround_struct[np.newaxis])

        own_liberty_counts = np.sum(all_own_liberties, axis=(1, 2))
        opp_liberty_counts = np.sum(all_opp_liberties, axis=(1, 2))

        # Possible invalids are on single liberties of opponent groups and on multi-liberties of own groups
        # Definite valids are on single liberties of own groups, multi-liberties of opponent groups
        # or you are not surrounded
        batch_possible_invalid_array[i] += np.sum(all_own_liberties[own_liberty_counts > 1], axis=0)
        batch_possible_invalid_array[i] += np.sum(all_opp_liberties[opp_liberty_counts == 1], axis=0)

        batch_definite_valids_array[i] += np.sum(all_own_liberties[own_liberty_counts == 1], axis=0)
        batch_definite_valids_array[i] += np.sum(all_opp_liberties[opp_liberty_counts > 1], axis=0)

    # All invalid moves are occupied spaces + (possible invalids minus the definite valids and it's surrounded)
    surrounded = ndimage.convolve(batch_all_pieces, surround_struct[np.newaxis], mode='constant', cval=1) == 4
    invalid_moves = batch_all_pieces + batch_possible_invalid_array * (batch_definite_valids_array == 0) * surrounded

    # Ko-protection
    for i, ko_protect in enumerate(batch_ko_protect):
        if ko_protect is not None:
            invalid_moves[i, ko_protect[0], ko_protect[1]] = 1
    return invalid_moves > 0


def update_pieces(state, adj_locs, player):
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
            state[opponent, opp_group_locs[:, 0], opp_group_locs[:, 1]] = 0
            killed_groups.append(opp_group_locs)

    return killed_groups


def batch_update_pieces(batch_non_pass, batch_state, batch_adj_locs, batch_player):
    batch_opponent = 1 - batch_player
    batch_killed_groups = []

    batch_all_pieces = np.sum(batch_state[:, [govars.BLACK, govars.WHITE]], axis=1)
    batch_empties = 1 - batch_all_pieces

    batch_all_opp_groups, _ = ndimage.measurements.label(batch_state[batch_non_pass, batch_opponent],
                                                         group_struct)

    batch_data = enumerate(zip(batch_all_opp_groups, batch_all_pieces, batch_empties, batch_adj_locs, batch_opponent))
    for i, (all_opp_groups, all_pieces, empties, adj_locs, opponent) in batch_data:
        killed_groups = []

        # Go through opponent groups
        all_adj_labels = all_opp_groups[adj_locs[:, 0], adj_locs[:, 1]]
        all_adj_labels = np.unique(all_adj_labels)
        for opp_group_idx in all_adj_labels[np.nonzero(all_adj_labels)]:
            opp_group = all_opp_groups == opp_group_idx
            liberties = empties * ndimage.binary_dilation(opp_group)
            if np.sum(liberties) <= 0:
                # Killed group
                opp_group_locs = np.argwhere(opp_group)
                batch_state[batch_non_pass[i], opponent, opp_group_locs[:, 0], opp_group_locs[:, 1]] = 0
                killed_groups.append(opp_group_locs)

        batch_killed_groups.append(killed_groups)

    return batch_killed_groups


def adj_data(state, action2d, player):
    neighbors = neighbor_deltas + action2d
    valid = (neighbors >= 0) & (neighbors < state.shape[1])
    valid = np.prod(valid, axis=1)
    neighbors = neighbors[np.nonzero(valid)]

    opp_pieces = state[1 - player]
    surrounded = (opp_pieces[neighbors[:, 0], neighbors[:, 1]] > 0).all()

    return neighbors, surrounded


def batch_adj_data(batch_state, batch_action2d, batch_player):
    batch_neighbors, batch_surrounded = [], []
    for state, action2d, player in zip(batch_state, batch_action2d, batch_player):
        neighbors, surrounded = adj_data(state, action2d, player)
        batch_neighbors.append(neighbors)
        batch_surrounded.append(surrounded)
    return batch_neighbors, batch_surrounded


def set_turn(state):
    """
    Swaps turn
    :param state:
    :return:
    """
    state[govars.TURN_CHNL] = 1 - state[govars.TURN_CHNL]


def batch_set_turn(batch_state):
    """
    Swaps turn
    :param state:
    :return:
    """
    batch_state[:, govars.TURN_CHNL] = 1 - batch_state[:, govars.TURN_CHNL]
