import numpy as np
from scipy import ndimage
from scipy.ndimage import measurements

from gym_go import govars

##############################################
# All set operations are in-place operations
##############################################

surround_struct = np.array([[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]])


def get_group_map(state: np.ndarray):
    group_map = [set(), set()]
    all_pieces = np.sum(state[[govars.BLACK, govars.WHITE]], axis=0)
    for player in [govars.BLACK, govars.WHITE]:
        pieces = state[player]
        labels, num_groups = measurements.label(pieces)
        for group_idx in range(1, num_groups + 1):
            group = govars.Group()

            group_matrix = (labels == group_idx)
            liberty_matrix = ndimage.binary_dilation(group_matrix) * (1 - all_pieces)
            liberties = np.argwhere(liberty_matrix)
            for liberty in liberties:
                group.liberties.add(tuple(liberty))

            locations = np.argwhere(group_matrix)
            for loc in locations:
                loc = tuple(loc)
                group.locations.add(loc)
                group_map[player].add(group)

    return group_map


def get_invalid_moves(state, group_map, player, ko_protect=None):
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

    all_pieces = np.sum(state[[govars.BLACK, govars.WHITE]], axis=0)

    # Possible invalids are on single liberties of opponent groups and on multi-liberties of own groups
    possible_invalid_array = np.zeros(state.shape[1:])
    possible_invalids, definite_valids = set(), set()
    own_groups, opp_groups = group_map[player], group_map[1 - player]
    for group in opp_groups:
        if len(group.liberties) == 1:
            possible_invalids.update(group.liberties)
        else:
            # Can connect to other groups with multi liberties
            definite_valids.update(group.liberties)
    for group in own_groups:
        if len(group.liberties) > 1:
            possible_invalids.update(group.liberties)
        else:
            # Can kill
            definite_valids.update(group.liberties)
    possible_invalids.difference_update(definite_valids)

    for loc in possible_invalids:
        possible_invalid_array[loc[0], loc[1]] = 1

    # Invalid moves
    surrounded = ndimage.convolve(all_pieces, surround_struct, mode='constant', cval=1) == 4
    invalid_moves = surrounded * possible_invalid_array + all_pieces

    # Ko-protection
    if ko_protect is not None:
        invalid_moves[ko_protect[0], ko_protect[1]] = 1
    return invalid_moves


def get_adj_data(state, action2d):
    neighbors = []
    if action2d[0] > 0:
        # Up
        neighbors.append((action2d[0] - 1, action2d[1]))
    if action2d[0] < state.shape[1] - 1:
        # Down
        neighbors.append((action2d[0] + 1, action2d[1]))
    if action2d[1] > 0:
        # Left
        neighbors.append((action2d[0], action2d[1] - 1))
    if action2d[1] < state.shape[2] - 1:
        # Right
        neighbors.append((action2d[0], action2d[1] + 1))

    all_pieces = np.sum(state[[govars.BLACK, govars.WHITE]], axis=0)
    surrounded = True
    for loc in neighbors:
        if all_pieces[loc] != 1:
            surrounded = False
            break

    return neighbors, surrounded


def get_adjacent_groups(group_map, adjacent_locations, player):
    our_groups, opponent_groups = set(), set()
    for adj_loc in adjacent_locations:
        adj_loc = tuple(adj_loc)
        found = False
        for group in group_map[player]:
            if adj_loc in group.locations:
                our_groups.add(group)
                found = True
                break

        if not found:
            for group in group_map[1 - player]:
                if adj_loc in group.locations:
                    opponent_groups.add(group)
                    break
    return our_groups, opponent_groups


def get_liberties(state: np.ndarray):
    blacks = state[govars.BLACK]
    whites = state[govars.WHITE]
    all_pieces = np.sum(state[[govars.BLACK, govars.WHITE]], axis=0)

    liberty_list = []
    for player_pieces in [blacks, whites]:
        liberties = ndimage.binary_dilation(player_pieces)
        liberties *= (1 - all_pieces).astype(np.bool)
        liberty_list.append(liberties)

    return liberty_list[0], liberty_list[1]


def get_num_liberties(state: np.ndarray):
    '''
    :param state:
    :return: Total black and white liberties
    '''
    black_liberties, white_liberties = get_liberties(state)
    black_liberties = np.count_nonzero(black_liberties)
    white_liberties = np.count_nonzero(white_liberties)

    return black_liberties, white_liberties


def get_board_size(state):
    assert state.shape[1] == state.shape[2]
    return (state.shape[1], state.shape[2])


def get_turn(state):
    """
    Returns who's turn it is (govars.BLACK/govars.WHITE)
    :param state:
    :return:
    """
    return int(state[govars.TURN_CHNL, 0, 0])


def set_turn(state):
    """
    Swaps turn
    :param state:
    :return:
    """
    state[govars.TURN_CHNL] = 1 - state[govars.TURN_CHNL]
