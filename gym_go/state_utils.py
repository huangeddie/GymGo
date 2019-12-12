import numpy as np
from scipy import ndimage
from scipy.ndimage import measurements

from gym_go import govars


##############################################
# All set operations are in-place operations
##############################################


def get_group_map(state: np.ndarray):
    group_map = np.empty(state.shape[1:], dtype=object)
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
                group_map[loc] = group

    return group_map


def get_invalid_moves(state, group_map):
    """
    Does not include ko-protection and assumes it will be taken care of elsewhere
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
    invalid_array = get_possible_invalids(group_map, state)

    structure = np.array([[0, 1, 0],
                          [1, 0, 1],
                          [0, 1, 0]])

    surrounded = ndimage.convolve(all_pieces, structure, mode='constant', cval=1) == 4

    return surrounded * invalid_array + all_pieces


def get_possible_invalids(group_map, state):
    invalid_array = np.zeros(state.shape[1:])
    player = get_turn(state)
    possible_invalids = set()
    definite_valids = set()
    own_groups = set(group_map[np.nonzero(state[player])])
    opp_groups = set(group_map[np.nonzero(state[1 - player])])
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
        invalid_array[loc] = 1
    return invalid_array


def get_adjacent_locations(state, location):
    """
    Returns adjacent locations to the specified one
    """

    adjacent_locations = set()
    drs = [-1, 0, 1, 0]
    dcs = [0, 1, 0, -1]

    # explore in all directions
    for dr, dc in zip(drs, dcs):
        # get the expanded area and player that it belongs to
        loc = (location[0] + dr, location[1] + dc)
        if is_within_bounds(state, loc):
            adjacent_locations.add(loc)
    return adjacent_locations


def get_adjacent_groups(state, group_map, adjacent_locations, player):
    our_groups, opponent_groups = set(), set()
    for adj_loc in adjacent_locations:
        group = group_map[adj_loc]
        if group is None:
            continue
        if state[player, adj_loc[0], adj_loc[1]] > 0:
            our_groups.add(group)
        else:
            opponent_groups.add(group)
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


def is_within_bounds(state, location):
    m, n = get_board_size(state)

    return 0 <= location[0] < m and 0 <= location[1] < n


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


def set_game_ended(state):
    """
    In place operator on board
    :param state:
    :return:
    """
    state[govars.DONE_CHNL] = 1


def set_turn(state):
    """
    Swaps turn
    :param state:
    :return:
    """
    state[govars.TURN_CHNL] = 1 - state[govars.TURN_CHNL]


def set_prev_player_passed(state, passed=1):
    """
    In place operator on board
    :param state:
    :param passed:
    :return:
    """
    state[govars.PASS_CHNL] = 1 if (passed == True or passed == 1) else 0
