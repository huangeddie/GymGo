import queue

import numpy as np
from scipy import ndimage
from scipy.ndimage import measurements

from gym_go.govars import ANYONE, NOONE, BLACK, WHITE, TURN_CHNL, INVD_CHNL, PASS_CHNL, DONE_CHNL, Group

"""
All set operations are in-place operations
"""


def get_all_groups(state: np.ndarray):
    group_map = np.empty(state.shape[1:], dtype=object)
    all_pieces = np.sum(state[[BLACK, WHITE]], axis=0)
    for player in [BLACK, WHITE]:
        pieces = state[player]
        labels, num_groups = measurements.label(pieces)
        for group_idx in range(1, num_groups + 1):
            group = Group()

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


def set_invalid_moves(state, group_map):
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

    # Occupied/ko-protection
    all_pieces = np.sum(state[[BLACK, WHITE]], axis=0)

    # Possible invalids are on single liberties of opponent groups and on multi-liberties of own groups
    invalid_array = get_possible_invalids(group_map, state)

    structure = np.array([[0, 1, 0],
                          [1, 0, 1],
                          [0, 1, 0]])

    surrounded = ndimage.convolve(all_pieces, structure, mode='constant', cval=1) == 4

    state[INVD_CHNL] = surrounded * invalid_array + all_pieces


def get_possible_invalids(group_map, state):
    invalid_array = np.zeros(state.shape[1:])
    player = get_turn(state)
    possible_invalids = set()
    definite_valids = set()
    own_groups = set(group_map[np.where(state[player])])
    opp_groups = set(group_map[np.where(state[1 - player])])
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
    blacks = state[BLACK]
    whites = state[WHITE]
    all_pieces = np.sum(state[[BLACK, WHITE]], axis=0)

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


def explore_territory(state, loc, visited: set):
    """
    Return which player this territory belongs to (can be None).
    Will visit all empty intersections connected to
    the initial location.
    :param state:
    :param loc:
    :param visited:
    :return: PLAYER, TERRITORY SIZE
    PLAYER may be 0 - BLACK, 1 - WHITE or None - NO PLAYER
    """

    # mark this as visited
    visited.add(loc)

    # Frontier
    q = queue.Queue()
    q.put(loc)

    teri_size = 1
    possible_owner = set()

    while not q.empty():
        loc = q.get()
        adj_locs = get_adjacent_locations(state, loc)
        for adj_loc in adj_locs:
            if adj_loc in visited:
                continue

            if state[BLACK, adj_loc[0], adj_loc[1]] > 0:
                possible_owner.add(BLACK)
            elif state[WHITE, adj_loc[0], adj_loc[1]] > 0:
                possible_owner.add(WHITE)
            else:
                visited.add(adj_loc)
                q.put(adj_loc)
                teri_size += 1

    # filter out ANYONE, and get unique players
    if ANYONE in possible_owner:
        possible_owner.remove(ANYONE)

    # if all directions returned the same player (could be 'n')
    # then return this player
    if len(possible_owner) <= 0:
        belong_to = ANYONE
    elif len(possible_owner) == 1:
        belong_to = possible_owner.pop()

    # if multiple players or it belonged to no one
    else:
        belong_to = NOONE

    return belong_to, teri_size


def get_board_size(state):
    assert state.shape[1] == state.shape[2]
    return (state.shape[1], state.shape[2])


def get_turn(state):
    """
    Returns who's turn it is (BLACK/WHITE)
    :param state:
    :return:
    """
    return int(state[TURN_CHNL, 0, 0])


def set_game_ended(state):
    """
    In place operator on board
    :param state:
    :return:
    """
    state[DONE_CHNL] = 1


def set_turn(state):
    """
    Swaps turn
    :param state:
    :return:
    """
    state[TURN_CHNL] = 1 - state[TURN_CHNL]


def set_prev_player_passed(state, passed=1):
    """
    In place operator on board
    :param state:
    :param passed:
    :return:
    """
    state[PASS_CHNL] = 1 if (passed == True or passed == 1) else 0
