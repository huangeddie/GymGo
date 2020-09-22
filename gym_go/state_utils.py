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


def get_invalid_moves(state, player, ko_protect=None):
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
    empties = 1 - all_pieces

    # Possible invalids are on single liberties of opponent groups and on multi-liberties of own groups
    possible_invalid_array = np.zeros(state.shape[1:])
    definite_valids_array = np.zeros(state.shape[1:])

    all_own_groups, num_own_groups = measurements.label(state[player])
    all_opp_groups, num_opp_groups = measurements.label(state[1 - player])
    expanded_own_groups = np.zeros((num_own_groups, *state.shape[1:]))
    expanded_opp_groups = np.zeros((num_opp_groups, *state.shape[1:]))

    for i in range(num_own_groups):
        expanded_own_groups[i] = all_own_groups == (i + 1)

    for i in range(num_opp_groups):
        expanded_opp_groups[i] = all_opp_groups == (i + 1)

    all_own_liberties = empties[np.newaxis] * ndimage.binary_dilation(expanded_own_groups, surround_struct[np.newaxis])
    all_opp_liberties = empties[np.newaxis] * ndimage.binary_dilation(expanded_opp_groups, surround_struct[np.newaxis])


    opp_liberty_counts = np.sum(all_opp_liberties, axis=(1, 2))
    possible_invalid_array += np.sum(all_opp_liberties[opp_liberty_counts == 1], axis=0)
    definite_valids_array += np.sum(all_opp_liberties[opp_liberty_counts > 1], axis=0)

    own_liberty_counts = np.sum(all_own_liberties, axis=(1, 2))
    possible_invalid_array += np.sum(all_own_liberties[own_liberty_counts > 1], axis=0)
    definite_valids_array += np.sum(all_own_liberties[own_liberty_counts == 1], axis=0)

    possible_invalid_array *= (definite_valids_array == 0)

    # Invalid moves
    surrounded = ndimage.convolve(all_pieces, surround_struct, mode='constant', cval=1) == 4
    invalid_moves = surrounded * possible_invalid_array + all_pieces

    # Ko-protection
    if ko_protect is not None:
        invalid_moves[ko_protect[0], ko_protect[1]] = 1
    return invalid_moves > 0


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


def get_liberties(state: np.ndarray):
    blacks = state[govars.BLACK]
    whites = state[govars.WHITE]
    all_pieces = np.sum(state[[govars.BLACK, govars.WHITE]], axis=0)

    liberty_list = []
    for player_pieces in [blacks, whites]:
        liberties = ndimage.binary_dilation(player_pieces, surround_struct)
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
