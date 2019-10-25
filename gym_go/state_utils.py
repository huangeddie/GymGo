import itertools
import queue

import numpy as np
from gym_go.govars import ANYONE, NOONE, BLACK, WHITE, TURN_CHNL, INVD_CHNL, PASS_CHNL, DONE_CHNL, Group

"""
All set operations are in-place operations
"""


def get_adjacent_groups(state, location):
    """
    Returns (turn's groups, other turn's groups)
    """
    our_groups = []
    opponent_groups = []

    player = get_turn(state)

    adjacent_locations = get_adjacent_locations(state, location)
    for loc in adjacent_locations:
        our_group = get_group(state, player, loc)
        opponent_group = get_group(state, 1 - player, loc)

        if our_group is not None:
            our_groups.append(our_group)
        if opponent_group is not None:
            opponent_groups.append(opponent_group)

    return our_groups, opponent_groups


def get_group(state, player, loc):
    """
    Returns the group containing the location or None if location is empty there
    """

    if state[player][loc] <= 0:
        return None

    m, n = get_board_size(state)
    visited = np.zeros((m, n), dtype=np.bool)
    group = Group()
    q = queue.SimpleQueue()

    # Mark location as visited
    visited[loc] = True
    q.put(loc)

    while not q.empty():
        loc = q.get()
        if state[player][loc] > 0:
            # Part of group
            group.locations.add(loc)
            # Now search for neighbors
            adj_locs = get_adjacent_locations(state, loc)
            for neighbor in adj_locs:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    q.put(neighbor)
        elif state[1 - player][loc] <= 0:
            # Part of liberty
            group.liberties.add(loc)

    return group


def is_within_bounds(state, location):
    m, n = get_board_size(state)

    return 0 <= location[0] < m and 0 <= location[1] < n


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


def explore_territory(state, location, visited):
    """
    Return which player this territory belongs to (can be None).
    Will visit all empty intersections connected to
    the initial location.
    :param state:
    :param location:
    :param visited:
    :return: PLAYER, TERRITORY SIZE
    PLAYER may be 0 - BLACK, 1 - WHITE or None - NO PLAYER
    """

    # mark this as visited
    visited[location] = True

    # Frontier
    q = queue.SimpleQueue()
    q.put(location)

    teri_size = 1
    possible_owner = set()

    while not q.empty():
        location = q.get()
        adj_locs = get_adjacent_locations(state, location)
        for adj_loc in adj_locs:
            if visited[adj_loc]:
                continue

            if state[0][adj_loc] > 0:
                possible_owner.add(BLACK)
            elif state[1][adj_loc] > 0:
                possible_owner.add(WHITE)
            else:
                visited[adj_loc] = True
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


def reset_invalid_moves(state):
    """
    In place operator on board
    :param state:
    :return:
    """
    state[INVD_CHNL] = 0


def add_invalid_moves(state):
    """
    Assumes ko-protection is taken care of previously
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
    state[INVD_CHNL] = np.sum(state[[BLACK, WHITE, INVD_CHNL]], axis=0)

    m, n = get_board_size(state)

    for i, j in itertools.product(range(m), range(n)):
        if state[INVD_CHNL][i, j] >= 1:  # Occupied/ko invalidness already taken care of
            continue

        our_groups, opponent_groups = get_adjacent_groups(state, (i, j))

        # Check whether we can kill
        can_kill = False
        for group in our_groups:
            if len(group.liberties) <= 1:
                can_kill = True
                break
        if can_kill:
            continue

        # Check whether completely surrounded,
        # next to a group with only one liberty AND not
        # next to others with more than one liberty
        group_with_one_liberty_exists = False
        group_with_multiple_liberties_exists = False
        completely_surrounded = True
        adjacent_locations = get_adjacent_locations(state, (i, j))
        for loc in adjacent_locations:
            if np.sum(state[[BLACK, WHITE], loc[0], loc[1]]) <= 0:
                completely_surrounded = False
                break
        if completely_surrounded:
            for group in opponent_groups:
                if len(group.liberties) <= 1:
                    group_with_one_liberty_exists = True
                else:
                    assert len(group.liberties) > 1
                    group_with_multiple_liberties_exists = True
                    break

            if group_with_one_liberty_exists and not group_with_multiple_liberties_exists:
                state[INVD_CHNL][i, j] = 1

        if state[INVD_CHNL][i, j] >= 1:
            # Already determined as invalid
            continue

        # Check if surrounded and cannot kill
        empty_adjacent_locations = get_adjacent_locations(state, (i, j))
        can_kill = False
        for group in our_groups:
            empty_adjacent_locations = empty_adjacent_locations - group.locations
            if len(group.liberties) <= 1:
                can_kill = True
                break

        # Check if surrounded and cannot kill
        if len(empty_adjacent_locations) <= 0 and not can_kill:
            state[INVD_CHNL][i, j] = 1


def get_board_size(state):
    assert state.shape[1] == state.shape[2]
    return (state.shape[1], state.shape[2])


def get_turn(state):
    """
    Returns who's turn it is (BLACK/WHITE)
    :param state:
    :return:
    """
    m, n = get_board_size(state)
    if np.count_nonzero(state[TURN_CHNL] == BLACK) == m * n:
        return BLACK
    else:
        return WHITE


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
