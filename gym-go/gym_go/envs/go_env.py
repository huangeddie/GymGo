import gym
from itertools import product
import numpy as np
from enum import Enum
from functools import reduce

class RewardMethod(Enum):
    REAL = 'real'
    HEURISTIC = 'heuristic'

class Turn(Enum):
    """
    Their value also represents their channel in the state
    """
    BLACK = 0
    WHITE = 1
    NEITHER = 2
    
    @property
    def other(self):
        return Turn(1 - self.value)
    
class Group:
    def __init__(self):
        self.locations = set()
        self.liberties = set()

class GoEnv(gym.Env):
    metadata = {'render.modes': ['terminal']}

    def __init__(self, size, reward_method='real', black_first=True, state_ref=None):
        '''
        @param reward_method: either 'heuristic' or 'real' 
        heuristic: gives # black pieces - # white pieces. 
        real: gives 0 for in-game move, 1 for winning, -1 for losing, 
            0 for draw, all from black player's perspective
        '''
        # determine board size
        # A numpy array representing the state of the game
        # Shape [4, SIZE, SIZE]
        # 0 - black
        # 1 - white
        # 2 - invalid moves (including ko-protection)
        # 3 - 0/1 means previous player didn't or did pass, -1 means game over
        self.board_size = size
        self.reward_method = RewardMethod(reward_method)
            
        # setup board
        if state_ref is None:
            self.state = np.zeros((4, self.board_size, self.board_size))
        else:
            self.state = state_ref
        self.reset(black_first)
        
    def reset(self, black_first=True, state=None):
        '''
        Reset state, go_board, curr_player, prev_player_passed,
        done, return state
        '''
        if state is None:
            self.state.fill(0)
        else:
            self.state = np.copy(state)

        assert self.state[0].shape[0] == self.state[0].shape[1]

        self.board_size = self.state[0].shape[0]
        self.turn = Turn.BLACK if black_first else Turn.WHITE

        assert reduce(lambda result, i: result or (np.count_nonzero(self.state[3] == i) == self.board_size**2, True),
                      [0,1,-1]), self.state[3]

        self.ko_protect = None

        return np.copy(self.state)

    @property
    def prev_player_passed(self):
        return np.count_nonzero(self.state[3] == 1) == self.board_size**2

    def set_prev_player_passed(self, passed):
        self.state[3] = 1 if (passed == True or passed == 1) else 0

    @property
    def game_over(self):
        return np.count_nonzero(self.state[3] == -1) == self.board_size**2

    def set_game_over(self):
        self.state[3] = -1

    def reset_invalid_moves(self):
        self.state[2] = 0

    def step(self, action):
        ''' 
        Assumes the correct player is making a move. Black goes first.
        return observation, reward, done, info 
        '''
        def _state_reward_done_info():
            return np.copy(self.state), self.get_reward(), self.game_over, self.get_info()
        
        # check if game is already over
        if self.game_over:
            raise Exception('Attempt to step at {} after game is over'.format(action))

        # if the current player passes
        if action is None:         
            # if two consecutive passes, game is over
            if self.prev_player_passed:
                self.set_game_over()
            else:
                self.set_prev_player_passed(True)

            # Update invalid channel
            self.reset_invalid_moves()
            self.add_invalid_moves()
                
            # Switch turn
            self.turn = self.turn.other
            
            # Return event
            return _state_reward_done_info()

        # make the move
        assert action is not None
        
        # Check move is valid
        if not self.is_within_bounds(action):
            raise Exception("Not Within bounds")
        elif self.state[2][action] > 0:
            raise Exception("Invalid Move")

        self.reset_invalid_moves()

        # Get all adjacent groups
        _, opponent_groups = self.get_adjacent_groups(action)

        # Go through opponent groups
        killed_single_piece = None
        empty_adjacents_before_kill = self.get_adjacent_locations(action)
        for group in opponent_groups:
            empty_adjacents_before_kill = empty_adjacents_before_kill - group.locations
            if len(group.liberties) <= 1:
                assert action in group.liberties

                # Remove group in board
                for loc in group.locations:
                    self.state[self.turn.other.value][loc] = 0
                    
                # Metric for ko-protection
                if len(group.locations) <= 1:
                    if killed_single_piece is not None:
                        killed_single_piece = None
                    else:
                        killed_single_piece = group.locations.pop()
                
        # If group was one piece, and location is surrounded by opponents, 
        # activate ko protection
        if killed_single_piece is not None and len(empty_adjacents_before_kill) <= 0:
            self.state[2][killed_single_piece] = 1

        # Add the piece!
        self.state[self.turn.value][action] = 1

        # Update illegal moves
        self.add_invalid_moves()

        # This move was not a pass
        self.set_prev_player_passed(False)

        # Switch turn
        self.turn = self.turn.other

        return _state_reward_done_info()

    def get_info(self):
        '''
        :return: {
            turn: 'b' or 'w'
            area: { 'w': white_area, 'b': black_area }
        }
        '''
        black_area, white_area = self.get_areas()
        return {
            'turn': 'b' if self.turn == Turn.BLACK else 'w',
            'area': {
                'w': white_area,
                'b': black_area,
            }
        }

    def get_state(self):
        """
        Returns deep copy of state
        """
        return np.copy(self.state)

    def get_reward(self):
        '''
        Return reward based on reward_method.
        heuristic: black total area - white total area
        real: 0 for in-game move, 1 for winning, -1 for losing, 
            0 for draw, from black player's perspective.
            Winning and losing based on the Area rule
        Area rule definition: https://en.wikipedia.org/wiki/Rules_of_Go#End
        '''
        black_area, white_area = self.get_areas()
        area_difference = black_area - white_area
        
        if self.reward_method == RewardMethod.REAL:
            if self.game_over:
                if area_difference == 0:
                    return 0
                elif area_difference > 0:
                    return 1
                else:
                    return -1
            else: 
                return 0

        elif self.reward_method == RewardMethod.HEURISTIC:
            if self.game_over:
                return (1 if area_difference > 0 else -1) * self.board_size**2
            return area_difference
        else:
            raise Exception("Unknown Reward Method")
            
    def add_invalid_moves(self):
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
        self.state[2] = np.sum(self.state[[0,1,2]], axis=0) # Occupied/ko-protection

        for i, j in product(range(self.board_size), range(self.board_size)):
            if self.state[2][i,j] >= 1: # Occupied/ko invalidness already taken care of
                continue
                
            our_groups, opponent_groups = self.get_adjacent_groups((i, j))

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
            adjacent_locations = self.get_adjacent_locations((i,j))
            for loc in adjacent_locations:
                if np.sum(self.state[[0,1], loc[0], loc[1]]) <= 0:
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
                    self.state[2][i,j] = 1
                    
            if self.state[2][i,j] >= 1: 
                # Already determined as invalid
                continue
                
            # Check if surrounded and cannot kill
            empty_adjacent_locations = self.get_adjacent_locations((i,j))
            can_kill = False
            for group in our_groups:
                empty_adjacent_locations = empty_adjacent_locations - group.locations
                if len(group.liberties) <= 1:
                    can_kill = True
                    break
                
            # Check if surrounded and cannot kill
            if len(empty_adjacent_locations) <= 0 and not can_kill:
                self.state[2][i,j] = 1

    def get_areas(self):
        '''
        Return black area, white area
        Use DFS helper to find territory.
        '''

        visited = np.zeros((self.board_size, self.board_size), dtype=np.bool)
        black_area = 0
        white_area = 0

        # loop through each intersection on board
        for r, c in product(range(self.board_size), repeat=2):
            # count pieces towards area
            if self.state[0][r,c] > 0:
                black_area += 1
            elif self.state[1][r,c] > 0:
                white_area += 1

            # do DFS on unvisited territory
            elif not visited[r, c]:
                player, area = self.explore_territory((r, c), visited)

                # add area to corresponding player
                if player == Turn.BLACK:
                    black_area += area
                elif player == Turn.WHITE:
                    white_area += area

        return black_area, white_area
    
    def get_adjacent_groups(self, location):
        """
        Returns (turn's groups, other turn's groups)
        """
        our_groups = []
        opponent_groups = []
        
        adjacent_locations = self.get_adjacent_locations(location)
        for loc in adjacent_locations:
            our_group = self.get_group(self.turn, loc)
            opponent_group = self.get_group(self.turn.other, loc)
            
            if our_group is not None:
                our_groups.append(our_group)
            if opponent_group is not None:
                opponent_groups.append(opponent_group)
        
        return our_groups, opponent_groups
    
    def get_group(self, turn, location):
        """
        Returns the group containing the location or None if location is empty there
        """
        def calculate_group_helper(group, turn, location, visited):
            # Mark location as visited
            visited[location] = True
            
            if self.state[turn.value][location] > 0:
                # Part of group
                group.locations.add(location)
                # Now search for neighbors
                adjacent_locations = self.get_adjacent_locations(location)
                for loc in adjacent_locations:
                    if not visited[loc]:
                        calculate_group_helper(group, turn, loc, visited)
            elif self.state[turn.other.value][location] <= 0:
                # Part of liberty
                group.liberties.add(location)
                    
        if self.state[turn.value][location] <= 0:
            return None
            
        visited = np.zeros((self.board_size, self.board_size), dtype=np.bool)
        group = Group()
        
        calculate_group_helper(group, turn, location, visited)
        
        return group
    
    def is_within_bounds(self, location):
            return location[0] >= 0 and location[0] < self.board_size \
                and location[1] >= 0 and location[1] < self.board_size
    
    def get_adjacent_locations(self, location):
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
            if self.is_within_bounds(loc):
                adjacent_locations.add(loc)
        return adjacent_locations

    def explore_territory(self, location, visited):
        '''
        Return which player this territory belongs to (can be None).  
        Will visit all empty intersections connected to 
        the initial location.
        '''
        # base case: already visited
        if visited[location]:
            return None, 0
        
        # base case: this is a piece
        if self.state[0][location] > 0:
            return Turn.BLACK, 0
        elif self.state[1][location] > 0:
            return Turn.WHITE, 0
        
        # mark this as visited
        visited[location] = True

        teri_size = 1
        possible_owner = []

        # explore in all directions
        for adj_loc in self.get_adjacent_locations(location):
            # get the expanded area and player that it belongs to
            player, area = self.explore_territory(adj_loc, visited)
            
            # add area to territory size, player to a list
            teri_size += area
            possible_owner.append(player)

        # filter out None, and get unique players
        possible_owner = list(filter(None, set(possible_owner)))

        # if all directions returned the same player (could be 'n')
        # then return this player
        if len(possible_owner) <= 0:
            belong_to = None
        elif len(possible_owner) == 1:
            belong_to = possible_owner[0]

        # if multiple players or it belonged to no one
        else:
            belong_to = Turn.NEITHER

        return belong_to, teri_size

    @property
    def action_space(self):
        '''
        Return a list of moves (tuples) that are legal for the next player
        '''
        result = []
        for r, c in product(range(self.board_size), repeat=2):
            # if the move is not illegal
            if self.state[2][r, c] == 0:
                result.append((r, c))
        # pass is always an valid move
        result.append(None)

        return result
    
    def render(self, mode='terminal'):
        board_str = ' '

        for i in range(self.board_size):
            board_str += '   {}'.format(i)
        board_str += '\n  '
        board_str += '----' * self.board_size + '-'
        board_str += '\n'
        for i in range(self.board_size):
            board_str += '{} |'.format(i)
            for j in range(self.board_size):
                if self.state[0][i,j] == 1:
                    board_str += ' B'
                elif self.state[1][i,j] == 1:
                    board_str += ' W'
                elif self.state[2][i,j] == 1:
                    board_str += ' .'
                else:
                    board_str += '  '

                board_str += ' |'

            board_str += '\n  '
            board_str += '----' * self.board_size + '-'
            board_str += '\n'
        info = self.get_info()
        board_str += '\tTurn: {}, Last Turn Passed: {}, Game Over: {}\n'.format(self.turn.name, self.prev_player_passed, self.game_over)
        board_str += '\tBlack Area: {}, White Area: {}, Reward: {}\n'.format(info['area']['b'], info['area']['w'], self.get_reward())

        print(board_str)

    def print_state(self):
        print("Turn: {}".format(self.curr_player))
        print("Your pieces (black):")
        print(self.state[0])
        print("Opponent's pieces (white):")
        print(self.state[1])
        print("Illegal moves:")
        print(self.state[2])
        print("The opponent passed: {}".format(self.state[3][0][0]))
