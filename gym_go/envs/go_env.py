import gym
from gym import error, spaces, utils
from gym.utils import seeding
from itertools import product
import numpy as np
from enum import Enum

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
    metadata = {'render.modes': ['human']}

    def __init__(self, size, reward_method='heuristic', state_ref=None):
        '''
        @param reward_method: either 'heuristic' or 'real' 
        heuristic: gives # black pieces - # white pieces. 
        real: gives 0 for in-game move, 1 for winning, -1 for losing, 
            0 for draw, all from black player's perspective
        '''
        # determine board size
        # A numpy array representing the state of the game
        # Shape [4, SIZE, SIZE]
        # 0 - black, 1 - white, 2 - invalid moves, 3 - previous move was passed
        self.board_size = size
        self.reward_method = RewardMethod(reward_method)
            
        # setup board
        if state_ref is None:
            self.state = np.zeros((4, self.board_size, self.board_size))
        else:
            self.state = state_ref
        self.reset()
        
    def reset(self):
        '''
        Reset state, go_board, curr_player, prev_player_passed,
        done, return state
        '''
        self.state.fill(0)
        self.turn = Turn.BLACK
        self.prev_player_passed = False
        self.ko_protect = None
        self.done = False

        return np.copy(self.state)
        
    def step(self, action):
        ''' 
        Assumes the correct player is making a move. Black goes first.
        return observation, reward, done, info 
        '''
        def _state_reward_done_info():
            return np.copy(self.state), self.get_reward(), self.done, self.get_info()
        
        # check if game is already over
        if self.done:
            raise Exception('Attempt to step at {} after game is over'.format(action))
            
        # if the current player passes
        if action is None:         
            # if two consecutive passes, game is over
            if self.prev_player_passed:
                self.done = True
                
            self.prev_player_passed = True
                
            # Set passing layer
            self.state[3] = 1
            
            # ko-protection is gone
            if self.ko_protect is not None:
                self.state[2][self.ko_protect] = 0
                self.ko_protect = None
                
            # Update invalid channel
            self.update_invalid_channel()
                
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
        
        # Get all adjacent groups
        _, opponent_groups = self.get_adjacent_groups(action)
        
        # Disable ko-proection
        self.ko_protect = None
        
        killed_some_opponent_pieces = False
        
        # Go through opponent groups
        killed_single_piece = False
        empty_adjacents_before_kill = self.get_adjacent_locations(action)
        for group in opponent_groups:
            empty_adjacents_before_kill = empty_adjacents_before_kill - group.locations
            if len(group.liberties) <= 1:
                assert action in group.liberties
                killed_some_opponent_pieces = True
                
                # Remove group in board
                for loc in group.locations:
                    self.state[self.turn.other.value][loc] = 0
                    
                # Metric for ko-protection
                if len(group.locations) <= 1:
                    killed_single_piece = True
                
        # If group was one piece, and location is surrounded by opponents, 
        # activate ko protection
        if killed_single_piece and len(empty_adjacents_before_kill) <= 0:
            self.ko_protect = group.locations.pop()
                    
        # Add the piece!
        self.state[self.turn.value][action] = 1

        # Update illegal moves
        self.update_invalid_channel()

        # This move was not a pass
        self.prev_player_passed = False
        # Update passing layer
        self.state[3] = 0
        
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
            if self.done:
                if area_difference == 0:
                    return 0
                elif area_difference > 0:
                    return 1
                else:
                    return -1
            else: 
                return 0

        elif self.reward_method == RewardMethod.HEURISTIC:
            return area_difference
        else:
            raise Exception("Unknown Reward Method")
            
    def update_invalid_channel(self):
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
        self.state[2] = np.sum(self.state[[0,1]], axis=0) # Occupied
        if self.ko_protect is not None:
            self.state[2][self.ko_protect] = 1
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
        board_str += '\tTurn: {} Last Turn Passed: {} Game Over: {}\n'.format(self.turn, self.prev_player_passed, self.done)
        board_str += '\tB: {} W:{}\n'.format(info['area']['b'], info['area']['w'])

        print(board_str)

    def close(self):
        pass

    def print_state(self):
        print("Turn: {}".format(self.curr_player))
        print("Your pieces (black):")
        print(self.state[0])
        print("Opponent's pieces (white):")
        print(self.state[1])
        print("Illegal moves:")
        print(self.state[2])
        print("The opponent passed: {}".format(self.state[3][0][0]))