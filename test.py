import unittest
import gym
import numpy as np
import random

class TestGoEnv(unittest.TestCase):

    def setUp(self) -> None:
        self.env = gym.make('gym_go:go-v0', size=7, reward_method='real')

    def tearDown(self) -> None:
        self.env.close()

    def test_empty_board(self):
        state = self.env.reset()
        self.assertEqual(np.count_nonzero(state), 0)

    def test_reset(self):
        state, reward, done, info = self.env.step((0,0))
        self.assertEqual(np.count_nonzero(state), 2)
        state = self.env.reset()
        self.assertEqual(np.count_nonzero(state), 0)


    def test_black_moves_first(self):
        """
        Make a move at 0,0 and assert that a black piece was placed
        :return:
        """
        next_state, reward, done, info = self.env.step((0,0))
        self.assertEqual(next_state[0][0, 0], 1)
        self.assertEqual(next_state[1][0, 0], 0)

    def test_simple_valid_moves(self):
        """
        1,2,3,4,5,6,7,
        _,_,_,_,_,_,_,
        _,_,_,_,_,_,_,
        _,_,_,_,_,_,_,
        _,_,_,_,_,_,_,
        _,_,_,_,_,_,_,
        _,_,_,_,_,_,_,


        1,_,_,_,_,_,_,
        _,2,_,_,_,_,_,
        _,_,3,_,_,_,_,
        _,_,_,4,_,_,_,
        _,_,_,_,5,_,_,
        _,_,_,_,_,6,_,
        _,_,_,_,_,_,7,

        1,_,_,_,_,_,_,
        2,_,_,_,_,_,_,
        3,_,_,_,_,_,_,
        4,_,_,_,_,_,_,
        5,_,_,_,_,_,_,
        6,_,_,_,_,_,_,
        7,_,_,_,_,_,_,

        :return:
        """
        for i in range(7):
            state, reward, done, info = self.env.step((0, i))
            self.assertEqual(done, False)

        self.env.reset()

        for i in range(7):
            state, reward, done, info = self.env.step((i, i))
            self.assertEqual(done, False)

        self.env.reset()

        for i in range(7):
            state, reward, done, info = self.env.step((i, 0))
            self.assertEqual(done, False)

    def test_valid_no_liberty_move(self):
        """
        _,   1,   2,   _,   _,   _,   _,

        3,   8,   7,   4,   _,   _,   _,

        _,   5,   6,   _,   _,   _,   _,

        _,   _,   _,   _,   _,   _,   _,

        _,   _,   _,   _,   _,   _,   _,

        _,   _,   _,   _,   _,   _,   _,

        _,   _,   _,   _,   _,   _,   _,


        :return:
        """
        for move in [(0,1),(0,2),(1,0),(1,3),(2,1),(2,2),(1,2),(1,1)]:
            state, reward, done, info = self.env.step(move)

        # Black should have 3 pieces
        self.assertEqual(np.count_nonzero(state[0]), 3)

        # White should have 4 pieces
        self.assertEqual(np.count_nonzero(state[1]), 4)
        # Assert values are ones
        self.assertEqual(np.count_nonzero(state[1] == 1), 4)

    def test_players_alternate(self):
        for i in range(7):
            # For the first move at i == 0, black went so now it should be white's turn
            state, reward, done, info = self.env.step((i, 0))
            self.assertIn('turn', info)
            self.assertEqual(info['turn'], 'w' if i % 2 == 0 else 'b')

    def test_passing(self):
        """
        None indicates pass
        :return:
        """

        # Pass on first move
        state, reward, done, info = self.env.step(None)
        # Expect empty board still
        self.assertEqual(np.count_nonzero(state[:3]), 0)
        # Expect passing layer channel to be all ones
        self.assertEqual(np.count_nonzero(state), 49)
        self.assertEqual(np.count_nonzero(state[3]), 49)
        self.assertEqual(np.count_nonzero(state[3] == 1), 49)

        self.assertIn('turn', info)
        self.assertEqual(info['turn'], 'w')

        # Make a move
        state, reward, done, info = self.env.step((0,0))

        # Expect the passing layer channel to be empty
        self.assertEqual(np.count_nonzero(state), 2)
        self.assertEqual(np.count_nonzero(state[1]), 1)
        self.assertEqual(np.count_nonzero(state[1] == 1), 1)
        self.assertEqual(np.count_nonzero(state[3]), 0)

        # Pass on second move
        self.env.reset()
        state, reward, done, info = self.env.step((0,0))
        # Expect two pieces (one in the invalid channel)
        self.assertEqual(np.count_nonzero(state), 2)
        self.assertIn('turn', info)
        self.assertEqual(info['turn'], 'w')

        # Pass
        state, reward, done, info = self.env.step(None)
        # Expect two pieces (one in the invalid channel)
        self.assertEqual(np.count_nonzero(state[:3]), 2)
        self.assertIn('turn', info)
        self.assertEqual(info['turn'], 'b')

    def test_incorrect_action_format(self):
        with self.assertRaises(Exception):
            self.env.step(0)

    def test_out_of_bounds_action(self):
        with self.assertRaises(Exception):
            self.env.step((-1,0))

        with self.assertRaises(Exception):
            self.env.step((0,100))

    def test_invalid_occupied_moves(self):
        # Test this 8 times at random
        for _ in range(8):
            self.env.reset()
            row = random.randint(0, 6)
            col = random.randint(0, 6)

            state, reward, done, info = self.env.step((row, col))

            # Assert that the invalid layer is correct
            self.assertEqual(np.count_nonzero(state[2]), 1)
            self.assertEqual(np.count_nonzero(state[2] == 1), 1)
            self.assertEqual(state[2][row, col], 1)

            with self.assertRaises(Exception):
                self.env.step((row, col))

    def test_invalid_ko_protection_moves(self):
        """
        _,   1,   2,   _,   _,   _,   _,

        3,   8, 7/9,   4,   _,   _,   _,

        _,   5,   6,   _,   _,   _,   _,

        _,   _,   _,   _,   _,   _,   _,

        _,   _,   _,   _,   _,   _,   _,

        _,   _,   _,   _,   _,   _,   _,

        _,   _,   _,   _,   _,   _,   _,

        :return:
        """

        for move in [(0,1),(0,2),(1,0),(1,3),(2,1),(2,2),(1,2),(1,1)]:
            state, reward, done, info = self.env.step(move)

        # Test invalid channel
        self.assertEqual(np.count_nonzero(state[2]), 8, state[2])
        self.assertEqual(np.count_nonzero(state[2] == 1), 8)
        self.assertEqual(state[2][1, 2], 1)

        # Assert pieces channel is empty at ko-protection coordinate
        self.assertEqual(state[0][1, 2], 0)
        self.assertEqual(state[1][1, 2], 0)

        final_move = (1,2)
        with self.assertRaises(Exception):
            self.env.step(final_move)

        # Assert ko-protection goes off
        state, reward, done, info = self.env.step((6,6))
        state, reward, done, info = self.env.step(None)
        self.assertEqual(np.count_nonzero(state[2]), 8)
        self.assertEqual(np.count_nonzero(state[2] == 1), 8)
        self.assertEqual(state[2][1, 2], 0)

    def test_invalid_no_liberty_move(self):
        """
        _,   1,   2,   _,   _,   _,   _,

        3,   8,   7,   _,   4,   _,   _,

        _,   5,   6,   _,   _,   _,   _,

        _,   _,   _,   _,   _,   _,   _,

        _,   _,   _,   _,   _,   _,   _,

        _,   _,   _,   _,   _,   _,   _,

        _,   _,   _,   _,   _,   _,   _,

        :return:
        """
        for move in [(0,1),(0,2),(1,0),(1,4),(2,1),(2,2),(1,2)]:
            state, reward, done, info = self.env.step(move)

        # Test invalid channel
        self.assertEqual(np.count_nonzero(state[2]), 9, state[2])
        self.assertEqual(np.count_nonzero(state[2] == 1), 9)
        self.assertEqual(state[2][1, 1], 1)
        self.assertEqual(state[2][0, 0], 1)
        # Assert empty space in pieces channels
        self.assertEqual(state[0][1, 1], 0)
        self.assertEqual(state[1][1, 1], 0)
        self.assertEqual(state[0][0, 0], 0)
        self.assertEqual(state[1][0, 0], 0)

        final_move = (1,1)
        with self.assertRaises(Exception):
            self.env.step(final_move)
            
    def test_valid_no_liberty_capture(self):
        """
        1,   7,   2,   3,   _,   _,   _,

        6,   4,   5,   _,   _,   _,   _,

        _,   _,   _,   _,   _,   _,   _,

        _,   _,   _,   _,   _,   _,   _,

        _,   _,   _,   _,   _,   _,   _,

        _,   _,   _,   _,   _,   _,   _,

        _,   _,   _,   _,   _,   _,   _,

        :return:
        """
        for move in [(0,0), (0,2), (0,3), (1,1), (1,2), (1,0)]:
            state, reward, done, info = self.env.step(move)

        # Test invalid channel
        self.assertEqual(np.count_nonzero(state[2]), 6, state[2])
        self.assertEqual(np.count_nonzero(state[2] == 1), 6)
        self.assertEqual(state[2][0, 1], 0, state[2])
        # Assert empty space in pieces channels
        self.assertEqual(state[0][0, 1], 0)
        self.assertEqual(state[1][0, 1], 0)

        final_move = (0,1)
        state, reward, done, info = self.env.step(final_move)
        
        # White should only have 2 pieces
        self.assertEqual(np.count_nonzero(state[1]), 2, state[1])
        self.assertEqual(np.count_nonzero(state[1] == 1), 2)
        # Black should have 4 pieces
        self.assertEqual(np.count_nonzero(state[0]), 4, state[0])
        self.assertEqual(np.count_nonzero(state[0] == 1), 4)

    def test_invalid_game_already_over_move(self):
        self.env.step(None)
        self.env.step(None)

        with self.assertRaises(Exception):
            self.env.step(None)

        self.env.reset()

        self.env.step(None)
        self.env.step(None)

        with self.assertRaises(Exception):
            self.env.step((0,0))


    def test_simple_capture(self):
        """
        _,   1,   _,   _,   _,   _,   _,

        3,   2,   5,   _,   _,   _,   _,

        _,   7,   _,   _,   _,   _,   _,

        _,   _,   _,   _,   _,   _,   _,

        _,   _,   _,   _,   _,   _,   _,

        _,   _,   _,   _,   _,   _,   _,

        _,   _,   _,   _,   _,   _,   _,

        :return:
        """

        for move in [(0,1),(1,1),(1,0),None,(1,2),None,(2,1)]:
            state, reward, done, info = self.env.step(move)

        # White should have no pieces
        self.assertEqual(np.count_nonzero(state[1]), 0)

        # Black should have 4 pieces
        self.assertEqual(np.count_nonzero(state[0]), 4)
        # Assert values are ones
        self.assertEqual(np.count_nonzero(state[0] == 1), 4)


    def test_large_group_capture(self):
        """
        _,   _,   _,   _,   _,   _,   _,

        _,   _,   2,   4,   6,   _,   _,

        _,  20,   1,   3,   5,   8,   _,

        _,  18,  11,   9,  7,  10,   _,

        _,   _,  16,  14,  12,   _,   _,

        _,   _,   _,   _,   _,   _,   _,

        _,   _,   _,   _,   _,   _,   _,

        :return:
        """
        for move in [(2,2),(1,2),(2,3),(1,3),(2,4),(1,4),(3,4),(2,5),(3,3),(3,5),(3,2),(4,4),None,(4,3),None,(4,2),None,
                     (3,1),None,(2,1)]:
            state, reward, done, info = self.env.step(move)

        # Black should have no pieces
        self.assertEqual(np.count_nonzero(state[0]), 0)

        # White should have 10 pieces
        self.assertEqual(np.count_nonzero(state[1]), 10)
        # Assert they are ones
        self.assertEqual(np.count_nonzero(state[1] == 1), 10)

    def test_large_group_suicide(self):
        """
        _,   _,   _,   _,   _,   _,   _,
        
        _,   _,   _,   _,   _,   _,   _,

        _,   _,   _,   _,   _,   _,   _,

        _,   _,   _,   _,   _,   _,   _,

        1,   3,   _,   _,   _,   _,   _,

        4,   6,   5,   _,   _,   _,   _,

        2,   8,   7,   _,   _,   _,   _,
        
        :return:
        """
        for move in [(4,0), (6,0), (4,1), (5,0), (5,2), (5,1), (6,2)]:
            state, reward, done, info = self.env.step(move)
            
        # Test invalid channel
        self.assertEqual(np.count_nonzero(state[2]), 8, state[2])
        self.assertEqual(np.count_nonzero(state[2] == 1), 8)
        # Assert empty space in pieces channels
        self.assertEqual(state[0][6, 1], 0)
        self.assertEqual(state[1][6, 1], 0)

        final_move = (6,1)
        with self.assertRaises(Exception):
            self.env.step(final_move)
        
    def test_group_edge_capture(self):
        """
        1,   3,   2,   _,   _,   _,   _,

        7,   5,   4,   _,   _,   _,   _,

        8,   6,   _,   _,   _,   _,   _,

        _,   _,   _,   _,  _,   _,   _,

        _,   _,   _,   _,   _,   _,   _,

        _,   _,   _,   _,   _,   _,   _,

        _,   _,   _,   _,   _,   _,   _,

        :return:
        """

        for move in [(0,0),(0,2),(0,1),(1,2),(1,1),(2,1),(1,0),(2,0)]:
            state, reward, done, info = self.env.step(move)

        # Black should have no pieces
        self.assertEqual(np.count_nonzero(state[0]), 0)

        # White should have 4 pieces
        self.assertEqual(np.count_nonzero(state[1]), 4)
        # Assert they are ones
        self.assertEqual(np.count_nonzero(state[1] == 1), 4)

    def test_cannot_capture_groups_with_multiple_holes(self):
        """
         _,   2,   4,   6,   8,  10,   _,

        32,   1,   3,   5,   7,   9,  12,

        30,  25,  34,  19,   _,  11,  14,

        28,  23,  21,  17,  15,  13,  16,

         _,  26,  24,  22,  20,  18,   _,

         _,   _,   _,   _,   _,   _,   _,

         _,   _,   _,   _,   _,   _,   _,

        :return:
        """
        for move in [(1,1),(0,1),(1,2),(0,2),(1,3),(0,3),(1,4),(0,4),(1,5),(0,5),(2,5),(1,6),(3,5),(2,6),(3,4),(3,6),
                     (3,3),(4,5),(2,3),(4,4),(3,2),(4,3),(3,1),(4,2),(2,1),(4,1),None,(3,0),None,(2,0),None,(1,0),None]:
            state, reward, done, info = self.env.step(move)

        final_move = (2,2)
        with self.assertRaises(Exception):
            self.env.step(final_move)

    def test_game_ends_with_two_consecutive_passes(self):
        state, reward, done, info = self.env.step(None)
        self.assertFalse(done)
        state, reward, done, info = self.env.step(None)
        self.assertTrue(done)

    def test_game_does_not_end_with_disjoint_passes(self):
        state, reward, done, info = self.env.step(None)
        self.assertFalse(done)
        state, reward, done, info = self.env.step((0,0))
        self.assertFalse(done)
        state, reward, done, info = self.env.step(None)
        self.assertFalse(done)

    def test_state(self):
        env = gym.make('gym_go:go-v0', size=7)
        state = env.reset()
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.shape[0], 4)

        env.close()

    def test_done(self):
        state, reward, done, info = self.env.step((0, 0))
        self.assertFalse(done)
        state, reward, done, info = self.env.step(None)
        self.assertFalse(done)
        state, reward, done, info = self.env.step(None)
        self.assertTrue(done)

    def test_real_reward(self):
        env = gym.make('gym_go:go-v0', size=7, reward_method='real')

        # In game
        state, reward, done, info = env.step((0,0))
        self.assertEqual(reward, 0)
        state, reward, done, info = env.step(None)
        self.assertEqual(reward, 0)

        # Win
        state, reward, done, info = env.step(None)
        self.assertEqual(reward, 1)

        # Lose
        env.reset()

        state, reward, done, info = env.step(None)
        self.assertEqual(reward, 0)
        state, reward, done, info = env.step((0, 0))
        self.assertEqual(reward, 0)
        state, reward, done, info = env.step(None)
        self.assertEqual(reward, 0)
        state, reward, done, info = env.step(None)
        self.assertEqual(reward, -1)

        env.close()

    def test_heuristic_reward(self):
        env = gym.make('gym_go:go-v0', size=7, reward_method='heuristic')

        # In game
        state, reward, done, info = env.step((0, 0))
        self.assertEqual(reward, 49)
        state, reward, done, info = env.step((0, 1))
        self.assertEqual(reward, 0)
        state, reward, done, info = env.step(None)
        self.assertEqual(reward, 0)
        state, reward, done, info = env.step((1, 0))
        self.assertEqual(reward, -49)

        # Lose
        state, reward, done, info = env.step(None)
        self.assertEqual(reward, -49)
        state, reward, done, info = env.step(None)
        self.assertEqual(reward, -49)

        # Win
        env.reset()

        state, reward, done, info = env.step((0, 0))
        self.assertEqual(reward, 49)
        state, reward, done, info = env.step(None)
        self.assertEqual(reward, 49)
        state, reward, done, info = env.step(None)
        self.assertEqual(reward, 49)

        env.close()

    def test_board_sizes(self):
        expected_sizes = [7, 13, 19]

        for expec_size in expected_sizes:
            env = gym.make('gym_go:go-v0', size=expec_size)
            state = env.reset()
            self.assertEqual(state.shape[1], expec_size)
            self.assertEqual(state.shape[2], expec_size)

            env.close()

    def test_invalid_env_arguments(self):
        with self.assertRaises(Exception):
            _ = gym.make('gym_go:go-v0', reward_method='foo')
        with self.assertRaises(Exception):
            _ = gym.make('gym_go:go-v0', size='bar')

        

if __name__ == '__main__':
    unittest.main()
