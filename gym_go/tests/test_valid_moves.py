import unittest

import gym
import numpy as np

from gym_go import govars


class TestGoEnvValidMoves(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = gym.make('gym_go:go-v0', size=7, reward_method='real')

    def setUp(self):
        self.env.reset()

    def test_simple_valid_moves(self):
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
        for move in [(0, 1), (0, 2), (1, 0), (1, 3), (2, 1), (2, 2), (1, 2), (1, 1)]:
            state, reward, done, info = self.env.step(move)

        # Black should have 3 pieces
        self.assertEqual(np.count_nonzero(state[govars.BLACK]), 3)

        # White should have 4 pieces
        self.assertEqual(np.count_nonzero(state[govars.WHITE]), 4)
        # Assert values are ones
        self.assertEqual(np.count_nonzero(state[govars.WHITE] == 1), 4)

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
        for move in [(0, 0), (0, 2), (0, 3), (1, 1), (1, 2), (1, 0)]:
            state, reward, done, info = self.env.step(move)

        # Test invalid channel
        self.assertEqual(np.count_nonzero(state[govars.INVD_CHNL]), 6, state[govars.INVD_CHNL])
        self.assertEqual(np.count_nonzero(state[govars.INVD_CHNL] == 1), 6)
        self.assertEqual(state[govars.INVD_CHNL, 0, 1], 0, state[govars.INVD_CHNL])
        # Assert empty space in pieces channels
        self.assertEqual(state[govars.BLACK, 0, 1], 0)
        self.assertEqual(state[govars.WHITE, 0, 1], 0)

        final_move = (0, 1)
        state, reward, done, info = self.env.step(final_move)

        # White should only have 2 pieces
        self.assertEqual(np.count_nonzero(state[govars.WHITE]), 2, state[govars.WHITE])
        self.assertEqual(np.count_nonzero(state[govars.WHITE] == 1), 2)
        # Black should have 4 pieces
        self.assertEqual(np.count_nonzero(state[govars.BLACK]), 4, state[govars.BLACK])
        self.assertEqual(np.count_nonzero(state[govars.BLACK] == 1), 4)

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

        for move in [(0, 1), (1, 1), (1, 0), None, (1, 2), None, (2, 1)]:
            state, reward, done, info = self.env.step(move)

        # White should have no pieces
        self.assertEqual(np.count_nonzero(state[govars.WHITE]), 0)

        # Black should have 4 pieces
        self.assertEqual(np.count_nonzero(state[govars.BLACK]), 4)
        # Assert values are ones
        self.assertEqual(np.count_nonzero(state[govars.BLACK] == 1), 4)

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
        for move in [(2, 2), (1, 2), (2, 3), (1, 3), (2, 4), (1, 4), (3, 4), (2, 5), (3, 3), (3, 5), (3, 2), (4, 4),
                     None, (4, 3), None, (4, 2), None,
                     (3, 1), None, (2, 1)]:
            state, reward, done, info = self.env.step(move)

        # Black should have no pieces
        self.assertEqual(np.count_nonzero(state[govars.BLACK]), 0)

        # White should have 10 pieces
        self.assertEqual(np.count_nonzero(state[govars.WHITE]), 10)
        # Assert they are ones
        self.assertEqual(np.count_nonzero(state[govars.WHITE] == 1), 10)

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
        for move in [(4, 0), (6, 0), (4, 1), (5, 0), (5, 2), (5, 1), (6, 2)]:
            state, reward, done, info = self.env.step(move)

        # Test invalid channel
        self.assertEqual(np.count_nonzero(state[govars.INVD_CHNL]), 8, state[govars.INVD_CHNL])
        self.assertEqual(np.count_nonzero(state[govars.INVD_CHNL] == 1), 8)
        # Assert empty space in pieces channels
        self.assertEqual(state[govars.BLACK, 6, 1], 0)
        self.assertEqual(state[govars.WHITE, 6, 1], 0)

        final_move = (6, 1)
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

        for move in [(0, 0), (0, 2), (0, 1), (1, 2), (1, 1), (2, 1), (1, 0), (2, 0)]:
            state, reward, done, info = self.env.step(move)

        # Black should have no pieces
        self.assertEqual(np.count_nonzero(state[govars.BLACK]), 0)

        # White should have 4 pieces
        self.assertEqual(np.count_nonzero(state[govars.WHITE]), 4)
        # Assert they are ones
        self.assertEqual(np.count_nonzero(state[govars.WHITE] == 1), 4)

    def test_group_kill_no_ko_protection(self):
        """
        Thanks to DeepGeGe for finding this bug.

        _,   _,   _,   _,   2,   1,  13,

        _,   _,   _,   _,   4,   3,  12/14,

        _,   _,   _,   _,   6,   5,   7,

        _,   _,   _,   _,   _,   8,  10,

        _,   _,   _,   _,   _,   _,   _,

        _,   _,   _,   _,   _,   _,   _,

        _,   _,   _,   _,   _,   _,   _,

        :return:
        """

        for move in [(0, 5), (0, 4), (1, 5), (1, 4), (2, 5), (2, 4), (2, 6), (3, 5), None, (3, 6), None, (1, 6),
                     (0, 6)]:
            state, reward, done, info = self.env.step(move)

        # Test final kill move (1, 6) is valid
        final_move = (1, 6)
        self.assertEqual(state[govars.INVD_CHNL, 1, 6], 0)
        state, _, _, _ = self.env.step(final_move)

        # Assert black is removed
        self.assertEqual(state[govars.BLACK].sum(), 0)

        # Assert 6 white pieces still on the board
        self.assertEqual(state[govars.WHITE].sum(), 6)


if __name__ == '__main__':
    unittest.main()
