import random
import unittest

import gym
import numpy as np

from gym_go import govars


class TestGoEnvInvalidMoves(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = gym.make('gym_go:go-v0', size=7, reward_method='real')

    def setUp(self):
        self.env.reset()

    def test_out_of_bounds_action(self):
        with self.assertRaises(Exception):
            self.env.step((-1, 0))

        with self.assertRaises(Exception):
            self.env.step((0, 100))

    def test_invalid_occupied_moves(self):
        # Test this 8 times at random
        for _ in range(8):
            self.env.reset()
            row = random.randint(0, 6)
            col = random.randint(0, 6)

            state, reward, done, info = self.env.step((row, col))

            # Assert that the invalid layer is correct
            self.assertEqual(np.count_nonzero(state[govars.INVD_CHNL]), 1)
            self.assertEqual(np.count_nonzero(state[govars.INVD_CHNL] == 1), 1)
            self.assertEqual(state[govars.INVD_CHNL, row, col], 1)

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

        for move in [(0, 1), (0, 2), (1, 0), (1, 3), (2, 1), (2, 2), (1, 2), (1, 1)]:
            state, reward, done, info = self.env.step(move)

        # Test invalid channel
        self.assertEqual(np.count_nonzero(state[govars.INVD_CHNL]), 8, state[govars.INVD_CHNL])
        self.assertEqual(np.count_nonzero(state[govars.INVD_CHNL] == 1), 8)
        self.assertEqual(state[govars.INVD_CHNL, 1, 2], 1)

        # Assert pieces channel is empty at ko-protection coordinate
        self.assertEqual(state[govars.BLACK, 1, 2], 0)
        self.assertEqual(state[govars.WHITE, 1, 2], 0)

        final_move = (1, 2)
        with self.assertRaises(Exception):
            self.env.step(final_move)

        # Assert ko-protection goes off
        state, reward, done, info = self.env.step((6, 6))
        state, reward, done, info = self.env.step(None)
        self.assertEqual(np.count_nonzero(state[govars.INVD_CHNL]), 8)
        self.assertEqual(np.count_nonzero(state[govars.INVD_CHNL] == 1), 8)
        self.assertEqual(state[govars.INVD_CHNL, 1, 2], 0)

    def test_invalid_ko_wall_protection_moves(self):
        """
      2/8,   7,   6,   _,   _,   _,   _,

        1,   4,   _,   _,   _,   _,   _,

        _,   _,   _,   _,   _,   _,   _,

        _,   _,   _,   _,   _,   _,   _,

        _,   _,   _,   _,   _,   _,   _,

        _,   _,   _,   _,   _,   _,   _,

        _,   _,   _,   _,   _,   _,   _,

        :return:
        """

        for move in [(1, 0), (0, 0), None, (1, 1), None, (0, 2), (0, 1)]:
            state, reward, done, info = self.env.step(move)

        # Test invalid channel
        self.assertEqual(np.count_nonzero(state[govars.INVD_CHNL]), 5, state[govars.INVD_CHNL])
        self.assertEqual(np.count_nonzero(state[govars.INVD_CHNL] == 1), 5)
        self.assertEqual(state[govars.INVD_CHNL, 0, 0], 1)

        # Assert pieces channel is empty at ko-protection coordinate
        self.assertEqual(state[govars.BLACK, 0, 0], 0)
        self.assertEqual(state[govars.WHITE, 0, 0], 0)

        final_move = (0, 0)
        with self.assertRaises(Exception):
            self.env.step(final_move)

        # Assert ko-protection goes off
        state, reward, done, info = self.env.step((6, 6))
        state, reward, done, info = self.env.step(None)
        self.assertEqual(np.count_nonzero(state[govars.INVD_CHNL]), 5)
        self.assertEqual(np.count_nonzero(state[govars.INVD_CHNL] == 1), 5)
        self.assertEqual(state[govars.INVD_CHNL, 0, 0], 0)

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
        for move in [(0, 1), (0, 2), (1, 0), (1, 4), (2, 1), (2, 2), (1, 2)]:
            state, reward, done, info = self.env.step(move)

        # Test invalid channel
        self.assertEqual(np.count_nonzero(state[govars.INVD_CHNL]), 9, state[govars.INVD_CHNL])
        self.assertEqual(np.count_nonzero(state[govars.INVD_CHNL] == 1), 9)
        self.assertEqual(state[govars.INVD_CHNL, 1, 1], 1)
        self.assertEqual(state[govars.INVD_CHNL, 0, 0], 1)
        # Assert empty space in pieces channels
        self.assertEqual(state[govars.BLACK, 1, 1], 0)
        self.assertEqual(state[govars.WHITE, 1, 1], 0)
        self.assertEqual(state[govars.BLACK, 0, 0], 0)
        self.assertEqual(state[govars.WHITE, 0, 0], 0)

        final_move = (1, 1)
        with self.assertRaises(Exception):
            self.env.step(final_move)

    def test_invalid_game_already_over_move(self):
        self.env.step(None)
        self.env.step(None)

        with self.assertRaises(Exception):
            self.env.step(None)

        self.env.reset()

        self.env.step(None)
        self.env.step(None)

        with self.assertRaises(Exception):
            self.env.step((0, 0))

    def test_small_suicide(self):
        """
        7,   8,   0,

        0,   5,   4,

        1,   2,   3/6,
        :return:
        """

        self.env = gym.make('gym_go:go-v0', size=3, reward_method='real')
        for move in [6, 7, 8, 5, 4, 8, 0, 1]:
            state, reward, done, info = self.env.step(move)

        with self.assertRaises(Exception):
            self.env.step(3)

    def test_invalid_after_capture(self):
        """
        1,   5,   6,

        7,   4,   _,

        3,   8,   2,
        :return:
        """

        self.env = gym.make('gym_go:go-v0', size=3, reward_method='real')
        for move in [0, 8, 6, 4, 1, 2, 3, 7]:
            state, reward, done, info = self.env.step(move)

        with self.assertRaises(Exception):
            self.env.step(5)

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
        for move in [(1, 1), (0, 1), (1, 2), (0, 2), (1, 3), (0, 3), (1, 4), (0, 4), (1, 5), (0, 5), (2, 5), (1, 6),
                     (3, 5), (2, 6), (3, 4), (3, 6),
                     (3, 3), (4, 5), (2, 3), (4, 4), (3, 2), (4, 3), (3, 1), (4, 2), (2, 1), (4, 1), None, (3, 0), None,
                     (2, 0), None, (1, 0)]:
            state, reward, done, info = self.env.step(move)

        self.env.step(None)
        final_move = (2, 2)
        with self.assertRaises(Exception):
            self.env.step(final_move)


if __name__ == '__main__':
    unittest.main()
