import unittest

import numpy as np

import gym
import gym_go
from gym_go import gogame
from gym_go import state_utils

class TestGoEnvSuperKo(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = gym.make('go-v0', size=2, super_ko=True)

    def setUp(self):
        self.env.reset()

    def test_initial_history(self):
        self.assertEqual(self.env.history, [])

    def test_initial_history_no_super_ko(self):
        self.env = gym.make('go-v0', size=2, super_ko=False)
        self.assertEqual(self.env.history, None)

    def test_step_builds_history(self):
        self.env.step((0, 0))
        self.assertEqual(len(self.env.history), 1)

    def test_step_ignores_history_no_super_ko(self):
        self.env = gym.make('go-v0', size=2, super_ko=False)
        self.env.reset()
        self.env.step((0, 0))
        self.assertEqual(self.env.history, None)

    def test_reset_clears_history(self):
        self.env.step((0, 0))
        self.assertNotEqual(self.env.history, [])
        self.env.reset()
        self.assertEqual(self.env.history, [])

    def test_reset_clears_history_no_super_ko(self):
        self.env = gym.make('go-v0', size=2, super_ko=False)
        self.env.reset()
        self.env.step((0, 0))
        self.assertEqual(self.env.history, None)
        self.env.reset()
        self.assertEqual(self.env.history, None)

    def test_invalid_moves(self):
        """Given an empty board and a history with a move, that same move should be invalid"""
        state = gogame.init_state(2)
        history = [gogame.next_state(state, 0)]

        invalid_moves = state_utils.compute_invalid_moves(state, 0, ko_protect=None, history=history)

        self.assertTrue((invalid_moves == [[1, 0], [0, 0]]).all())

    def test_batch_invalid_moves(self):
        """Given an empty board and a history with a move, that same move should be invalid"""
        state = gogame.init_state(2)
        history = [gogame.next_state(state, 0)]

        invalid_moves = state_utils.batch_compute_invalid_moves(
            np.expand_dims(state, 0),
            np.array([0]),
            batch_ko_protect=np.array([None]),
            batch_history=np.expand_dims(history, 0)
        )

        self.assertTrue((invalid_moves == [[[1, 0], [0, 0]]]).all())

if __name__ == '__main__':
    unittest.main()
