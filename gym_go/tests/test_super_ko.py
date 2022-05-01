import unittest

import gym
import gym_go

class TestGoEnvSuperKo(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = gym.make('go-v0', size=2)

    def setUp(self):
        self.env.reset()

    def test_initial_history(self):
        self.assertEqual(self.env.history, [])

    def test_step_builds_history(self):
        self.env.step((0, 0))
        self.assertEqual(len(self.env.history), 1)

    def test_reset_clears_history(self):
        self.env.step((0, 0))
        self.assertNotEqual(self.env.history, [])
        self.env.reset()
        self.assertEqual(self.env.history, [])


if __name__ == '__main__':
    unittest.main()
