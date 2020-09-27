import unittest

import gym
import numpy as np

from gym_go import govars


class TestGoEnvBasics(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = gym.make('gym_go:go-v0', size=7, reward_method='real')

    def setUp(self):
        self.env.reset()

    def test_state(self):
        env = gym.make('gym_go:go-v0', size=7)
        state = env.reset()
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.shape[0], govars.NUM_CHNLS)

        env.close()

    def test_board_sizes(self):
        expected_sizes = [7, 13, 19]

        for expec_size in expected_sizes:
            env = gym.make('gym_go:go-v0', size=expec_size)
            state = env.reset()
            self.assertEqual(state.shape[1], expec_size)
            self.assertEqual(state.shape[2], expec_size)

            env.close()

    def test_empty_board(self):
        state = self.env.reset()
        self.assertEqual(np.count_nonzero(state), 0)

    def test_reset(self):
        state, reward, done, info = self.env.step((0, 0))
        self.assertEqual(np.count_nonzero(state[[govars.BLACK, govars.WHITE, govars.INVD_CHNL]]), 2)
        self.assertEqual(np.count_nonzero(state), 51)
        state = self.env.reset()
        self.assertEqual(np.count_nonzero(state), 0)

    def test_preserve_original_state(self):
        state = self.env.reset()
        original_state = np.copy(state)
        self.env.gogame.next_state(state, 0)
        assert (original_state == state).all()

    def test_black_moves_first(self):
        """
        Make a move at 0,0 and assert that a black piece was placed
        :return:
        """
        next_state, reward, done, info = self.env.step((0, 0))
        self.assertEqual(next_state[govars.BLACK, 0, 0], 1)
        self.assertEqual(next_state[govars.WHITE, 0, 0], 0)

    def test_turns(self):
        for i in range(7):
            # For the first move at i == 0, black went so now it should be white's turn
            state, reward, done, info = self.env.step((i, 0))
            self.assertIn('turn', info)
            self.assertEqual(info['turn'], 1 if i % 2 == 0 else 0)

    def test_multiple_action_formats(self):
        for _ in range(10):
            action_1d = np.random.randint(50)
            action_2d = None if action_1d == 49 else (action_1d // 7, action_1d % 7)

            self.env.reset()
            state_from_1d, _, _, _ = self.env.step(action_1d)

            self.env.reset()
            state_from_2d, _, _, _ = self.env.step(action_2d)

            self.assertTrue((state_from_1d == state_from_2d).all())

    def test_passing(self):
        """
        None indicates pass
        :return:
        """

        # Pass on first move
        state, reward, done, info = self.env.step(None)
        # Expect empty board still
        self.assertEqual(np.count_nonzero(state[[govars.BLACK, govars.WHITE]]), 0)
        # Expect passing layer and turn layer channels to be all ones
        self.assertEqual(np.count_nonzero(state), 98, state)
        self.assertEqual(np.count_nonzero(state[govars.PASS_CHNL]), 49)
        self.assertEqual(np.count_nonzero(state[govars.PASS_CHNL] == 1), 49)

        self.assertIn('turn', info)
        self.assertEqual(info['turn'], 1)

        # Make a move
        state, reward, done, info = self.env.step((0, 0))

        # Expect the passing layer channel to be empty
        self.assertEqual(np.count_nonzero(state), 2)
        self.assertEqual(np.count_nonzero(state[govars.WHITE]), 1)
        self.assertEqual(np.count_nonzero(state[govars.WHITE] == 1), 1)
        self.assertEqual(np.count_nonzero(state[govars.PASS_CHNL]), 0)

        # Pass on second move
        self.env.reset()
        state, reward, done, info = self.env.step((0, 0))
        # Expect two pieces (one in the invalid channel)
        # Plus turn layer is all ones
        self.assertEqual(np.count_nonzero(state), 51, state)
        self.assertEqual(np.count_nonzero(state[[govars.BLACK, govars.WHITE, govars.INVD_CHNL]]), 2, state)

        self.assertIn('turn', info)
        self.assertEqual(info['turn'], 1)

        # Pass
        state, reward, done, info = self.env.step(None)
        # Expect two pieces (one in the invalid channel)
        self.assertEqual(np.count_nonzero(state[[govars.BLACK, govars.WHITE, govars.INVD_CHNL]]), 2,
                         state[[govars.BLACK, govars.WHITE, govars.INVD_CHNL]])
        self.assertIn('turn', info)
        self.assertEqual(info['turn'], 0)

    def test_game_ends(self):
        state, reward, done, info = self.env.step(None)
        self.assertFalse(done)
        state, reward, done, info = self.env.step(None)
        self.assertTrue(done)

        self.env.reset()

        state, reward, done, info = self.env.step((0, 0))
        self.assertFalse(done)
        state, reward, done, info = self.env.step(None)
        self.assertFalse(done)
        state, reward, done, info = self.env.step(None)
        self.assertTrue(done)

    def test_game_does_not_end_with_disjoint_passes(self):
        state, reward, done, info = self.env.step(None)
        self.assertFalse(done)
        state, reward, done, info = self.env.step((0, 0))
        self.assertFalse(done)
        state, reward, done, info = self.env.step(None)
        self.assertFalse(done)

    def test_num_liberties(self):
        env = gym.make('gym_go:go-v0', size=7)

        steps = [(0, 0), (0, 1)]
        libs = [(2, 0), (1, 2)]

        env.reset()
        for step, libs in zip(steps, libs):
            state, _, _, _ = env.step(step)
            blacklibs, whitelibs = env.gogame.num_liberties(state)
            self.assertEqual(blacklibs, libs[0], state)
            self.assertEqual(whitelibs, libs[1], state)

        steps = [(2, 1), None, (1, 2), None, (2, 3), None, (3, 2), None]
        libs = [(4, 0), (4, 0), (6, 0), (6, 0), (8, 0), (8, 0), (9, 0), (9, 0)]

        env.reset()
        for step, libs in zip(steps, libs):
            state, _, _, _ = env.step(step)
            blacklibs, whitelibs = env.gogame.num_liberties(state)
            self.assertEqual(blacklibs, libs[0], state)
            self.assertEqual(whitelibs, libs[1], state)

    def test_komi(self):
        env = gym.make('gym_go:go-v0', size=7, komi=2.5, reward_method='real')

        # White win
        _ = env.step(None)
        state, reward, done, info = env.step(None)
        self.assertEqual(-1, reward)

        # White still win
        env.reset()
        _ = env.step(0)
        _ = env.step(2)

        _ = env.step(1)
        _ = env.step(None)

        state, reward, done, info = env.step(None)
        self.assertEqual(-1, reward)

        # Black win
        env.reset()
        _ = env.step(0)
        _ = env.step(None)

        _ = env.step(1)
        _ = env.step(None)

        _ = env.step(2)
        _ = env.step(None)
        state, reward, done, info = env.step(None)
        self.assertEqual(1, reward)

        env.close()

    def test_children(self):
        for canonical in [False, True]:
            for _ in range(20):
                action = self.env.uniform_random_action()
                self.env.step(action)
            state = self.env.state()
            children = self.env.children(canonical, padded=True)
            valid_moves = self.env.valid_moves()
            for a in range(len(valid_moves)):
                if valid_moves[a]:
                    child = self.env.gogame.next_state(state, a, canonical)
                    equal = children[a] == child
                    self.assertTrue(equal.all(), (canonical, np.argwhere(~equal)))
                else:
                    self.assertTrue((children[a] == 0).all())

    def test_real_reward(self):
        env = gym.make('gym_go:go-v0', size=7, reward_method='real')

        # In game
        state, reward, done, info = env.step((0, 0))
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

        # Tie
        env.reset()

        state, reward, done, info = env.step(None)
        self.assertEqual(reward, 0)
        state, reward, done, info = env.step(None)
        self.assertEqual(reward, 0)

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


if __name__ == '__main__':
    unittest.main()
