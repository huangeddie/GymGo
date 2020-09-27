import time
import unittest

import gym
import numpy as np
from tqdm import tqdm


class Efficiency(unittest.TestCase):
    boardsize = 9
    iterations = 64

    def setUp(self) -> None:
        self.env = gym.make('gym_go:go-v0', size=self.boardsize, reward_method='real')

    def testOrderedTrajs(self):
        durs = []
        for _ in tqdm(range(self.iterations)):
            start = time.time()
            self.env.reset()
            for a in range(self.boardsize ** 2 - 2):
                self.env.step(a)
            end = time.time()

            dur = end - start
            durs.append(dur)

        avg_time = np.mean(durs)
        std_time = np.std(durs)
        print(f"Ordered Trajs: {avg_time:.3f} AVG, {std_time:.3f} STD", flush=True)

    def testLowerBound(self):
        durs = []
        for _ in tqdm(range(self.iterations)):
            start = time.time()
            state = self.env.reset()

            max_steps = self.boardsize ** 2
            for s in range(max_steps):
                for _ in range(max_steps - s):
                    np.copy(state)

                pi = np.ones(self.boardsize ** 2 + 1) / (self.boardsize ** 2 + 1)
                a = np.random.choice(np.arange(self.boardsize ** 2 + 1), p=pi)
                np.copy(state)

            end = time.time()

            dur = end - start
            durs.append(dur)

        avg_time = np.mean(durs)
        std_time = np.std(durs)
        print(f"Lower bound: {avg_time:.3f} AVG, {std_time:.3f} STD", flush=True)

    def testRandTrajsWithChildren(self):
        durs = []
        num_steps = []
        for _ in tqdm(range(self.iterations)):
            start = time.time()
            self.env.reset()

            max_steps = 2 * self.boardsize ** 2
            s = 0
            for s in range(max_steps):
                valid_moves = self.env.valid_moves()
                self.env.children(canonical=True)
                # Do not pass if possible
                if np.sum(valid_moves) > 1:
                    valid_moves[-1] = 0
                probs = valid_moves / np.sum(valid_moves)
                a = np.random.choice(np.arange(self.boardsize ** 2 + 1), p=probs)
                state, _, done, _ = self.env.step(a)
                if done:
                    break
            num_steps.append(s)

            end = time.time()

            dur = end - start
            durs.append(dur)

        avg_time = np.mean(durs)
        std_time = np.std(durs)
        avg_steps = np.mean(num_steps)
        print(f"Rand Trajs w/ Children: {avg_time:.3f} AVG SEC, {std_time:.3f} STD SEC, {avg_steps:.1f} AVG STEPS",
              flush=True)


if __name__ == '__main__':
    unittest.main()
