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
            for a in range(self.boardsize):
                self.env.step(a)
            end = time.time()

            dur = end - start
            durs.append(dur)

        avg_time = np.mean(durs)
        std_time = np.std(durs)
        print(f"{avg_time:.3f} AVG, {std_time:.3f} STD")

    def testUnorderedTrajs(self):
        durs = []
        for _ in tqdm(range(self.iterations)):
            start = time.time()
            self.env.reset()

            max_steps = self.boardsize**2
            for s in range(max_steps):
                valid_moves = self.env.get_valid_moves()
                # Do not pass if possible
                if np.sum(valid_moves) > 1:
                    valid_moves[-1] = 0
                probs = valid_moves / np.sum(valid_moves)
                a = np.random.choice(np.arange(self.boardsize ** 2 + 1), p=probs)
                _, _, done, _ = self.env.step(a)
                if done:
                    break

            end = time.time()

            dur = end - start
            durs.append(dur)

        avg_time = np.mean(durs)
        std_time = np.std(durs)
        print(f"{avg_time:.3f} AVG, {std_time:.3f} STD")


if __name__ == '__main__':
    unittest.main()
