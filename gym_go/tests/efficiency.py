import time
import unittest

import gym
import gym_go
import numpy as np
from tqdm import tqdm


class Efficiency(unittest.TestCase):
    boardsize = 9
    iterations = 64

    def testOrderedTrajs(self):
        self.env = gym.make('go-v0', size=self.boardsize, reward_method='real')
        self.doOrderedTrajs()

    def testOrderedTrajsSuperKo(self):
        self.env = gym.make('go-v0', size=self.boardsize, reward_method='real', super_ko=True)
        self.doOrderedTrajs('super ko')

    def testLowerBound(self):
        self.env = gym.make('go-v0', size=self.boardsize, reward_method='real')
        self.doLowerBound()

    def testLowerBoundSuperKo(self):
        self.env = gym.make('go-v0', size=self.boardsize, reward_method='real', super_ko=True)
        self.doLowerBound('super ko')

    def testRandTrajsWithChildren(self):
        self.env = gym.make('go-v0', size=self.boardsize, reward_method='real')
        self.doRandTrajsWithChildren()

    def testRandTrajsWithChildrenSuperKo(self):
        self.env = gym.make('go-v0', size=self.boardsize, reward_method='real', super_ko=True)
        self.doRandTrajsWithChildren('super ko')


    def doOrderedTrajs(self, msg=''):
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
        if msg != '':
            msg = f' ({msg})'
        print(f"Ordered Trajs{msg}: {avg_time:.3f} AVG, {std_time:.3f} STD", flush=True)


    def doLowerBound(self, msg=''):
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
        if msg != '':
            msg = f' ({msg})'
        print(f"Lower bound{msg}: {avg_time:.3f} AVG, {std_time:.3f} STD", flush=True)

    def doRandTrajsWithChildren(self, msg=''):
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
        if msg != '':
            msg = f' ({msg})'
        print(f"Rand Trajs w/ Children{msg}: {avg_time:.3f} AVG SEC, {std_time:.3f} STD SEC, {avg_steps:.1f} AVG STEPS",
              flush=True)


if __name__ == '__main__':
    unittest.main()
