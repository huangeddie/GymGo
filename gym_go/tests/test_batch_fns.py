import unittest

from gym_go import gogame, govars


class TestBatchFns(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        pass

    def test_batch_canonical_form(self):
        states = gogame.batch_init_state(2, 7)
        states[0] = gogame.next_state(states[0], 0)

        self.assertEqual(states[0, govars.BLACK].sum(), 1)
        self.assertEqual(states[0, govars.WHITE].sum(), 0)

        states = gogame.batch_canonical_form(states)

        self.assertEqual(states[0, govars.BLACK].sum(), 0)
        self.assertEqual(states[0, govars.WHITE].sum(), 1)

        self.assertEqual(states[1, govars.BLACK].sum(), 0)
        self.assertEqual(states[1, govars.WHITE].sum(), 0)

        for i in range(2):
            self.assertEqual(gogame.turn(states[i]), govars.BLACK)

        canon_again = gogame.batch_canonical_form(states)

        self.assertTrue((canon_again == states).all())


if __name__ == '__main__':
    unittest.main()
