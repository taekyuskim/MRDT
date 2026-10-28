import unittest
from plan_b.mvimgnet.epoch_sampling import EpochPermutation, RankSamples


class SamplingTests(unittest.TestCase):
    def test_each_object_exactly_once_including_batch_boundary(self):
        n = 103
        order = EpochPermutation(n)
        previous = None
        for epoch in range(5):
            sequence = [order[epoch * n + j] for j in range(n)]
            self.assertEqual(sorted(sequence), list(range(n)))
            self.assertNotEqual(sequence, previous)
            previous = sequence

    def test_rank_union_covers_global_stream_and_resume(self):
        indices = [list(RankSamples(0, 20, 16, r, 2)) for r in range(2)]
        self.assertFalse(set(indices[0]) & set(indices[1]))
        self.assertEqual(sorted(indices[0] + indices[1]), list(range(640)))
        for r in range(2):
            self.assertEqual(indices[r], list(RankSamples(0, 7, 16, r, 2)) + list(RankSamples(7, 20, 16, r, 2)))
        continuous, resumed = EpochPermutation(103), EpochPermutation(103)
        expected = {i: continuous[i] for i in range(640)}
        for r in range(2):
            for i in RankSamples(7, 20, 16, r, 2):
                self.assertEqual(resumed[i], expected[i])

    def test_independent_worker_order_does_not_change_samples(self):
        reference = EpochPermutation(103, 7)
        expected = {i: reference[i] for i in range(600)}
        workers = [EpochPermutation(103, 7) for _ in range(4)]
        for i in reversed(range(600)):
            self.assertEqual(workers[i % 4][i], expected[i])
        self.assertNotEqual([EpochPermutation(103, 8)[i] for i in range(103)], [expected[i] for i in range(103)])


if __name__ == '__main__':
    unittest.main(verbosity=2)
