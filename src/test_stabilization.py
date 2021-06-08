import unittest

from stabilization import stabilization_tracker

class StabilizationTracker(unittest.TestCase):

    def setUp(self):
        pass

    def test_stabilize_unstabilize(self):
        indices = [0, 1, 2, 3, 4]
        heuristics = [10, 1, 4, 3, 5]
        tracker = stabilization_tracker(indices, heuristics)

        self.assertEqual(tracker.stabilize(0), [])
        self.assertEqual(tracker.stabilize(2), [0, 4])
        self.assertEqual(tracker.unstabilize(1), [4])
        self.assertEqual(tracker.stabilize(2), [4])
        self.assertEqual(tracker.unstabilize(1), [4])
        self.assertEqual(tracker.stabilize(3), [4, 2])




class StabilizeLogN(unittest.TestCase):

    def setUp(self):
        pass

if __name__ == '__main__':
    unittest.main()
