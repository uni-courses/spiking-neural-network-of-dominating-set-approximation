import unittest

from code.src.common_operations import spiking_max


class TestMain(unittest.TestCase):
    """ """

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay = 4

    def test_spiking_max_int(self):
        """ """
        less = 3
        more = 5
        result = spiking_max(less, more)
        expected_result = more
        self.assertEqual(expected_result, result)

    def test_spiking_max_int_swithced_order(self):
        """ """
        less = 3
        more = 5
        result = spiking_max(more, less)
        expected_result = more
        self.assertEqual(expected_result, result)

    def test_spiking_max_float(self):
        """ """
        less = 3.3
        more = 5.5
        result = spiking_max(less, more)
        expected_result = more
        self.assertEqual(expected_result, result)

    def test_spiking_max_float_switched_order(self):
        """ """
        less = 3.3
        more = 5.5
        result = spiking_max(more, less)
        expected_result = more
        self.assertEqual(expected_result, result)

    def test_spiking_max_float_more(self):
        """ """
        less = 101.6
        more = 307.2
        result = spiking_max(less, more)
        expected_result = more
        self.assertEqual(expected_result, result)

    def test_spiking_max_float_switched_order_more(self):
        """ """
        less = 101.6
        more = 307.2
        result = spiking_max(more, less)
        expected_result = more
        self.assertEqual(expected_result, result)

    def test_spiking_max_float_neg(self):
        """ """
        less = -101.6
        more = -307.2

        with self.assertRaises(Exception) as context:
            spiking_max(less, more)

        self.assertTrue(f"Please enter a positive value" in str(context.exception))

    def test_spiking_max_float_switched_order_neg(self):
        """ """
        less = -101.6
        more = -307.2

        with self.assertRaises(Exception) as context:
            spiking_max(more, less)

        self.assertTrue(f"Please enter a positive value" in str(context.exception))

    def test_spiking_max_float_posneg(self):
        """ """
        less = -10
        more = 10.4
        with self.assertRaises(Exception) as context:
            spiking_max(less, more)

        self.assertTrue(f"Please enter a positive value" in str(context.exception))

    def test_spiking_max_float_switched_order_posneg(self):
        """ """
        less = -10
        more = 10.4

        with self.assertRaises(Exception) as context:
            spiking_max(more, less)

        self.assertTrue(f"Please enter a positive value" in str(context.exception))

    def test_spiking_max_float_one(self):
        """ """
        less = 0.3
        more = 5.5
        result = spiking_max(less, more)
        expected_result = more
        self.assertEqual(expected_result, result)

    def test_spiking_max_float_switched_order_one(self):
        """ """
        less = 0.3
        more = 5.5
        result = spiking_max(more, less)
        expected_result = more
        self.assertEqual(expected_result, result)

    def test_spiking_max_float_zero(self):
        """ """
        less = 0
        more = 1.5
        result = spiking_max(less, more)
        expected_result = more
        self.assertEqual(expected_result, result)

    def test_spiking_max_float_switched_order_zero(self):
        """ """
        less = 0
        more = 1.5

        result = spiking_max(more, less)
        expected_result = more
        self.assertEqual(expected_result, result)

    def test_spiking_max_duplicate_zeros(self):
        """ """
        less = 0.0
        more = 0.0
        result = spiking_max(less, more)
        expected_result = more
        self.assertEqual(expected_result, result)

    def test_spiking_max_duplicate_values(self):
        """ """
        less = 5.5
        more = 5.5
        result = spiking_max(less, more)

        expected_result = more
        self.assertEqual(expected_result, result)


if __name__ == "__main__":
    unittest.main()
