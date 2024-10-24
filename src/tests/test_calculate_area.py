from ..calculate_area import calculate_area
from unittest import TestCase


class calculate_area_tests(TestCase):

    def test_width_and_height_type(self):

        with self.assertRaises(Exception) as exception:
            calculate_area("1", 0)
        
        self.assertEqual(
            str(exception.exception),
            str(Exception("Invalid width"))
        )

        with self.assertRaises(Exception) as exception:
            calculate_area(None, 0)
        
        self.assertEqual(
            str(exception.exception),
            str(Exception("Invalid width"))
        )

        with self.assertRaises(Exception) as exception:
            calculate_area(1, "1")
        
        self.assertEqual(
            str(exception.exception),
            str(Exception("Invalid height"))
        )

        with self.assertRaises(Exception) as exception:
            calculate_area(1, None)
        
        self.assertEqual(
            str(exception.exception),
            str(Exception("Invalid height"))
        )
    

    def test_returns_area(self):

        self.assertEqual(calculate_area(2, 2), 4)

        