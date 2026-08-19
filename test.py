import unittest

from micrograd import Value


class ValueTest(unittest.TestCase):
    def test_arithmetic(self):
        value = Value(2) + 3
        value = value * 4

        self.assertEqual(value.data, 20)

    def test_backward(self):
        left = Value(2)
        right = Value(3)
        result = left * right + left

        result.backward()

        self.assertEqual(left.grad, 4)
        self.assertEqual(right.grad, 2)
        self.assertEqual(result.grad, 1.0)

    def test_relu(self):
        for data, expected_data, expected_grad in (
            (-2, 0, 0),
            (0, 0, 0),
            (3, 3, 1),
        ):
            with self.subTest(data=data):
                value = Value(data)
                result = value.relu()

                self.assertEqual(result.data, expected_data)

                result.backward()

                self.assertEqual(value.grad, expected_grad)


if __name__ == "__main__":
    unittest.main()
