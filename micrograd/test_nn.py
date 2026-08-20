import unittest
from unittest.mock import patch

from micrograd import Value
from micrograd.nn import Layer, MLP, Module, Neuron


class ModuleTest(unittest.TestCase):
    def test_default_parameters_are_empty(self):
        self.assertEqual(Module().parameters(), [])

    def test_zero_grad(self):
        class TestModule(Module):
            def __init__(self):
                self._parameters = [Value(1), Value(2)]

            def parameters(self):
                return self._parameters

        module = TestModule()
        parameters = module.parameters()
        parameters[0].grad = 3
        parameters[1].grad = -4

        module.zero_grad()

        self.assertEqual([parameter.grad for parameter in parameters], [0, 0])


class NeuronTest(unittest.TestCase):
    @patch("micrograd.nn.random.uniform", side_effect=[0.5, -0.25])
    def test_relu_neuron(self, uniform):
        neuron = Neuron(2)
        output = neuron([Value(2), Value(4)])

        self.assertEqual(output.data, 0)
        self.assertEqual(len(neuron.parameters()), 3)
        self.assertEqual(repr(neuron), "ReLUNeuron(2)")
        uniform.assert_has_calls([unittest.mock.call(-1, 1), unittest.mock.call(-1, 1)])

    @patch("micrograd.nn.random.uniform", return_value=0.5)
    def test_linear_neuron_forward_and_backward(self, uniform):
        neuron = Neuron(2, nonlin=False)
        output = neuron([Value(2), Value(4)])
        output.backward()

        self.assertEqual(output.data, 3)
        self.assertEqual([parameter.grad for parameter in neuron.parameters()], [2, 4, 1])
        self.assertEqual(repr(neuron), "LinearNeuron(2)")
        self.assertEqual(uniform.call_count, 2)
        uniform.assert_called_with(-1, 1)


class LayerTest(unittest.TestCase):
    @patch("micrograd.nn.random.uniform", return_value=0.5)
    def test_single_neuron_returns_value(self, uniform):
        layer = Layer(2, 1, nonlin=False)
        output = layer([Value(2), Value(4)])

        self.assertIsInstance(output, Value)
        self.assertEqual(output.data, 3)
        self.assertEqual(len(layer.parameters()), 3)
        self.assertEqual(repr(layer), "Layer of [LinearNeuron(2)]")

    @patch("micrograd.nn.random.uniform", return_value=0.5)
    def test_multiple_neurons_return_list(self, uniform):
        layer = Layer(2, 2, nonlin=False)
        output = layer([Value(2), Value(4)])

        self.assertIsInstance(output, list)
        self.assertEqual([value.data for value in output], [3, 3])
        self.assertEqual(len(layer.parameters()), 6)
        self.assertEqual(
            repr(layer),
            "Layer of [LinearNeuron(2), LinearNeuron(2)]",
        )


class MLPTest(unittest.TestCase):
    @patch("micrograd.nn.random.uniform", return_value=0.5)
    def test_forward_parameters_and_repr(self, uniform):
        model = MLP(2, [2, 1])
        output = model([Value(2), Value(4)])

        self.assertIsInstance(output, Value)
        self.assertEqual(output.data, 3)
        self.assertEqual(len(model.parameters()), 9)
        self.assertEqual(
            repr(model),
            "MLP of [Layer of [ReLUNeuron(2), ReLUNeuron(2)], "
            "Layer of [LinearNeuron(2)]]",
        )

    @patch("micrograd.nn.random.uniform", return_value=0.5)
    def test_zero_grad_after_backward(self, uniform):
        model = MLP(2, [1])
        output = model([Value(2), Value(4)])
        output.backward()

        self.assertTrue(any(parameter.grad != 0 for parameter in model.parameters()))

        model.zero_grad()

        self.assertTrue(all(parameter.grad == 0 for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
