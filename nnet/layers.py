from nnet.activations import *
from nnet.utils import *
from nnet.convolution import *

import numpy as np
from abc import ABC, abstractmethod


class Layer(ABC):
    """
    An abstract base class for Layers
    """

    def __init__(self):
        self.input = None
        self.output = None
        self.weight = None
        self.bias = None
        self.dW = None
        self.db = None

    def __repr__(self) -> str:
        return "Layer"

    def __str__(self) -> str:
        return self.__repr__()

    @abstractmethod
    def forward(self, input_data):
        pass

    @abstractmethod
    def backward(self):
        pass

    def __repr__(self) -> str:
        return "Layer"

    def update(self, lr):
        pass


class Input(Layer):
    """
    A class for input layer. Provides the input shape.
    """

    def __init__(self, input_shape, name="Input"):
        super().__init__()
        self.input_shape = input_shape
        self.name = name
        self.trainable = False

    def __repr__(self) -> str:
        return f"Input({self.input_shape})"

    def forward(self, input_data):
        self.input = input_data
        self.output = input_data
        return self.input

    def backward(self):
        pass

    def _output_shape(self, input_shape):
        return input_shape

    def _calc_W_b_shape(self, input_shape, output_shape):
        return None, None

    def _num_params(self, W_shape, b_shape):
        return 0


class Dense(Layer):
    """
    The dense layer
    """

    def __init__(self, neurons, activation="sigmoid", name=None, l1=0, l2=0):
        super().__init__()
        self.activation = parse_activation(activation)
        self.neurons = neurons
        self.name = name
        self.weight = None
        self.bias = None
        self.dW = None
        self.db = None
        self.l1 = l1
        self.l2 = l2

    def __repr__(self) -> str:
        return f"Dense({self.neurons})"

    def forward(self, input):
        self.input = input
        Z = np.dot(self.weight, input) + self.bias
        self.Z = Z
        A = self.activation(Z)
        self.output = A
        return A

    def backward(self, delta_l):
        l1_loss = self.l1 * np.sign(self.weight)
        l2_loss = self.l2 * self.weight
        delta_next = delta_l * self.activation.derivative(self.Z)
        dW = (
            np.dot(delta_next, self.input.T)
            + l1_loss * np.sign(self.weight)
            + l2_loss * self.weight
        )
        db = np.sum(delta_next, axis=1, keepdims=True)
        assert dW.shape == self.weight.shape
        assert db.shape == self.bias.shape
        if dW.max() > 500:
            raise ValueError("dW is Exploding", dW.max(), dW.shape)
        if db.max() > 500:
            raise ValueError("db is Exploding", db.max(), db.shape)
        self.dW = dW
        self.db = db
        delta_next = np.dot(self.weight.T, delta_next)
        return delta_next

    def update(self, lr):
        self.weight -= lr * self.dW
        self.bias -= lr * self.db

    def _output_shape(self, input_shape):
        return (self.neurons,)

    def _calc_W_b_shape(self, input_shape, output_shape):
        return (output_shape[0], input_shape[0]), (output_shape[0], 1)

    def _num_params(self, W_shape, b_shape):
        return np.prod(W_shape) + np.prod(b_shape)


class Dropout(Layer):
    """
    The dropout layer
    """

    def __init__(self, rate=0.5, name=None):
        super().__init__()
        self.rate = rate
        self.name = name
        self.mask = None

    def __repr__(self) -> str:
        return f"Dropout({self.rate})"

    def forward(self, input):
        self.input = input
        self.mask = np.random.rand(*input.shape) < (1 - self.rate)
        self.output = self.input * self.mask
        return self.output

    def backward(self, delta_l):
        delta_next = delta_l * self.mask
        return delta_next

    def _output_shape(self, input_shape):
        return input_shape

    def _calc_W_b_shape(self, input_shape, output_shape):
        return None, None

    def _num_params(self, W_shape, b_shape):
        return 0


class Flatten(Layer):
    """
    The flatten layer
    """

    def __init__(self, name=None):
        super().__init__()
        self.name = name

    def __repr__(self) -> str:
        return f"Flatten()"

    def forward(self, input):
        self.input = input
        m = input.shape[-1]
        self.output = input.reshape((np.prod(input.shape[:-1]), m))
        return self.output

    def backward(self, delta_l):
        delta_next = delta_l.reshape(self.input.shape)
        return delta_next

    def _output_shape(self, input_shape):
        return (np.prod(input_shape),)

    def _calc_W_b_shape(self, input_shape, output_shape):
        return None, None

    def _num_params(self, W_shape, b_shape):
        return 0


class Reshape(Layer):
    """
    The reshape layer
    """

    def __init__(self, output_shape, name=None):
        super().__init__()
        self.output_shape = output_shape
        self.name = name

    def __repr__(self) -> str:
        return f"Reshape({self.output_shape})"

    def forward(self, input):
        self.input = input
        m = input.shape[-1]
        output = np.zeros(
            (self.output_shape[0], self.output_shape[1], self.output_shape[2], m)
        )
        for i in range(m):
            output[:, :, :, i] = input[:, i].reshape(self.output_shape)
        self.output = output
        return self.output

    def backward(self, delta_l):
        delta_next = delta_l.reshape(self.input.shape)
        return delta_next

    def _output_shape(self, input_shape):
        return self.output_shape

    def _calc_W_b_shape(self, input_shape, output_shape):
        return None, None

    def _num_params(self, W_shape, b_shape):
        return 0


class Conv2D(Layer):
    """
    The Convolutional layer
    """

    def __init__(
        self,
        filters,
        kernel_size,
        stride=1,
        padding="same",
        activation="relu",
        name=None,
    ):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_height, self.kernel_width = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = parse_activation(activation)
        self.name = name
        self.weight = None
        self.bias = None
        self.dW = None
        self.db = None
        self.conv = Convolution()

    def __repr__(self) -> str:
        return (
            f"Conv2D({self.filters}, {self.kernel_size}, {self.stride}, {self.padding})"
        )

    def forward(self, input):
        self.input = input
        self.Z = self.conv.convolve(
            input=input,
            kernels=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )
        self.output = self.activation(self.Z)
        return self.output

    def backward(self, delta_l):
        delta_next = self.conv.convolve_backward(
            delta_l,
            self.weight,
            self.stride,
            self.padding,
        )
        delta_next = delta_next * self.activation.derivative(self.output)
        self.dW = self.conv.convolve_filter_backward(
            delta_l,
            self.input,
            self.stride,
            self.padding,
        )
        self.db = np.sum(delta_l, axis=(0, 2, 3), keepdims=True)
        return delta_next

    def update(self, lr):
        self.weight -= lr * self.dW
        self.bias -= lr * self.db

    def _output_shape(self, input_shape):
        size = self.conv._output_shape(
            input_shape,
            self.kernel_size,
            self.stride,
            self.padding,
            self.filters,
        )
        return size

    def _calc_W_b_shape(self, input_shape, output_shape):
        hi, wi, ci = input_shape
        ho, wo, co = output_shape
        W_shape = (self.kernel_height, self.kernel_width, ci, co)
        b_shape = (co, 1)
        return W_shape, b_shape

    def _num_params(self, W_shape, b_shape):
        return np.prod(W_shape) + np.prod(b_shape)


class MaxPool2D(Layer):
    """
    The Max Pooling layer
    """

    def __init__(
        self,
        kernel_size=2,
        stride=2,
        name=None,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.kernel_height, self.kernel_width = kernel_size
        self.stride = stride
        self.name = name
        self.pool = Convolution()

    def __repr__(self) -> str:
        return f"MaxPool2D({self.kernel_size}, {self.stride})"

    def forward(self, input):
        self.input = input
        self.output = self.pool.max_pool(
            input=input, kernel_size=self.kernel_size, stride=self.stride
        )
        return self.output

    def backward(self, delta_l):
        delta_next = self.pool.max_pool_backward(
            delta_l,
            self.input,
            self.kernel_size,
            self.stride,
        )
        return delta_next

    def _output_shape(self, input_shape):
        hi, wi, ci = input_shape
        ho = (hi - self.kernel_height) // self.stride + 1
        wo = (wi - self.kernel_width) // self.stride + 1
        return (
            ho,
            wo,
            ci,
        )

    def _calc_W_b_shape(self, input_shape, output_shape):
        return None, None

    def _num_params(self, W_shape, b_shape):
        return 0


class AveragePool2D(Layer):
    """
    The Average Pooling layer
    """

    def __init__(
        self,
        kernel_size=2,
        stride=2,
        name=None,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.name = name
        self.pool = Convolution()

    def __repr__(self) -> str:
        return f"AveragePool2D({self.kernel_size}, {self.stride})"

    def forward(self, input):
        self.input = input
        self.output = self.pool.average_pool(
            input=input, kernel_size=self.kernel_size, stride=self.stride
        )
        return self.output

    def backward(self, delta_l):
        delta_next = self.pool.average_pool_backward(
            delta_l, self.input, self.kernel_size, self.stride
        )
        return delta_next

    def _output_shape(self, input_shape):
        hi, wi, ci = input_shape
        ho = (hi - self.kernel_size) // self.stride + 1
        wo = (wi - self.kernel_size) // self.stride + 1
        return (
            ho,
            wo,
            ci,
        )

    def _calc_W_b_shape(self, input_shape, output_shape):
        return None, None

    def _num_params(self, W_shape, b_shape):
        return 0
