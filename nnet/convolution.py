import numpy as np
from scipy.signal import correlate as corr


class Convolution:
    def __init__(self) -> None:
        pass

    def __checks(self, input, kernels):
        if len(input.shape) != 3:
            raise ValueError(
                f"The input must be of shape (H, W, C) where H is the height, W is the width and C is the number of channels: {input.shape}"
            )
        if len(kernels.shape) != 4:
            raise ValueError(
                f"The kernel must be of shape (H, W, C, K) where H is the height, W is the width, C is the number of channels and K is the number of kernels: {kernels.shape}"
            )
        if input.shape[-1] != kernels.shape[-2]:
            raise ValueError(
                f"The number of channels in the image and kernel must be the same: {input.shape[-1]} != {kernels.shape[-2]}"
            )
        return None

    def convolve(self, input, kernels, bias, stride=1, padding="valid"):
        self.__checks(input, kernels)
        hi, wi, ci = input.shape
        hk, wk, kci, kco = kernels.shape

        if padding == "same":
            padding = int((hk - 1) / 2)
        elif padding == "valid":
            padding = 0
        elif padding == "full":
            padding = int(hk - 1)
        else:
            raise ValueError(f"Invalid padding: {padding}")
        ho = int((hi - hk + 2 * padding)) + 1
        wo = int((wi - wk + 2 * padding)) + 1
        image = np.pad(
            input, ((padding, padding), (padding, padding), (0, 0)), "constant"
        )

        output = np.zeros((ho, wo, kco))
        for i in range(kco):
            feature_map = (
                self._convolve2d_fast(image, kernels[:, :, :, i], (ho, wo)) + bias[i]
            )
            output[:, :, i] = feature_map

        # account for strides
        if stride > 1:
            output = output[::stride, ::stride, :]
        return output

    def _convolve_slow(self, image, kernel, stride=1, padding="same"):
        """
        Calculate the convolution of an image with a kernel. Works with any number of channels.

        The function is slow because it uses a for loop to iterate over the output volume.

        Parameters
        ----------
        image : numpy.ndarray
            The image to convolve shape (H, W, C)
        kernel : numpy.ndarray
            The kernel to convolve with shape (H, W, C)
        stride : int
            The stride of the convolution, default is 1
        padding : str or int
            The padding of the convolution, default is 'same'
            - If 'same' the padding is calculated to keep the output size the same as the input size
            - If 'valid' no padding is applied
            - If 'full' the padding is calculated to keep the output size the same as the input size + the kernel size - 1
            - If int, a padding of that size is applied

        Returns
        -------
        numpy.ndarray
            The convolved image, shape (H, W)
        """
        # Get shapes of image and kernel
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
        if len(kernel.shape) == 2:
            kernel = kernel[:, :, np.newaxis]
        (iH, iW, iC) = image.shape
        (kH, kW, kC) = kernel.shape

        # Check if the kernel and image have the same number of channels
        if iC != kC:
            raise ValueError(
                "The number of channels in the image and kernel must be the same"
            )

        if padding == "same":
            padding = int((kW - 1) / 2)
        elif padding == "valid":
            padding = 0
        elif padding == "full":
            padding = int(kW - 1)

        # Compute the size of the output volume
        oH = int((iH - kH + 2 * padding) / stride) + 1
        oW = int((iW - kW + 2 * padding) / stride) + 1

        # Initialize the output volume
        output = np.zeros((oH, oW))

        # Pad the image
        image = np.pad(
            image, ((padding, padding), (padding, padding), (0, 0)), "constant"
        )

        # Loop over the output volume
        for y in np.arange(0, oH):
            for x in np.arange(0, oW):
                roi = image[y * stride : y * stride + kH, x * stride : x * stride + kW]

                k = np.sum(roi * kernel)
                output[y, x] = k

        # Return the output volume
        return output

    def _convolve2d_fast(self, image, kernel, output_shape=None):
        """
        Calculate the convolution of an image with a kernel. Works with any number of channels.

        The function uses `scipy.signal.correlate` to calculate the convolution and hence is faster.

        Parameters
        ----------
        image : numpy.ndarray
            The image to convolve shape (H, W, C)
        kernel : numpy.ndarray
            The kernel to convolve with shape (H, W, C)
        padding : str or int
            The padding of the convolution, default is 'same'
            - If 'same' the padding is calculated to keep the output size the same as the input size
            - If 'valid' no padding is applied
            - If 'full' the padding is calculated to keep the output size the same as the input size + the kernel size - 1
            - If int, a padding of that size is applied

        Returns
        -------
        numpy.ndarray
            The convolved image, shape (H, W)
        """
        ho, wo = output_shape
        output = np.zeros((ho, wo))
        ci = image.shape[-1]

        # Loop over the color channel and use scipy correlate
        for c in range(ci):
            output += corr(image[:, :, c], kernel[:, :, c], mode="valid")

        # Return the output volume
        return output

    def max_pool(self, image, pool_size, stride=1):
        """
        Calculate the maxpool of an image.

        The function uses `scipy.signal.correlate` to calculate the convolution and hence is faster.

        Parameters
        ----------
        image : numpy.ndarray
            The image to convolve shape (H, W, C)
        pool_size : int
            The size of the pooling window
        stride : int
            The stride of the pooling, default is 1

        Returns
        -------
        numpy.ndarray
            The convolved image, shape (H, W)
        """
        hi, wi, _ = image.shape
        ho = int((hi - pool_size) / stride) + 1
        wo = int((wi - pool_size) / stride) + 1
        output = np.zeros((ho, wo))

        for x in range(wo):
            for y in range(ho):
                roi = image[
                    y * stride : y * stride + pool_size,
                    x * stride : x * stride + pool_size,
                ]
                max_id = np.argmax(roi)
                output[y, x] = np.max(roi)
        return output

    def average_pool(self, image, pool_size, stride=1):
        """
        Calculate the averagepool of an image.

        The function uses `scipy.signal.correlate` to calculate the convolution and hence is faster.

        Parameters
        ----------
        image : numpy.ndarray
            The image to convolve shape (H, W, C)
        pool_size : int
            The size of the pooling window
        stride : int
            The stride of the pooling, default is 1

        Returns
        -------
        numpy.ndarray
            The convolved image, shape (H, W)
        """
        hi, wi, _ = image.shape
        ho = int((hi - pool_size) / stride) + 1
        wo = int((wi - pool_size) / stride) + 1
        output = np.zeros((ho, wo))

        for x in range(wo):
            for y in range(ho):
                roi = image[
                    y * stride : y * stride + pool_size,
                    x * stride : x * stride + pool_size,
                ]
                output[y, x] = np.mean(roi)
        return output

    def convolve_backward(self, input, kernels, output_grad, stride=1, padding="same"):
        # TODO Implement the backward pass for the convolutional layer
        pass

    def convolve_filter_backward(
        self, input, kernels, output_grad, stride=1, padding="same"
    ):
        # TODO Implement the backward pass for the convolutional layer
        pass

    def max_pool_backward(self, input, pool_size, output_grad, stride=1):
        """
        Calculate the gradient of the output with respect to the input.

        Parameters
        ----------
        input : numpy.ndarray
            The input to the maxpool layer, shape (H, W, C)
        pool_size : int
            The size of the pooling window
        output_grad : numpy.ndarray
            The gradient of the loss with respect to the output of the maxpool layer, shape (H, W, C)
        stride : int
            The stride of the pooling, default is 1

        Returns
        -------
        numpy.ndarray
            The gradient of the loss with respect to the input, shape (H, W, C)
        """
        # TODO: Implement the maxpool backward pass
        pass

    def average_pool_backward(self, input, pool_size, output_grad, stride=1):
        """
        Calculate the gradient of the output with respect to the input.

        Parameters
        ----------
        input : numpy.ndarray
            The input to the averagepool layer, shape (H, W, C)
        pool_size : int
            The size of the pooling window
        output_grad : numpy.ndarray
            The gradient of the loss with respect to the output of the averagepool layer, shape (H, W, C)
        stride : int
            The stride of the pooling, default is 1

        Returns
        -------
        numpy.ndarray
            The gradient of the loss with respect to the input, shape (H, W, C)
        """
        # TODO Implement the averagepool backward
        pass
