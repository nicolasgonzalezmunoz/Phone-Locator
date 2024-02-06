"""Implement classes that contain custom model architectures."""
import numpy as np
import torch
from modules.utilities import (
    get_layer_output_size,
    get_max_conv_depth,
    get_block_output_size
)


class ResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.base1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding='same'
            ),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU(True) 
        )
        self.base2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding='same'),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(True)
        )

    def forward(self, x):
        x = self.base1(x) + x
        x = self.base2(x)
        return x


class ConvolutionalLocator(torch.nn.Module):
    """Implement a convolutional network with dense layers on top of that."""
    def __init__(
        self,
        c_in: int,
        h_in: int,
        w_in: int,
        conv_depth: int,
        dense_depth: int,
        c_out_factor: int = 2,
        conv_init_kernel_size: int = 5,
        conv_kernel_size: int = 3,
        conv_stride: int = 1,
        conv_padding: int = 0,
        conv_dilation: int = 1,
        pool_kernel_size: int = 3,
        pool_stride: int = 1,
        pool_padding:int = 0,
        pool_dilation: int = 1,
        p: float = 0.3,
        activation_class: torch.nn.Module = torch.nn.ReLU
    ) -> None:
        """
        Parameters
        ----------
        c_in, h_in, w_in: int
            Amount of channels, height and width of each image, respectively.
        conv_depth:
            Number of convolutional stacks. Each stack is composed of a
            MaxPool2d layer, followed by a Conv2d, the activation layer,
            a BatchNorm2d layer and a Dropout2d. There's also an initial layer
            composed by a BatchNorm2d and a Conv2d layer, followed by an
            activation function and a dropout.
        dense_depth: int
            The depth of the dense stack on top of the convolutional layers.
            The output size of each dense layer decaes exponentially.
        c_out_factor: int
            This factor is used to multiply the number of channels on the
            input, so that the output of each Conv2d layer has
            c_in * c_out_factor channels.
        conv_init_kernel_size: int
            The size of the kernel of the first Conv2d layer.
        conv_kernel_size, conv_stride, conv_padding, conv_dilation: int
            Arguments to pass to the Conv2d layers.
        pool_kernel_size, pool_stride, pool_padding, pool_dilation: int
            Arguments to pass to the MaxPool2d layers.
        p: float
            Dropout2d probability.
        activation_class: nn.Module
            activation function to use, only the class without initialization.

        Return
        ------
        None
        """
        super().__init__()

        max_conv_depth = get_max_conv_depth(
            h_in,
            w_in,
            conv_init_kernel_size,
            conv_kernel_size,
            conv_stride,
            conv_padding,
            conv_dilation,
            pool_kernel_size,
            pool_stride,
            pool_padding,
            pool_dilation
        )
        conv_depth = min(conv_depth, max_conv_depth)

        # Build first convolutional stack
        self.convolutional_stack = torch.nn.Sequential()

        # Set output of the first Conv2d layer
        c_out = c_in * c_out_factor
        h_out, w_out = get_layer_output_size(
            h_in,
            w_in,
            conv_init_kernel_size,
            conv_stride,
            conv_padding,
            conv_dilation
        )

        self.convolutional_stack.append(
            torch.nn.Conv2d(
                c_in,
                c_out,
                kernel_size=conv_init_kernel_size,
                stride=conv_stride,
                padding=conv_padding,
                dilation=conv_dilation
            )
        )

        # Update c_in and c_out for the next layers
        c_in = c_out
        c_out = c_in * c_out_factor

        self.convolutional_stack.append(activation_class())
        self.convolutional_stack.append(torch.nn.BatchNorm2d(c_in))
        self.convolutional_stack.append(torch.nn.Dropout2d(p))

        for i in range(conv_depth - 1):
            self.convolutional_stack.append(
                torch.nn.MaxPool2d(
                    pool_kernel_size,
                    stride=pool_stride,
                    padding=pool_padding,
                    dilation=pool_dilation
                )
            )

            # Update size of the previous layer's output
            h_out, w_out = get_layer_output_size(
                h_out,
                w_out,
                pool_kernel_size,
                pool_stride,
                pool_padding,
                pool_dilation
            )

            self.convolutional_stack.append(
                torch.nn.Conv2d(
                    c_in,
                    c_out,
                    kernel_size=conv_kernel_size,
                    stride=conv_stride,
                    padding=conv_padding,
                    dilation=conv_dilation
                )
            )
            self.convolutional_stack.append(activation_class())

            # Update c_in and c_out for the next layers
            c_in = c_out
            c_out = c_in * c_out_factor

            # Update size of the previous layer's output
            h_out, w_out = get_layer_output_size(
                h_out,
                w_out,
                conv_kernel_size,
                conv_stride,
                conv_padding,
                conv_dilation
            )

            self.convolutional_stack.append(torch.nn.BatchNorm2d(c_in))
            self.convolutional_stack.append(torch.nn.Dropout2d(p))

        # Output size of dense layers decaes exponentially
        n_inputs = c_in * h_out * w_out
        n_outputs = 2
        factor = np.log(n_inputs) / np.log(n_outputs)
        sizes = 2 ** np.linspace(1.0, factor, dense_depth + 1)
        sizes = sizes[::-1]
        sizes[0] = n_inputs
        sizes[-1] = n_outputs

        self.dense_stack = torch.nn.Sequential(torch.nn.Flatten())

        # Build dense stack
        for i in range(dense_depth - 1):
            n_inputs = int(sizes[i])
            n_outputs = int(sizes[i + 1])
            self.dense_stack.append(torch.nn.Linear(n_inputs, n_outputs))
            self.dense_stack.append(activation_class())

        # Add final layer (without activation)
        self.dense_stack.append(
            torch.nn.Linear(int(sizes[-2]), int(sizes[-1]))
        )

    def forward(self, x: torch.Tensor):
        """
        Pass the input x through the model's layers and get a prediction.

        Parameters
        ----------
        x: torch.Tensor of shape (n_batches, c_in, h_in, w_in)
            Model's input

        Return
        ------
        locations: torch.Tensor of shape (n_batches, 2)
            The predicted locations of each sample on x.
        """
        x = self.convolutional_stack(x)
        locations = self.dense_stack(x)
        return locations


class PhoneNet(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        factor: int = 2,
        n_res_layers: int = 4,
        n_dense_layers: int = 1,
        p: float = 0.5
    ):
        super().__init__()
        h_out = 326
        w_out = 490
        self.res_stack = torch.nn.Sequential()
        for i in range(n_res_layers):
            out_channels = in_channels * factor
            self.res_stack.append(ResBlock(in_channels, out_channels))
            self.res_stack.append(torch.nn.MaxPool2d(2))
            in_channels = out_channels
            h_out, w_out = get_block_output_size(h_out, w_out)

        self.flatten = torch.nn.Flatten()
        self.dense_stack = torch.nn.Sequential()

        n_inputs = h_out * w_out * in_channels
        n_outputs = 2
        dense_factor = np.log(n_inputs) / np.log(n_outputs)
        sizes = 2 ** np.linspace(1.0, dense_factor, n_dense_layers + 1)
        sizes = sizes[::-1]
        sizes[0] = n_inputs
        sizes[-1] = n_outputs

        for i in range(n_dense_layers - 1):
            n_inputs = int(sizes[i])
            n_outputs = int(sizes[i + 1])
            self.dense_stack.append(torch.nn.Linear(n_inputs, n_outputs))
            self.dense_stack.append(torch.nn.ReLU(True))
            self.dense_stack.append(torch.nn.Dropout(p))

        self.dense_stack.append(
            torch.nn.Linear(int(sizes[-2]), int(sizes[-1]))
        )
  
    def forward(self, x: torch.Tensor):
        x = self.res_stack(x)
        x = self.flatten(x)
        return self.dense_stack(x)
