from typing import List, Optional, Union

import torch
import torch.nn as nn

from .layers import (
    Crop,
    TimeDistributedLinear,
    TimeDistributedConv2d,
    TimeDistributedConvTranspose2d,
    TimeDistributedMaxPool2d,
    TimeDistributedBatchNorm1d,
    TimeDistributedBatchNorm2d,
    TimeDistributedUpsamplingBilinear2d
)
from .utils.checks import ValidateDimension
from .utils.enums import eDecoder


class BaseEncoderDecoder(nn.Module):

    def __init__(self,
                 nx: int,
                 nc: int,
                 layers: List[int],
                 latent_dim: Optional[int] = None,
                 activation: nn.Module = nn.Tanh(),
                 dropout: float = 0.0,
                 batch_norm: bool = False) -> None:

        """Base class for Encoder / Decoder

        Parameters
        ----------
        nx: int
            Resolution of the input field.
        nc: int
            Number of channels / velocities.
        layers: List[int]
            List of layer sizes to use in the encoder / decoder.
        latent_dim: Optional[int]
            Size of the latent space.
        activation: nn.Module
            Activation to use for the network.
        dropout: float
            Dropout rate for training.
        batch_norm: bool
            Flag to determine whether to use BatchNormalisation.
        """

        super().__init__()

        if not all(map(lambda x: (x != 0) and (x & (x - 1) == 0), layers)):
            raise ValueError('Layer values must be powers of two.')

        self.nx = nx
        self.nc = nc
        self.layers = layers
        self.latent_dim = latent_dim
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm

        # specify dimensions for each layer
        self.dims = (self.nc, *self.layers)

        # record dimensionality of space prior to Linear layers
        _nx = self.nx + (self.nx % 2 != 0)

        self.prelatent_nx = _nx // (2 ** len(self.layers))
        self.prelatent_shape = self.layers[-1] * self.prelatent_nx ** 2

        # variable to store operation layers
        self.module_layers: Union[nn.Module, None] = None


class Autoencoder(BaseEncoderDecoder):

    def __init__(self,
                 nx: int,
                 nc: int,
                 layers: List[int],
                 latent_dim: Optional[int],
                 decoder: eDecoder = eDecoder.upsampling,
                 activation: nn.Module = nn.Tanh(),
                 dropout: float = 0.0,
                 batch_norm: bool = False) -> None:

        """Autoencoder class.

        Parameters
        ----------
        nx: int
            Resolution of the input field.
        nc: int
            Number of channels / velocities.
        layers: List[int]
            List of layer sizes to use in the encoder / decoder.
        latent_dim: Optional[int]
            Size of the latent space.
        decoder: eDecoder
            Type of decoder to use, i.e: UpsamplingDecoder, TransposeDecoder.
        activation: nn.Module
            Activation to use for the network.
        dropout: float
            Dropout rate for training.
        batch_norm: bool
            Flag to determine whether to use BatchNormalisation.
        """

        super().__init__(nx, nc, layers, latent_dim, activation, dropout, batch_norm)

        if not isinstance(decoder, eDecoder):
            raise TypeError('Must provide an eDecoder instance.')

        # define encoder
        self.encoder = Encoder(nx, nc, layers, latent_dim, activation, dropout, batch_norm)

        # define decoder
        self.decoder: Union[UpsamplingDecoder, TransposeDecoder]
        if decoder == eDecoder.upsampling:
            self.decoder = UpsamplingDecoder(nx, nc, layers, latent_dim, activation, dropout, batch_norm)
        elif decoder == eDecoder.transpose:
            self.decoder = TransposeDecoder(nx, nc, layers, latent_dim, activation, dropout, batch_norm)
        else:
            raise ValueError('Invalid eDecoder instance given.')

    @ValidateDimension(ndim=5)
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """Conduct a single forward-pass through the model.

        Parameters
        ----------
        x: torch.Tensor
            Input Tensor.

        Returns
        -------
        x: torch.Tensor
            Output Tensor.
        """

        x = self.encoder(x)
        x = self.decoder(x)

        return x


class Encoder(BaseEncoderDecoder):

    def __init__(self,
                 nx: int,
                 nc: int,
                 layers: List[int],
                 latent_dim: Optional[int],
                 activation: nn.Module = nn.Tanh(),
                 dropout: float = 0.0,
                 batch_norm: bool = False) -> None:

        """Encoder class.

        Parameters
        ----------
        nx: int
            Resolution of the input field.
        nc: int
            Number of channels / velocities.
        layers: List[int]
            List of layer sizes to use in the encoder / decoder.
        latent_dim: int
            Size of the latent space.
        activation: nn.Module
            Activation to use for the network.
        dropout: float
            Dropout rate for training.
        batch_norm: bool
            Flag to determine whether to use BatchNormalisation.
        """

        super().__init__(nx, nc, layers, latent_dim, activation, dropout, batch_norm)

        # define convolutional architecture
        encoder_layers: List[nn.Module] = []

        if self.nx % 2 != 0:
            encoder_layers.append(nn.ZeroPad2d((0, 1, 0, 1)))

        for i in range(len(self.dims) - 1):

            # define convolutional layer
            conv = TimeDistributedConv2d(
                in_channels=self.dims[i],
                out_channels=self.dims[i + 1],
                kernel_size=(3, 3),
                padding='same',
                padding_mode='circular'
            )

            encoder_layers.append(conv)
            encoder_layers.append(self.activation)
            encoder_layers.append(TimeDistributedMaxPool2d(kernel_size=(2, 2)))

            if self.batch_norm:
                encoder_layers.append(TimeDistributedBatchNorm2d(self.dims[i + 1]))
            if self.dropout > 0.0:
                encoder_layers.append(nn.Dropout(self.dropout, inplace=True))

        # define linear architecture -- transformation to latent space
        linear_layers: List[nn.Module] = []
        if self.latent_dim:

            linear_layers.extend([
                nn.Flatten(start_dim=2),
                TimeDistributedLinear(self.prelatent_shape, self.latent_dim),
                self.activation,
            ])

            if self.batch_norm:
                linear_layers.append(TimeDistributedBatchNorm1d(self.latent_dim))
            if self.dropout > 0.0:
                linear_layers.append(nn.Dropout(self.dropout, inplace=True))

        # produce sequential operation for the module layers
        self.module_layers: nn.Module = nn.Sequential(*encoder_layers, *linear_layers)

    @ValidateDimension(ndim=5)
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """Conduct a single forward-pass through the model.

        Parameters
        ----------
        x: torch.Tensor
            Input Tensor.

        Returns
        -------
        torch.Tensor
            Output Tensor.
        """

        return self.module_layers(x)


class UpsamplingDecoder(BaseEncoderDecoder):

    def __init__(self,
                 nx: int,
                 nc: int,
                 layers: List[int],
                 latent_dim: Optional[int],
                 activation: nn.Module = nn.Tanh(),
                 dropout: float = 0.0,
                 batch_norm: bool = False) -> None:

        """Decoder class with Upsampling.

        Parameters
        ----------
        nx: int
            Resolution of the input field.
        nc: int
            Number of channels / velocities.
        layers: List[int]
            List of layer sizes to use in the encoder / decoder.
        latent_dim: Optional[int]
            Size of the latent space.
        activation: nn.Module
            Activation to use for the network.
        dropout: float
            Dropout rate for training.
        batch_norm: bool
            Flag to determine whether to use BatchNormalisation.
        """

        super().__init__(nx, nc, layers, latent_dim, activation, dropout, batch_norm)

        # define convolutional architecture
        decoder_layers: List[nn.Module] = []
        for idx_layer in range(len(self.dims) - 1):

            # index to count through layers in reverse
            idx = len(self.dims) - idx_layer - 1

            # upsample the input to the relevant shape
            upsample_size = (
                int(self.prelatent_nx * 2 ** (idx_layer + 1)),
                int(self.prelatent_nx * 2 ** (idx_layer + 1))
            )
            decoder_layers.append(TimeDistributedUpsamplingBilinear2d(size=upsample_size))

            # define convolutional layer
            conv = TimeDistributedConv2d(
                in_channels=self.dims[idx],
                out_channels=self.dims[idx - 1],
                kernel_size=(3, 3),
                padding='same',
                padding_mode='circular'
            )

            decoder_layers.append(conv)

            # ensure not to add unnecessary modules to output layer
            if idx_layer < len(self.layers) - 1:

                decoder_layers.append(self.activation)

                if self.batch_norm:
                    decoder_layers.append(TimeDistributedBatchNorm2d(self.dims[idx - 1]))
                if self.dropout > 0.0:
                    decoder_layers.append(nn.Dropout(self.dropout, inplace=True))

        if self.nx % 2 != 0:
            decoder_layers.append(Crop(idx_lb=0, idx_ub=self.nx, ndim=self.nc))

        # define linear architecture -- transformation from latent space
        linear_layers: List[nn.Module] = []
        if self.latent_dim:

            linear_layers.extend([
                TimeDistributedLinear(self.latent_dim, self.prelatent_shape),
                self.activation,
            ])

            if self.batch_norm:
                linear_layers.append(TimeDistributedBatchNorm1d(self.prelatent_shape))
            if self.dropout > 0.0:
                linear_layers.append(nn.Dropout(self.dropout, inplace=True))

            linear_layers.append(nn.Unflatten(dim=2, unflattened_size=(self.dims[-1], self.prelatent_nx, self.prelatent_nx)))

        # produce sequential operation for the module layers
        self.module_layers: nn.Module = nn.Sequential(*linear_layers, *decoder_layers)

    @ValidateDimension(ndim=3)
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """Conduct a single forward-pass through the model.

        Parameters
        ----------
        x: torch.Tensor
            Input Tensor.

        Returns
        -------
        torch.Tensor
            Output Tensor.
        """

        return self.module_layers(x)


class TransposeDecoder(BaseEncoderDecoder):

    def __init__(self,
                 nx: int,
                 nc: int,
                 layers: List[int],
                 latent_dim: Optional[int],
                 activation: nn.Module = nn.ReLU(inplace=True),
                 dropout: float = 0.0,
                 batch_norm: bool = False) -> None:

        """Decoder class with ConvTranspose2d.

        Parameters
        ----------
        nx: int
            Resolution of the input field.
        nc: int
            Number of channels / velocities.
        layers: List[int]
            List of layer sizes to use in the encoder / decoder.
        latent_dim: Optional[int]
            Size of the latent space.
        activation: nn.Module
            Activation to use for the network.
        dropout: float
            Dropout rate for training.
        batch_norm: bool
            Flag to determine whether to use BatchNormalisation.
        """

        super().__init__(nx, nc, layers, latent_dim, activation, dropout, batch_norm)

        # define convolutional architecture
        decoder_layers: List[nn.Module] = []
        for idx_layer in range(len(self.dims) - 1):

            # index to count through layers in reverse
            idx = len(self.dims) - idx_layer - 1

            # define transposed convolutional layer
            t_conv = TimeDistributedConvTranspose2d(
                in_channels=self.dims[idx],
                out_channels=self.dims[idx - 1],
                kernel_size=(3, 3),
                stride=2,
                padding=1,
                output_padding=1,
                padding_mode='circular'
            )

            decoder_layers.append(t_conv)

            # ensure not to add unnecessary modules to output layer
            if idx_layer < len(self.layers) - 1:

                decoder_layers.append(self.activation)

                if self.batch_norm:
                    decoder_layers.append(TimeDistributedBatchNorm2d(self.dims[idx - 1]))
                if self.dropout > 0.0:
                    decoder_layers.append(nn.Dropout(self.dropout, inplace=True))

        if self.nx % 2 != 0:
            decoder_layers.append(Crop(idx_lb=0, idx_ub=self.nx, ndim=self.nc))

        # define linear architecture -- transformation from latent space
        linear_layers: List[nn.Module] = []
        if self.latent_dim:

            linear_layers.extend([
                TimeDistributedLinear(self.latent_dim, self.prelatent_shape),
                self.activation
            ])

            if self.batch_norm:
                linear_layers.append(TimeDistributedBatchNorm1d(self.prelatent_shape))
            if self.dropout > 0.0:
                linear_layers.append(nn.Dropout(self.dropout, inplace=True))

            linear_layers.append(nn.Unflatten(dim=2, unflattened_size=(self.dims[-1], self.prelatent_nx, self.prelatent_nx)))

        # produce sequential operation for the module layers
        self.module_layers: nn.Module = nn.Sequential(*linear_layers, *decoder_layers)

    @ValidateDimension(ndim=3)
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """Conduct a single forward-pass through the model.

        Parameters
        ----------
        x: torch.Tensor
            Input Tensor.

        Returns
        -------
        torch.Tensor
            Output Tensor.
        """

        return self.module_layers(x)
