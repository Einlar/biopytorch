import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union
import math
import logging

logger = logging.getLogger("bioconv2d")

_size_2_t = Union[int, Tuple[int, int]]  # from nn.common_types


def _size_2_t_to_tuple(arg: _size_2_t) -> Tuple[int, int]:
    """Converts an int `arg` to a 2-tuple of equal entries `(arg, arg)`. If `arg` is already a 2-tuple, nothing is done.

    Examples
    --------
    1      -> (1, 1)
    (3,)   -> (3, 3)
    (1, 2) -> (1, 2)
    """

    try:
        arg_tuple = tuple(arg)
    except TypeError:
        arg_tuple = tuple([arg])

    if len(arg_tuple) != 2:
        return (int(arg_tuple[0]),) * 2

    return (int(arg[0]), int(arg[1]))


def same_padding(
    kernel_size: Tuple[int, int], dilation: Tuple[int, int] = (1, 1)
) -> Union[Tuple[int, int], Tuple[int, int, int, int]]:
    """Compute the amount of padding needed to keep the output of a Conv2D layer of the same size as the input, given the `kernel_size` of the filters and the amount of `dilation`. Stride is assumed to be 1, as no other values are supported by PyTorch (see F.conv2d docs).
    Note: if kernel_size is even along an axis, and dilation is odd along that same axis, the required padding will be asymmetric. In that case, additional padding is added to the right/bottom.

    Returns
    -------
    (pad_left, pad_right, pad_top, pad_bottom) if pad_left != pad_right OR pad_top != pad_bottom
    otherwise
    (pad_height, pad_width) if pad_top == pad_bottom AND pad_left == pad_right
    """

    pad_height = dilation[0] * (kernel_size[0] - 1) / 2
    pad_width = dilation[1] * (kernel_size[1] - 1) / 2

    if pad_height.is_integer() and pad_width.is_integer():
        return (int(pad_height), int(pad_width))

    pad_y = (math.floor(pad_height), math.ceil(pad_height))
    pad_x = (math.floor(pad_width), math.ceil(pad_width))

    return (*pad_x, *pad_y)


def split_padding(
    padding: Union[str, _size_2_t],
    kernel_size: _size_2_t,
    dilation: _size_2_t,
    padding_mode: str = "zeros",
) -> Tuple[Tuple[4 * (int,)], Tuple[int, int], str]:
    """
    Apply to a tensor `x` any padding that cannot be applied directly by F.conv2d (e.g. asymmetric padding, or when padding_mode is not 'zeros').

    Returns
    -------
    added_padding : Tuple[int, int, int, int]
        Padding that must be applied with F.pad
    remaining_padding : Tuple[int, int]
        Padding that can be applied by F.conv2d.
    padding_mode : str
        Padding mode adjusted for use in F.pad
    """

    symmetric = True

    if isinstance(padding, str):
        if padding == "valid":
            padding = 0
        elif padding == "same":
            padding = same_padding(kernel_size, dilation)

            if len(padding) == 4:
                symmetric = False  # Asymmetric padding is needed
        else:
            raise NotImplementedError(f'"{padding}" is not implemented.')

    if symmetric:
        padding = _size_2_t_to_tuple(padding)

        if padding_mode != "zeros":  # F.pad must be used
            pad_height, pad_width = padding
            return (pad_width, pad_width, pad_height, pad_height), (0, 0), padding_mode

        return (0, 0, 0, 0), padding, padding_mode

    else:  # Asymmetric must be manual
        if padding_mode == "zeros":
            padding_mode = "constant"  # because F.pad uses a different convention...
        return padding, (0, 0), padding_mode


class BioConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        lebesgue_p: int = 3,
        ranking_param: int = 2,
        delta: float = 0.05,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        """
        2d Convolutional layer that can be trained using the bio-inspired Hebbian learning rule
        from "Unsupervised learning by competing hidden units", D. Krotov, J. Hopfield, 2019
        (https://www.pnas.org/content/116/16/7723).
        Most of the usual parameters from nn.Conv2d are supported.
        Call the `training_step` method to make a single training step using a given input batch.

        Parameters
        ----------
        in_channels : int
            Number of input channels. Input data should be of shape (batch_size, in_channels, height, width).
        out_channels : int
            Number of kernels to be learned.
        kernel_size : _size_2_t
            Shape of a kernel. It can be a tuple (kernel_height, kernel_width),
            or a single integer if both dimensions are equal.
        lebesgue_p : int, optional
            Lebesgue p-norm to be used for learning weights.
            A higher value leads to "sparser" weights, i.e. fewer values in a kernel
            that are significantly different from 0.
        ranking_param : int, optional
            Maximum number of kernels encoding a "similar" pattern.
            Specifically, for each sample, kernels are ranked by their "current" (i.e. activation).
            The kernel in the k-th place (where k is the ranking_param) will be pushed *away*
            from that sample (anti-Hebbian learning).
        delta : float, optional
            Strength of anti-Hebbian learning, i.e. how much kernels that fall in the
            k-th place (k = ranking_param) for a sample are pushed *away* from it.
            A too high value can impair convergence.

        The following parameters are the same from nn.Conv2d:
        stride : _size_2_t, optional
        padding : Union[str, _size_2_t], optional
        dilation : _size_2_t, optional
        groups : int, optional
        bias : bool, optional
        padding_mode : str, optional
        device : [type], optional
        dtype : [type], optional
        """

        super().__init__()

        if bias == True:
            raise NotImplementedError("Only bias=False is supported for now.")
        # TODO The only not implemented thing is the bias. This can be solved by adding a batchnorm (even without learnable parameters, i.e. with affine=False) before this layer. In fact, I'm not sure how the Krotov learning rule can be generalized to biases, since the original paper does not use them. Perhaps simply adding ones in the input? Then how to do this in Conv2d layers? I'll leave this problem for a future version...

        assert (
            out_channels // groups > 1
        ), f"There must be more than one kernel per group to have competition. The number of output channels ({out_channels}) must be a higher multiple of the number of groups ({groups})."
        assert (
            in_channels % groups == 0
        ), f"The input has {in_channels} which is not divisible by {groups}"
        assert (
            out_channels % groups == 0
        ), f"The output channels are {out_channels}, which is not divisible by {groups}"

        # nn.Conv2d parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.kernel_size = _size_2_t_to_tuple(kernel_size)
        self.stride = _size_2_t_to_tuple(stride)
        self.padding = padding

        if (self.stride != (1, 1)) and (padding == "same"):
            raise NotImplementedError(
                "padding = 'same' is not implemented for strided kernels."
            )

        self.dilation = _size_2_t_to_tuple(dilation)
        self.F_padding, self.conv2d_padding, self.padding_mode = split_padding(
            padding, self.kernel_size, self.dilation, padding_mode
        )

        logger.debug(
            f"F_padding: {self.F_padding}, conv2d_padding: {self.conv2d_padding}"
        )

        self.F_pad = False  # If True, F.pad must be used to add custom padding (adding significant overhead). Otherwise F.conv2d suffices (faster).
        if sum(self.F_padding) > 0:
            self.F_pad = True

        self.weight = nn.Parameter(
            torch.randn(
                out_channels,
                in_channels // groups,
                *self.kernel_size,
                device=device,
                dtype=dtype,
            ),
            requires_grad=False,
        )

        # "BioLearn" parameters
        self.lebesgue_p = lebesgue_p
        self.ranking_param = ranking_param
        self.delta = delta

        self.batch_norm = nn.BatchNorm2d(
            self.in_channels, affine=False, device=device, dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the output of the Convolutional layer.

        Parameters
        ----------
        x : torch.Tensor
            Input batch tensor, of shape (batch_size, in_channels, height, width)

        Returns
        -------
        torch.Tensor
        """

        self.batch_norm.eval()

        x_normalized = self.batch_norm(x)
        if self.F_pad:
            x_normalized = F.pad(x_normalized, self.F_padding, self.padding_mode)

        return F.conv2d(
            x_normalized,
            torch.sign(self.weight) * torch.abs(self.weight) ** (self.lebesgue_p - 1),
            bias=None,
            stride=self.stride,
            padding=self.conv2d_padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def delta_weights(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Proposed update for `weights` given a batch of inputs `x`, according
        to the bio-inspired hebbian learning rule. All parameters for the convolution
        (e.g. stride, padding, dilation...) are the ones passed to the initializer.

        Parameters
        ----------
        x : torch.Tensor
        weight : torch.Tensor

        Returns
        -------
        torch.Tensor
        """

        if self.F_pad:
            x = F.pad(x, self.F_padding, self.padding_mode)

        currents = F.conv2d(
            x,
            torch.sign(weight) * torch.abs(weight) ** (self.lebesgue_p - 1),
            stride=self.stride,
            padding=self.conv2d_padding,
            dilation=self.dilation,
            groups=1,
        )

        # Apply each kernel (total: channel_out) to each "patch" in the image
        batch_size, out_channels, height_out, width_out = currents.shape
        currents = currents.transpose(0, 1).reshape(out_channels, -1)
        # Shape is now (out_channels, n_all_patches_in_minibatch)

        _, ranking_indices = currents.topk(self.ranking_param, dim=0)
        # Shape is (ranking_param, n_all_patches_in_minibatch)

        post_activation_currents = torch.zeros_like(currents)
        # Shape is (out_channels, n_all_patches_in_minibatch)

        batch_indices = torch.arange(currents.size(-1), device=currents.device)
        post_activation_currents[
            ranking_indices[0], batch_indices
        ] = 1.0  # The kernels that are most active for a patch are updated towards it
        post_activation_currents[
            ranking_indices[self.ranking_param - 1], batch_indices
        ] = (
            -self.delta
        )  # Others that are less activated are pushed in the opposite direction (i.e. they will be "less active" when encountering a similar patch in the future)

        activation_filter = post_activation_currents.view(
            out_channels, batch_size, height_out, width_out
        )

        stride_mask = torch.zeros(self.stride, device=x.device, dtype=x.dtype)
        stride_mask[0][0] = 1
        activation_filter = torch.kron(
            activation_filter, stride_mask
        )  # construct a block matrix

        filter_dims = (
            x.size(-2) + 2 * self.conv2d_padding[0] + 1 - self.kernel_size[0],
            x.size(-1) + 2 * self.conv2d_padding[1] + 1 - self.kernel_size[1],
        )

        activation_filter = activation_filter[..., : filter_dims[0], : filter_dims[1]]

        delta_weights = F.conv2d(
            x.transpose(0, 1),
            activation_filter,
            padding=self.conv2d_padding,
            stride=self.dilation,
        ).swapaxes(
            0, 1
        )  # setting stride=dilation gives for free support for the dilation argument, in the sense that it gives equal results with the unfold-version code. However, this should be checked a bit better, as I don't have a full theoretical understanding on why this works in practice.
        # Shape is (out_channels, in_channels, kernel_height, kernel_width)

        second_term = torch.sum(torch.mul(post_activation_currents, currents), dim=1)
        # Shape is (out_channels,)
        delta_weights.sub_(second_term.view(-1, 1, 1, 1) * weight)

        # nc = torch.amax(torch.abs(delta_weights))
        nc = torch.abs(delta_weights).amax((1, 2, 3), keepdim=True)
        delta_weights.div_(nc + 1e-5)

        return delta_weights

    def training_step(self, x: torch.Tensor, learning_rate: float = 0.1) -> float:
        """Apply the "BioLearn" rule to update the weights, according to:
        ``self.weights += learning_rate * delta_weights```

        Parameters
        ----------
        x : torch.Tensor
            Input values, a Tensor of shape (batch_size, in_channels, height, width)
        learning_rate : float, optional
            Learning rate, by default .1
        """

        self.batch_norm.train()

        x = self.batch_norm(x)

        if self.groups == 1:
            delta_weights = self.delta_weights(x, self.weight)
        else:
            x_groups = torch.chunk(x, self.groups, dim=1)
            weight_groups = torch.chunk(self.weight, self.groups, dim=0)
            delta_weights = torch.cat(
                [self.delta_weights(xi, wi) for xi, wi in zip(x_groups, weight_groups)]
            )

        self.weight.add_(learning_rate * delta_weights)
        # p-norm of weights should converge to 1
        weights = self.weight.view(self.out_channels, -1)
        norm = torch.sum(torch.abs(weights) ** self.lebesgue_p, axis=1)
        convergence = torch.max(torch.abs(norm - torch.ones_like(norm))).cpu().numpy()

        return convergence

    def __str__(self) -> str:
        """String representation for the layer."""

        return (
            "BioConv2d(\n"
            + f"in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, "
            + f"lebesgue_p={self.lebesgue_p}, ranking_param={self.ranking_param}, delta={self.delta}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}, groups={self.groups}, bias=False, padding_mode='{self.padding_mode}'\n)"
        )
