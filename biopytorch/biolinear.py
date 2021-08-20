import torch
import torch.nn as nn
import torch.nn.functional as F


class BioLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        delta: float = 0.05,
        ranking_param: int = 2,
        lebesgue_p: int = 3,
        device=None,
        dtype=None,
    ) -> None:
        """
        Analogue of a nn.Linear layer implementing the bio-inspired Hebbian learning rule from [1].
        Automatically flattens the input, and applies batch normalization (without any learnable parameters,
        so that it does not require backprop). Use the method `training_step` to perform a single weights update.

        [1]: "Unsupervised learning by competing hidden units", D. Krotov, J. J. Hopfield, 2019,
            https://www.pnas.org/content/116/16/7723

        Parameters
        ----------
        in_features : int
            Input dimension of the Linear layer
        out_features : int
            Output dimension of the Linear layer
        bias : bool

        delta : float
            Strength of anti-Hebbian learning (from eq. 9 in [1]).
        lebesgue_p : float
            Parameter for Lebesgue measure, used for defining an inner product (from eq. 2 in [1]).
        ranking_param: int
            Rank of the current to which anti-hebbian learning is applied. Should be >= 2. This is the `k` from eq. 10 in [1].
        device
        dtype
        """

        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.delta = delta
        self.ranking_param = ranking_param
        self.lebesgue_p = lebesgue_p

        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, device=device, dtype=dtype),
            requires_grad=False,
        )

        self.bias = None
        if bias:
            self.bias = nn.Parameter(
                torch.randn(
                    out_features,
                ),
                requires_grad=False,
            )

        self.batch_norm = nn.BatchNorm1d(
            self.in_features, affine=False, device=device, dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute output of the layer (forward pass).

        Parameters
        ----------
        x : torch.Tensor
            Input. Expected to be of shape (batch_size, ...), where ... denotes an arbitrary
            sequence of dimensions, with product equal to in_features.
        """

        self.batch_norm.eval()

        x = x.view(x.size(0), -1)  # Auto flatten
        return F.linear(
            self.batch_norm(x),
            torch.sign(self.weight) * torch.abs(self.weight) ** (self.lebesgue_p - 1),
            self.bias,
        )

    def delta_weights(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Compute the change of `weights` given by the Krotov learning rule (eq. 3 from [1], with R=1)
        for a given batch of `x`. See "A fast implementation" section in [1].

        The formula is:
        ```delta_weights = g(currents) @ x - normalization_mtx (*) weights
        currents = (sgn(weights) (*) abs(weights) ** lebesgue_p) @ x.T```

        where `normalization_mtx` is a matrix of the same shape of `weights`, with all columns equal to:
        ```\sum_{batches} [g(currents) (*) currents]```

        The symbol `@` denotes matrix multiplication, while `(*)` is element-wise multiplication (Hadamard product).
        Finally, the function `g` (eq. 10 in [1]) returns:
        ```g(currents[i,j]) = 1      if currents[i,j] is the highest in the j-th column (sample),
                              -Delta if currets[i,j] is the k-th highest value in the j-th column (sample),
                               0     otherwise```


        Parameters
        ----------
        x : torch.Tensor
            Input. Expected to be of shape (batch_size, in_features).
        weights : torch.Tensor of shape (output_size, input_size)
            Model's weights

        Returns
        -------
        delta_weights : torch.Tensor of shape (output_size, input_size)
            Change of weights given by the fast implementation of Krotov learning rule. The tensor is normalized
            so that its maximum is equal to 1.
        """

        batch_size = x.shape[0]

        # ---Currents---#
        x = torch.t(x)  # Shape is (batch_size, input_size) -> (input_size, batch_size)
        currents = torch.matmul(
            torch.sign(weights) * torch.abs(weights) ** (self.lebesgue_p - 1), x
        )  # Shape is (output_size, batch_size)

        # ---Activations---#
        _, ranking_indices = currents.topk(
            self.ranking_param, dim=0
        )  # Shape is (self.ranking_param, batch_size)
        # Indices of the top k currents produced by each input sample

        post_activation_currents = torch.zeros_like(
            currents
        )  # Shape is (output_size, batch_size)
        # Computes g(currents)
        # Note that all activations are 0, except the largest current (activation of 1) and the k-th largest (activation of -delta)

        batch_indices = torch.arange(batch_size, device=post_activation_currents.device)
        post_activation_currents[ranking_indices[0], batch_indices] = 1.0
        post_activation_currents[
            ranking_indices[self.ranking_param - 1], batch_indices
        ] = -self.delta

        # ---Compute change of weights---#
        delta_weights = torch.matmul(
            post_activation_currents, torch.t(x)
        )  # Overlap between post_activation_currents and inputs
        second_term = torch.sum(
            torch.mul(post_activation_currents, currents), dim=1
        )  # Overlap between currents and post_activation_currents
        # Results are summed over batches, resulting in a shape of (output_size,)

        delta_weights = delta_weights - second_term.unsqueeze(1) * weights

        # ---Normalize---#
        nc = torch.abs(delta_weights).amax()  # .amax(1, keepdim=True)
        delta_weights.div_(nc + 1e-5)

        return delta_weights  # Maximum (absolute) change of weight is set to +1.

    def training_step(self, x: torch.Tensor, learning_rate: float = 0.1) -> float:
        """Apply the "BioLearn" rule to update the weights, according to:
        ``self.weights += learning_rate * delta_weights```

        Parameters
        ----------
        x : torch.Tensor
            Input. Expected to be of shape (batch_size, ...), where ... denotes an arbitrary
            sequence of dimensions, with product equal to in_features.
        learning_rate : float, optional
            Learning rate, by default .1

        Returns
        -------
        convergence : float
            Weights should converge so that:
            ```torch.sum(torch.abs(weights) ** self.lebesgue_p, axis=1)```
            is a vector of ones.
            As a metric of convergence, the maximum absolute deviation from 1. is returned:
            ```convergence = torch.max(torch.abs(norm - torch.ones_like(norm))).cpu().numpy()```
        """

        self.batch_norm.train()

        x = x.view(x.size(0), -1)  # Auto flatten
        x = self.batch_norm(x)

        weights = self.weight

        if self.bias is not None:
            # print(x.shape, x[:,0].shape)
            x = torch.cat(
                [
                    torch.ones(
                        (x.size(0), 1), dtype=self.bias.dtype, device=self.bias.device
                    ),
                    x,
                ],
                dim=1,
            )
            weights = torch.cat([self.bias.unsqueeze(1), self.weight], dim=1)

            delta_weights = self.delta_weights(x, weights)
            self.weight.add_(learning_rate * delta_weights[:, 1:])  # problem here
            self.bias.add_(learning_rate * delta_weights[:, 0].squeeze())
        else:
            delta_weights = self.delta_weights(x, weights)
            self.weight.add_(learning_rate * delta_weights)

        # p-norm of weights should converge to 1
        weights = weights.view(weights.size(0), -1)
        norm = torch.sum(torch.abs(weights) ** self.lebesgue_p, axis=1)
        convergence = torch.max(torch.abs(norm - torch.ones_like(norm))).cpu().numpy()

        return convergence

    def __str__(self) -> str:
        """String representation for the layer."""

        return (
            "BioLinear(\n"
            + f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias}, "
            + f"lebesgue_p={self.lebesgue_p}, ranking_param={self.ranking_param}, delta={self.delta}\n)"
        )
