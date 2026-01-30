"""Shared layers for Artificial-Dopamine (AD) models."""
from abc import ABC
from dataclasses import field
from functools import partial
from typing import Callable, Optional, Sequence, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp


class ADCell(nn.Module, ABC):
    r"""Base class for Artificial-Dopamine cells.

    The forward pass computes hidden activations :math:`h` and a predicted
    output :math:`y` from an input :math:`x`.

    Args:
        hidden_features: the number of hidden features.
        out_features: number of output features.

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Activations: :math:`(N, *, H_{act})` where all but the last dimension are
          the same shape as the input and :math:`H_{act} = \text{hidden\_features}`.
        - Output: :math:`(N, *, H_{out})` where all but the last dimension are
          the same shape as the input and :math:`H_{out} = \text{out\_features}`.
    """

    hidden_features: int
    n_actions: int

    def setup(self) -> None:
        """Setup the layer."""
        raise NotImplementedError

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Compute the forward pass.

        Args:
            x: The input tensor.

        Returns:
            The hidden activations and the output tensor.
        """
        raise NotImplementedError


@partial(jax.jit, static_argnames=('k', 'pad'))
def k_folding(x: jax.Array, p0: jax.Array, k: int, pad: bool = True) \
        -> jax.Array:
    r"""Perform a k-folding operation on the input array.

    Applies :math:`k` equally spaced folds to the input array along the
    last dimension to produce an array of shape :math:`(N, *, k)`.

    The output is a :math:`k`-dimensional vector where the :math:`i`-th component
    is the dot product between the activations and the :math:`i`-th weight vector.

    Args:
        x: The input array. An array of shape :math:`(N, *, H_{in})` where
            :math:`*` means any number of dimensions including none and
            :math:`H_{in}` is the number of input features. Note that the last
            dimension must be divisible by :math:`k`.
        p0: The activations. An array of shape where all but the last dimension
            are the same shape as ``x``. The last dimension should be the size
            of a single fold, i.e. :math:`H_{in} / k`.
        k: The number of folds to apply to the input array. Must be positive.
        pad: If ``True``, the input array is padded with zeros along the last
            dimension to ensure that it is divisible by ``k``. The activations
            ``p0`` are also padded with zeros to match the shape of each folded
            component of the input array. If ``False``, an error is raised if
            the last dimension of ``x`` is not divisible by ``k``.

    Returns:
        An array of shape :math:`(N, *, k)` where all but the last dimension are
        the same shape as ``x``.

    Remarks:
        The k-folding operation can be thought of a downsampling operation where
        the :math:`i`-th component of the output array is a linear combination
        of the activations with coefficients determined by the :math:`i`-th
        partition of the input array.

    Raises:
        ValueError: If ``k`` is not positive.
        ValueError: If the shape of ``x`` and ``p0`` are not the same except
            for the last dimension.
        ValueError: If the last dimension of ``x`` is not divisible by ``k``
            and ``pad`` is ``False``.

    Examples::

        >>> x = jax.random.normal(jax.random.PRNGKey(0), (128, 256))
        >>> p0 = jax.random.normal(jax.random.PRNGKey(0), (128, 64))
        >>> y = k_folding(x, p0, 4)
        >>> y.shape
        (128, 4)
        >>> x = jnp.asarray([[1, 2, 3, 4, 5, 6, 7, 8]])
        >>> p0 = jnp.asarray([[-1, 2, 0.5, 0]])
        >>> y = k_folding(x, p0, 2)
        >>> y.tolist()
        [[4.5, 10.5]]
    """
    # Ensure that k is positive
    if k <= 0:
        raise ValueError(f'k must be positive, but got {k}.')

    # Ensure that x and p0 have the same shape except for the last dimension
    if x.shape[:-1] != p0.shape[:-1]:
        raise ValueError(
            f'Input array and activations must have the same shape except '
            f'for the last dimension, but got {x.shape} and {p0.shape}.'
        )

    # Ensure that the last dimension of x is divisible by k, padding if necessary
    h_in = x.shape[-1]
    if h_in % k != 0:
        if pad:
            x = jnp.pad(x, ((0, 0),) * (x.ndim - 1) + ((0, k - h_in % k),))
            h_in = x.shape[-1]
        else:
            raise ValueError(
                f'Input array must have a last dimension divisible by '
                f'{k}, but has shape {x.shape}. Please set pad=True to '
                f'automatically pad it with zeros.'
            )

    # Ensure that p0 has the correct number of features
    chunk_size = int(h_in // k)
    if p0.shape[-1] != chunk_size:
        if pad:
            p0 = jnp.pad(p0, ((0, 0),) * (p0.ndim - 1) + ((0, chunk_size - p0.shape[-1]),))
        else:
            raise ValueError(
                f'Activations must have {chunk_size} features, but got '
                f'{p0.shape[-1]} instead. Please set pad=True to '
                f'automatically pad it with zeros.'
            )

    # Reshape x to be a (k x chunk_size) matrix of weights
    x = x.reshape(x.shape[:-1] + (k, chunk_size))
    return jnp.matmul(x, jnp.expand_dims(p0, axis=-1)).squeeze(-1)


class AttentionADCell(ADCell):
    r"""An Artificial-Dopamine cell with an attention mechanism.

    Args:
        hidden_features: the number of features in the hidden activations
        out_features: number of features in the output
        num_heads: The number of heads. Default: ``8``
        attn_weights: Whether to return the attention weights in the
            forward pass. Default: ``False``.
        use_binary_actions: Whether to use binary action encoding instead of
            one-hot encoding. Uses log2(n_actions) binary values (-1 or 1). 
            Default: ``False``.

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Activations: :math:`(N, *, H_{act})` where all but the last dimension are
          the same shape as the input and :math:`H_{act} = \text{hidden\_features}`.
        - Output: :math:`(N, *, H_{out})` where all but the last dimension are
          the same shape as the input and :math:`H_{out} = \text{out\_features}`.
        - Attention weights: :math:`(N, *, H_{act}, H_{out})` where all but the
            last two dimensions are the same shape as the input and
            :math:`H_{act} = \text{hidden\_features}` and
            :math:`H_{out} = \text{out\_features}`.
    """
    num_heads: int = 8
    attn_weights: bool = False
    goodness_type: str = ""
    input_conditioning: bool = True
    use_binary_actions: bool = False

    def setup(self) -> None:
        """Setup the layer."""
        self.fc_h = nn.Dense(self.hidden_features)
        self.fc_z1 = nn.Dense(self.num_heads * self.hidden_features, use_bias=False)
        if not self.input_conditioning:
            self.fc_z2 = nn.Dense(self.num_heads * self.n_actions * self.hidden_features, use_bias=False)
        else:
            self.fc_z2 = nn.Dense(self.num_heads * self.hidden_features, use_bias=False)
        self.layer_norm = nn.LayerNorm(1e-6)
        
        # Pre-compute binary encoding dimensions and patterns for scalar actions
        if self.use_binary_actions:
            import math
            self.k = int(math.log2(self.n_actions))  # Number of binary dimensions needed
            assert 2**self.k == self.n_actions, f"n_actions ({self.n_actions}) must be a power of 2 for binary encoding, got 2^{self.k}={2**self.k}"
            
            # Pre-compute binary action patterns
            action_indices = jnp.arange(self.n_actions)
            binary_actions = []
            for i in range(self.k):
                bit = (action_indices >> i) & 1  # Extract i-th bit
                binary_bit = 2 * bit - 1  # Map 0->-1, 1->1
                binary_actions.append(binary_bit)
            self.binary_actions = jnp.stack(binary_actions, axis=1).astype(jnp.float32)  # (n_actions, k)

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Compute the forward pass.

        Args:
            x: The input tensor.
            weights: Whether to return the weights. Default: ``False``.

        Returns:
            The output tensor and the attention weights.
        """
        if not self.input_conditioning:
            num_heads, d_h, d_g = self.num_heads, self.hidden_features, self.n_actions * self.hidden_features
        else:
            num_heads, d_h, d_g = self.num_heads, self.hidden_features, self.hidden_features
        sz_b = x.shape[0]

        # Apply linear projection fc_h to get the hidden activations
        h = self.layer_norm(nn.relu(self.fc_h(x)))

        # INSERT ACTION AS INPUT
        if self.input_conditioning:
            if self.use_binary_actions:
                # For binary action encoding: use k binary values (-1 or 1) where 2^k = n_actions
                # Use pre-computed binary action patterns
                action_candidates = jnp.tile(self.binary_actions, (sz_b, 1))  # (sz_b * n_actions, k)
                x = jnp.expand_dims(x, axis=1)                     # (B, 1, D)
                x = jnp.tile(x, (1, self.n_actions, 1))                          # (B, n_actions, D)
                x = x.reshape(-1, x.shape[-1])                      # (B * n_actions, D)
                x = jnp.concatenate([x, action_candidates], axis=-1)
                sz_b *= self.n_actions
            else:
                # Original one-hot encoding for discrete actions
                action_candidates = jnp.tile(jnp.arange(self.n_actions), (sz_b,))
                x = jnp.expand_dims(x, axis=1)                     # (B, 1, D)
                x = jnp.tile(x, (1, self.n_actions, 1))                          # (B, n_actions, D)
                x = x.reshape(-1, x.shape[-1])                      # (B * n_actions, D)
                action_one_hot = jax.nn.one_hot(action_candidates, num_classes=self.n_actions).astype(jnp.float32)
                x = jnp.concatenate([x, action_one_hot], axis=-1)
                sz_b *= self.n_actions

        # Apply linear projections fc_z1 and fc_z2
        z1 = jnp.reshape(self.fc_z1(x), (sz_b, num_heads, d_h))  # (sz_b, num_heads, d_h)
        z2 = jnp.reshape(self.fc_z2(x), (sz_b, num_heads, d_g))  # (sz_b, num_heads, d_g)

        # Multiply z1 and z2 to get a d_g x d_h weight matrix
        w = jnp.matmul(z2.transpose(0, 2, 1), z1)  # (sz_b, d_g, d_h)
        w = (w - jnp.mean(w, axis=-1, keepdims=True)) / (jnp.std(w, axis=-1, keepdims=True) + 1e-6)
        w = nn.tanh(w)

        # Repeat hidden activations, then apply the weight matrix to the hidden activations to get the output
        if self.input_conditioning:
            v = jnp.expand_dims(h, axis=1)  
            v = jnp.tile(v, (1, self.n_actions, 1)) 
            v = v.reshape(-1, v.shape[-1])   
            v = jnp.expand_dims(v, axis=-1) 
            y_ = jnp.matmul(w, v).squeeze(-1)  # (sz_b, d_g)

        # for action input conditioning
        else:
            y = jnp.matmul(w, jnp.expand_dims(h, axis=-1)).squeeze(-1)  # (sz_b, d_g)
            y_ = jnp.reshape(y, (sz_b * self.n_actions, -1))

        y = self._calc_logits(y_, 0)   # comment out when not using msq

        q = jnp.reshape(y, (-1, self.n_actions))

        if self.attn_weights:
            return h, q, w
        else:
            return h, q, y_

    def _calc_logits(self, z: jnp.ndarray, layer_idx: int, update_baseline: bool = True) -> jnp.ndarray:
        if self.goodness_type == 'msq':
            logits = jnp.mean(jnp.square(z), axis=tuple(range(1, z.ndim)))
            logits = logits[:, None]  # same as unsqueeze(1)

        elif self.goodness_type == 'mean':
            logits = jnp.mean(z, axis=tuple(range(1, z.ndim)))
            logits = logits[:, None]

        elif self.goodness_type == 'std':
            logits = jnp.std(z, axis=tuple(range(1, z.ndim)))
            logits = logits[:, None]

        elif self.goodness_type == 'rms':
            logits = jnp.sqrt(jnp.mean(jnp.square(z), axis=tuple(range(1, z.ndim))))
            logits = logits[:, None]

        elif self.goodness_type == 'variance':
            logits = jnp.var(z, axis=tuple(range(1, z.ndim)))
            logits = logits[:, None]

        elif self.goodness_type == 'weighted':
            logits = self.goodness_weights[layer_idx](z)

        return logits
