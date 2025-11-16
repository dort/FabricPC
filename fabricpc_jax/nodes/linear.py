"""
Linear node implementation for JAX predictive coding networks.

This implements a linear transformation node with configurable activation functions.
The node has a single multi-input slot that accepts multiple incoming connections.
"""

from typing import Dict, Any, Tuple
import jax
import jax.numpy as jnp

from fabricpc_jax.nodes.base import NodeBase, SlotSpec
from fabricpc_jax.core.types import NodeParams
from fabricpc_jax.core.activations import get_activation
from fabricpc_jax.core.initialization import initialize_weights


class LinearNode(NodeBase):
    """
    Linear transformation node: y = activation(W @ x + b)

    This node type:
    - Has a single multi-input slot named "in"
    - Concatenates all inputs and applies a linear transformation
    - Supports various activation functions
    - Implements local Hebbian learning
    """

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """
        Linear nodes have a single multi-input slot.

        Returns:
            Dictionary with one slot "in" that accepts multiple inputs
        """
        return {
            "in": SlotSpec(name="in", is_multi_input=True)
        }

    @staticmethod
    def initialize_params(
        key: jax.Array,  # from jax.random.PRNGKey
        node_dim: int,
        input_dims: Dict[str, int],
        config: Dict[str, Any]
    ) -> NodeParams:
        """
        Initialize weight matrix and bias vector.

        Args:
            key: JAX random key
            node_dim: Output dimension of this node
            input_dims: Dictionary with EdgeInfo.key -> input dimension for that edge
            config: Node configuration with weight_init settings

        Returns:
            NodeParams with initialized W and b
        """
        # Counter for total input dimension from the "in" slot
        total_in_dim = 0

        # Get weight initialization config
        default_cfg = {"type": "normal", "mean": 0.0, "std": 0.05}
        weight_init_config = config.get("weight_init", default_cfg)

        # Split key for weights and biases
        key_w, key_b = jax.random.split(key)

        # Initialize weight matrix
        # "in" slot in multiinput; create separate weights for each incoming edge
        weights_dict = {}
        rand_key_w = dict(zip(input_dims.keys(), jax.random.split(key_w, len(input_dims))))
        for edge_key, in_dim in input_dims.items():
            if ":in" not in edge_key:
                raise ValueError(f"Linear node requires 'in' slot dimension. got edge key {edge_key}")  # validate that edges correspond to "in" slot
            weights_dict[edge_key] = initialize_weights(weight_init_config, rand_key_w[edge_key], (in_dim, node_dim))
            total_in_dim += in_dim

        # Initialize bias (usually zeros)
        use_bias = config.get("use_bias", True)
        if use_bias:
            b = jnp.zeros((1, node_dim))

        return NodeParams(
            weights=weights_dict,
            biases={"b": b} if use_bias else {}
        )

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        config: Dict[str, Any],
        node_out_shape: Tuple[int, ...],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass: linear transformation with activation.

        Args:
            params: Node parameters (weights, biases)
            inputs: Dictionary mapping edge keys to input tensors
            config: Node configuration (contains activation function, etc.)
            node_out_shape: shape of the node output (same as latent state)

        Returns:
            Tuple of (z_mu, pre_activation)
        """

        pre_activation = jnp.zeros(node_out_shape)

        # Linear transformation
        for edge_key, x in inputs.items():
            pre_activation = pre_activation + jnp.matmul(x, params.weights[edge_key])

        # Add bias if present
        if "b" in params.biases and params.biases["b"].size > 0:
            pre_activation = pre_activation + params.biases["b"]

        # Apply activation function
        default_activ = {"type": "identity"}
        activation_fn, _ = get_activation(config.get("activation", default_activ))
        z_mu = activation_fn(pre_activation)

        return z_mu, pre_activation

    @staticmethod
    def compute_jacobian(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],  # EdgeInfo.key -> inputs data
        config: Dict[str, Any],
        node_out_shape: Tuple[int, ...],
    ) -> Dict[str, jnp.ndarray]:  # EdgeInfo.key -> Jacobian matrix
        """
        Direct computation of Jacobian for linear nodes.

        For linear nodes, the Jacobian is simply the weight matrix slice.

        Args:
            params: Node parameters
            inputs: Dictionary mapping slot names to concatenated input tensors
            config: Node configuration
            node_out_shape: shape of the node output (same as latent state)

        Returns:
            dictionary of Jacobian matrix of shape (input_dim, output_dim) for each edge key
        """
        jacobian_dict = {}
        for edge_key in inputs.keys():
            jacobian_dict[edge_key] = params.weights[edge_key].T  # shape after transpose (dim_node_latent, dim_input)

        # TODO apply the activation derivative
        return jacobian_dict

    @staticmethod
    def compute_params_gradient(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        gain_mod_error: jnp.ndarray,
        config: Dict[str, Any]
    ) -> NodeParams:
        """
        Compute local gradients for weights and biases.

        For linear nodes:
        - Weight gradient: -(input.T @ gain_mod_error)
        - Bias gradient: -sum(gain_mod_error, axis=0)

        Args:
            params: Current node parameters
            inputs: Dictionary with edge_key -> input tensor
            gain_mod_error: Error weighted by activation derivative, just this node
            config: Node configuration

        Returns:
            NodeParams containing gradients
        """

        # fix the test file line 203 to check for params keyed on edge strings

        weight_grads = {}
        bias_grads = {}

        # Weight gradient
        for edge_key, in_tensor in inputs.items():
            grad_w = -jnp.matmul(in_tensor.T, gain_mod_error)
            weight_grads[edge_key] = grad_w

        # Bias gradient
        if "b" in params.biases:
            grad_b = -jnp.sum(gain_mod_error, axis=0, keepdims=True)
            bias_grads["b"] = grad_b

        return NodeParams(weights=weight_grads, biases=bias_grads)
