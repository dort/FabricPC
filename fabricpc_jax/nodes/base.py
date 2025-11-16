"""
Base node classes for JAX predictive coding networks.

This module provides the abstract base class for all node types, defining the
interface for custom transfer functions, multiple input slots, and local gradient computation.
All node methods are pure functions (no side effects) for JAX compatibility.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Type, Optional, NamedTuple
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from fabricpc_jax.core.types import NodeParams

@dataclass(frozen=True)
class SlotSpec:
    """Specification for an input slot to a node."""
    name: str
    is_multi_input: bool  # True = multiple inputs allowed, False = single input only


@dataclass(frozen=True)
class Slot:
    """Runtime slot information with connected edges."""
    spec: SlotSpec
    in_neighbors: Dict[str, str]  # edge_key -> source_node_name mapping

class NodeBase(ABC):
    """
    Abstract base class for all predictive coding nodes.

    All methods are pure functions (no side effects) for JAX compatibility.
    Nodes can have multiple input slots and custom transfer functions.
    """

    @staticmethod
    @abstractmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """
        Define the input slots for this node type.

        Returns:
            Dictionary mapping slot names to SlotSpec objects

        Example:
            return {
                "in": SlotSpec(name="in", is_multi_input=True),
                "gate": SlotSpec(name="gate", is_multi_input=False)
            }
        """
        pass

    @staticmethod
    @abstractmethod
    def initialize_params(
        key: jax.Array,  # from jax.random.PRNGKey
        node_dim: int,
        input_dims: Dict[str, int],  # slot_name -> total input dimension
        config: Dict[str, Any]
    ) -> NodeParams:
        """
        Initialize parameters for this node.

        Args:
            key: JAX random key
            node_dim: Dimension of this node's output
            input_dims: Dictionary mapping slot names to total input dimensions
            config: Node configuration (may contain initialization settings)

        Returns:
            NodeParams with initialized weights and biases

        Example:
            For a linear node with one multi-input slot "in":
            weights = {"W": initialize_weights(key, (input_dims["in"], node_dim))}
            biases = {"b": jnp.zeros((1, node_dim))}
            return NodeParams(weights=weights, biases=biases)
        """
        pass

    @staticmethod
    @abstractmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],  # EdgeInfo.key -> inputs data
        config: Dict[str, Any],
        node_out_shape: Tuple[int, ...], # shape of the node output
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass through the node.

        Args:
            params: Node parameters (weights, biases)
            inputs: Dictionary mapping edge keys to input tensors
            config: Node configuration (contains activation function, etc.)
            node_out_shape: shape of the node output (same as latent state)

        Returns:
            Tuple of (z_mu, pre_activation):
                - z_mu: Output after activation function
                - pre_activation: Output before activation function

        Example:
            pre_act = jnp.matmul(inputs["in"], params.weights["W"]) + params.biases["b"]
            z_mu = activation_fn(pre_act)
            return z_mu, pre_act
        """
        pass

    @staticmethod
    def compute_jacobian(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],  # EdgeInfo.key -> inputs data
        config: Dict[str, Any],
        node_out_shape: Tuple[int, ...],
    ) -> Dict[str, jnp.ndarray]:  # EdgeInfo.key -> Jacobian matrix
        """
        Compute Jacobian of output w.r.t. a specific input source.

        Default implementation uses JAX automatic differentiation.
        Can be overridden for non-differentiable nodes with custom derivatives.

        Args:
            params: Node parameters
            inputs: Dictionary mapping slot names to concatenated input tensors
            config: Node configuration
            node_out_shape: shape of the node output (same as latent state)

        Returns:
            dictionary of Jacobian matrix of shape (input_dim, output_dim) for each edge key
        """
        # Default: use JAX autodiff
        def output_fn(source_input, edge_str: str):
            # Replace the specific source input
            modified_inputs = inputs.copy()
            modified_inputs[edge_str] = source_input
            z_mu, _ = NodeBase.forward(params, modified_inputs, config, node_out_shape)
            return z_mu

        jacobian_dict = {}
        for edge_key in inputs.keys():
            jacobian = jax.jacobian(output_fn)(inputs[edge_key], edge_key)

            # jacobian has shape (output_dim, batch_size, input_dim, batch_size)
            # Take the first batch element
            jacobian_single = jacobian[:, 0, :, 0]  # (output_dim, input_dim)
            jacobian_dict[edge_key] = jacobian_single
        return jacobian_dict

    @staticmethod
    @abstractmethod
    def compute_params_gradient(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        gain_mod_error: jnp.ndarray,
        config: Dict[str, Any]
    ) -> NodeParams:
        """
        Compute gradients of weights for local learning.

        The local gradient for weights is: -(input.T @ gain_mod_error)

        Args:
            params: Current node parameters
            inputs: Dictionary mapping slot names to concatenated input tensors
            gain_mod_error: Gain-modulated error (error * activation_derivative)
            config: Node configuration

        Returns:
            NodeParams containing weight and bias gradients
        """
        # TODO autograd as default, override in subclass for efficiency
        pass
