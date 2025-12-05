"""
Base node classes for JAX predictive coding networks.

This module provides the abstract base class for all node types, defining the
interface for custom transfer functions, multiple input slots, and local gradient computation.
All node methods are pure functions (no side effects) for JAX compatibility.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from fabricpc.core.types import NodeParams, NodeState, NodeInfo


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


class FlattenInputMixin:
    """
    Mixin providing flatten/reshape utilities for dense (fully-connected) nodes.

    Use this mixin when your node needs to:
    - Flatten arbitrary-shaped inputs to 2D for matrix multiplication
    - Reshape flat outputs back to a target shape

    Example usage:
        @register_node("my_dense")
        class MyDenseNode(FlattenInputMixin, NodeBase):
            @staticmethod
            def forward(params, inputs, state, node_info):
                batch_size = state.z_latent.shape[0]
                out_shape = node_info.shape

                # Flatten inputs and compute linear transformation
                pre_activation = FlattenInputMixin.compute_linear(
                    inputs, params.weights, batch_size, out_shape
                )
                ...
    """

    @staticmethod
    def flatten_input(x: jnp.ndarray) -> jnp.ndarray:
        """
        Flatten input tensor to 2D: (batch, *shape) -> (batch, numel).

        Args:
            x: Input tensor with batch dimension first

        Returns:
            Flattened tensor of shape (batch, numel)
        """
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)

    @staticmethod
    def reshape_output(x_flat: jnp.ndarray, out_shape: Tuple[int, ...]) -> jnp.ndarray:
        """
        Reshape flat tensor to target shape: (batch, numel) -> (batch, *out_shape).

        Args:
            x_flat: Flat tensor of shape (batch, numel)
            out_shape: Target shape (excluding batch dimension)

        Returns:
            Reshaped tensor of shape (batch, *out_shape)
        """
        batch_size = x_flat.shape[0]
        return x_flat.reshape(batch_size, *out_shape)

    @staticmethod
    def compute_linear(
        inputs: Dict[str, jnp.ndarray],
        weights: Dict[str, jnp.ndarray],
        batch_size: int,
        out_shape: Tuple[int, ...]
    ) -> jnp.ndarray:
        """
        Compute linear transformation: sum of (flattened_input @ weight) for each edge.

        Flattens each input, applies matmul with corresponding weight matrix,
        accumulates results, and reshapes to output shape.

        Args:
            inputs: Dictionary mapping edge keys to input tensors
            weights: Dictionary mapping edge keys to weight matrices (in_numel, out_numel)
            batch_size: Batch size for output initialization
            out_shape: Target output shape (excluding batch)

        Returns:
            Pre-activation tensor of shape (batch, *out_shape)
        """
        import numpy as np
        out_numel = int(np.prod(out_shape))
        pre_activation_flat = jnp.zeros((batch_size, out_numel))

        for edge_key, x in inputs.items():
            x_flat = FlattenInputMixin.flatten_input(x)
            pre_activation_flat = pre_activation_flat + jnp.matmul(x_flat, weights[edge_key])

        return FlattenInputMixin.reshape_output(pre_activation_flat, out_shape)


class NodeBase(ABC):
    """
    Abstract base class for all predictive coding nodes.

    All methods are pure functions (no side effects) for JAX compatibility.
    Nodes can have multiple input slots and custom transfer functions.

    Required class attributes (validated during registration):
        - CONFIG_SCHEMA: dict specifying node configuration validation
        - DEFAULT_ENERGY_CONFIG: dict specifying default energy functional

    Example:
        @register_node("my_node")
        class MyNode(NodeBase):
            CONFIG_SCHEMA = {
                "my_param": {"type": int, "default": 10}
            }
            ...
    """

    # CONFIG_SCHEMA is required - subclasses must define it
    # Use empty dict {} if no additional config parameters are needed
    CONFIG_SCHEMA: Dict[str, Any]

    # DEFAULT_ENERGY_CONFIG - Can be overridden by default in subclass and per-node via node_config["energy"]
    DEFAULT_ENERGY_CONFIG: Dict[str, Any] = {"type": "gaussian"}

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
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],  # edge_key -> source shape
        config: Dict[str, Any]
    ) -> NodeParams:
        """
        Initialize parameters for this node.
        Describe the weights and biases structure in the docstring.

        Args:
            key: JAX random key
            node_shape: Output shape of this node (excluding batch dimension)
            input_shapes: Dictionary mapping edge keys to source node shapes
            config: Node configuration (may contain initialization settings)

        Returns:
            NodeParams with initialized weights and biases

        Example:
            For a linear node with inputs from edge "a->b:in":
            in_numel = int(np.prod(input_shapes["a->b:in"]))
            out_numel = int(np.prod(node_shape))
            weights = {"a->b:in": initialize_weights(key, (in_numel, out_numel))}
            biases = {"b": jnp.zeros((1,) + node_shape)}
            return NodeParams(weights=weights, biases=biases)
        """
        pass

    @staticmethod
    def forward_inference(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],  # EdgeInfo.key -> inputs data
        state: NodeState,  # state object for the present node
        node_info: NodeInfo,
    ) -> Tuple[NodeState, Dict[str, jnp.ndarray]]:
        """
        Forward pass: updates node state and computes gradients w.r.t. inputs.

        Args:
            params: Node parameters (weights, biases)
            inputs: Dictionary mapping edge keys to input tensors
            state: state object for the present node
            node_info: NodeInfo object (contains activation function, etc.)

        Returns:
            Tuple of (NodeState, gradient_wrt_inputs):
                - NodeState: updated node state (z_mu, pre_activation, etc.)
                - gradient_wrt_inputs: dictionary of gradients w.r.t. each input edge
        """
        from fabricpc.nodes import get_node_class
        node_class = get_node_class(node_info.node_type)

        # Use JAX's value_and_grad to compute gradients w.r.t. inputs
        (total_energy, new_state), input_grads = jax.value_and_grad(
            node_class.forward,
            argnums=1,  # inputs
            has_aux=True
        )(params, inputs, state, node_info)

        return new_state, input_grads

    @staticmethod
    def forward_learning(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,  # state object for the present node
        node_info: NodeInfo
    ) -> Tuple[NodeState, NodeParams]:
        """
        Forward pass: update state and compute gradients of weights for local learning.

        The local gradient for weights is: -(input.T @ gain_mod_error)

        Args:
            params: Current node parameters
            inputs: Dictionary with edge_key -> input tensor
            state: state object for the present node
            node_info: NodeInfo object

        Returns:
            Tuple of (NodeState, params_grad):
                - NodeState: updated node state (z_mu, pre_activation, etc.)
                - params_grad: NodeParams containing weight and bias gradients
        """
        from fabricpc.nodes import get_node_class
        node_class = get_node_class(node_info.node_type)

        # Use JAX's value_and_grad to compute gradients w.r.t. params
        (total_energy, new_state), params_grad = jax.value_and_grad(
            node_class.forward,
            argnums=0,  # params
            has_aux=True
        )(params, inputs, state, node_info)

        return new_state, params_grad

    @staticmethod
    @abstractmethod
    def forward(
            params: NodeParams,
            inputs: Dict[str, jnp.ndarray],  # EdgeInfo.key -> inputs data
            state: NodeState,  # state object for the present node
            node_info: NodeInfo,
    ) -> tuple[jax.Array, NodeState]:
        """
        Forward pass through the node, returning energy scalar and updated state.
        Computes:
            forward pass -> compute error -> compute energy -> total energy

        Args:
            params: Node parameters (weights, biases)
            inputs: Dictionary mapping edge keys to input tensors
            state: state object for the present node
            node_info: NodeInfo object (contains activation function, etc.)

        Returns:
            Tuple of (total_energy, NodeState):
                - total_energy: scalar energy value for this node
                - NodeState: updated node state (z_mu, pre_activation, etc.)
        """
        pass

    @staticmethod
    def energy_functional(
        state: NodeState,
        node_info: NodeInfo
    ) -> NodeState:
        """
        Compute the energy and the derivative with respect to the node's latent state.

        The energy functional to use is determined by node_info.node_config["energy"].
        If not specified, the node class's DEFAULT_ENERGY_CONFIG is used during
        graph construction.

        Energy config format:
            {
                "energy": {
                    "type": "gaussian",  # or "binaryce", "crossentropy", etc.
                    "precision": 1.0     # energy-specific parameters
                }
            }

        Args:
            state: NodeState object (contains z_latent, z_mu, etc.)
            node_info: NodeInfo object (contains energy config in node_config)

        Returns:
            Updated NodeState with:
                energy: energy value per batch element, shape (batch_size,)
                latent_grad: derivative of energy w.r.t. z_latent (accumulated)
        """
        from fabricpc.core.energy import get_energy_and_gradient

        # Get energy config from node_config (should be set during graph construction)
        energy_config = node_info.node_config.get("energy", None)
        if energy_config is None:
            raise ValueError(f"graph was improperly constructed. Node {node_info.name} is missing default energy functional.")

        # Compute energy and gradient using the energy registry
        energy, grad = get_energy_and_gradient(state.z_latent, state.z_mu, energy_config)

        # Accumulate gradient with existing latent_grad
        latent_grad = state.latent_grad + grad

        # Update node state
        state = state._replace(energy=energy, latent_grad=latent_grad)

        return state

    @staticmethod
    def get_energy_functional(energy_name: str):
        """
        Retrieve an energy functional class by name.

        Args:
            energy_name: Name of the energy functional (e.g., "gaussian", "bernoulli")

        Returns:
            The EnergyFunctional class

        Example:
            energy_class = NodeBase.get_energy_functional("bernoulli")
            energy = energy_class.energy(z_latent, z_mu, config)
        """
        from fabricpc.core.energy import get_energy_class
        return get_energy_class(energy_name)