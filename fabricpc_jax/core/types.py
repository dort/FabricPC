"""
Core JAX types for predictive coding networks.

All types are immutable and registered as JAX pytrees for automatic differentiation.
"""

from typing import Dict, Any, Tuple, NamedTuple, Optional
import jax.numpy as jnp
from jax import tree_util
from dataclasses import dataclass


@dataclass(frozen=True)
class SlotInfo:
    """Metadata for an input slot to a node."""
    name: str  # Slot name (e.g., "in")
    parent_node: str  # Name of the parent node
    is_multi_input: bool  # True if slot accepts multiple edges, False for single edge
    in_neighbors: Tuple[str, ...]  # TODO resolve definition: Tuple of source node names, Tuple of edge keys connecting to this slot

@dataclass(frozen=True)
class NodeInfo:
    """Metadata for a single node in the graph."""

    name: str
    dim: int
    node_type: str  # "linear", "transformer", etc.
    node_config: Dict[str, Any]
    activation_config: Dict[str, Any]  # {"type": "sigmoid", ...}
    slots: Dict[str, SlotInfo]  # {"in": SlotInfo, ...}
    in_degree: int  # Number of incoming edges
    out_degree: int  # Number of outgoing edges
    in_edges: Tuple[str, ...]  # Tuple of edge keys
    out_edges: Tuple[str, ...]  # Tuple of edge keys

@dataclass(frozen=True)
class EdgeInfo:
    """Metadata for a single edge in the graph."""

    key: str  # "source->target:slot"
    source: str
    target: str
    slot: str

class NodeParams(NamedTuple):
    """Parameters for a single node (weights, biases, etc.)."""
    weights: Dict[str, jnp.ndarray]  # Named weight matrices, where name identifies the substructure of the node for the parameters
    biases: Dict[str, jnp.ndarray]   # Named bias vectors

class GraphParams(NamedTuple):
    """
    Learnable parameters of the predictive coding network.

    Now organized by node rather than edge, supporting complex nodes with
    multiple internal parameters (e.g., transformer blocks).

    Attributes:
        nodes: Dictionary mapping node names to their parameters
            Each node has a dict of weights and a dict of biases
            Complex nodes may have multiple named weight/bias matrices
    """

    nodes: Dict[str, NodeParams]  # {node_name: NodeParams}

    def __repr__(self) -> str:
        n_nodes = len(self.nodes)
        total_params = 0
        for node_params in self.nodes.values():
            if "weights" in node_params._fields:
                total_params += sum(w.size for w in node_params.weights.values())
            if "biases" in node_params._fields:
                total_params += sum(b.size for b in node_params.biases.values())
        return f"GraphParams(nodes={n_nodes}, total_params={total_params})"


class GraphState(NamedTuple):
    """
    Dynamic state of the network during inference.

    All states are dictionaries mapping node names to arrays.

    Attributes:
        z_latent: Latent states (what the network infers)
        z_mu: Predicted expectations (what the network predicts)
        error: Prediction errors (z_latent - z_mu)
        pre_activation: Pre-activation values (before activation function)
        gain_mod_error: Gain-modulated errors (error * activation_derivative)
        latent_grad: Gradients w.r.t. latent states for inference updates
    """

    z_latent: Dict[str, jnp.ndarray]
    z_mu: Dict[str, jnp.ndarray]
    error: Dict[str, jnp.ndarray]
    energy: Dict[str, jnp.ndarray]
    pre_activation: Dict[str, jnp.ndarray]
    gain_mod_error: Dict[str, jnp.ndarray]  # TODO deprecate
    latent_grad: Dict[str, jnp.ndarray]  # For local gradient accumulation

    def __repr__(self) -> str:
        n_nodes = len(self.z_latent)
        batch_size = next(iter(self.z_latent.values())).shape[0] if self.z_latent else 0
        return f"GraphState(nodes={n_nodes}, batch_size={batch_size})"


class GraphStructure(NamedTuple):
    """
    Static graph topology (compile-time constant).

    This structure is immutable and marked as static during JIT compilation.

    Attributes:
        nodes: Dictionary mapping node names to NodeInfo
        edges: Dictionary mapping edge keys to EdgeInfo
        task_map: Dictionary mapping task names to node names
        node_order: Topological order for forward pass
    """

    nodes: Dict[str, NodeInfo]
    edges: Dict[str, EdgeInfo]
    task_map: Dict[str, str]
    node_order: Tuple[str, ...]  # Topological sort for inference

    def __repr__(self) -> str:
        n_nodes = len(self.nodes)
        n_edges = len(self.edges)
        return f"GraphStructure(nodes={n_nodes}, edges={n_edges})"


# Register as pytrees for JAX transformations
tree_util.register_pytree_node(
    GraphParams,
    lambda gp: ((gp.nodes,), None),
    lambda aux, children: GraphParams(*children),
)

tree_util.register_pytree_node(
    NodeParams,
    lambda np: ((np.weights, np.biases), None),
    lambda aux, children: NodeParams(*children),
)

tree_util.register_pytree_node(
    GraphState,
    lambda gs: (
        (gs.z_latent, gs.z_mu, gs.error, gs.energy, gs.pre_activation, gs.gain_mod_error, gs.latent_grad),
        None,
    ),
    lambda aux, children: GraphState(*children),
)

# GraphStructure is static, so we register it as having no dynamic components
tree_util.register_pytree_node(
    GraphStructure,
    lambda gs: ((), (gs.nodes, gs.edges, gs.task_map, gs.node_order)),
    lambda aux, _: GraphStructure(*aux),  # Reconstruct from aux data
)
