"""
Core inference dynamics for JAX predictive coding networks with local Hebbian learning.

This module implements the functional inference loop that updates latent states
using local gradients computed via Jacobian for true predictive coding.
"""

from typing import Dict, Tuple, Optional
from functools import partial
import jax
import jax.numpy as jnp

from fabricpc_jax.core.types import GraphParams, GraphState, GraphStructure
from fabricpc_jax.core.activations import get_activation
from fabricpc_jax.nodes import get_node_class_from_type
from fabricpc_jax.core.types import NodeParams, NodeInfo


def gather_inputs(
        node_info: NodeInfo,
        structure: GraphStructure,
        z_latent: Dict[str, jax.Array],  # node_name -> latent state
) -> Dict[str, jax.Array]:
    """
    Gather inputs for a node from the graph structure.
    """
    in_edges_data = {}
    for edge_key in node_info.in_edges:
        edge_info = structure.edges[edge_key]  # get the edge object
        in_edges_data[edge_key] = z_latent[edge_info.source]  # get the data sent along this edge

    return in_edges_data


def compute_node_projection(
    params: GraphParams,
    z_latent: Dict[str, jnp.ndarray],
    node_name: str,
    structure: GraphStructure,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute prediction z_mu for a node from its incoming connections.

    Args:
        params: Model parameters organized by node
        z_latent: Current latent states for all nodes
        node_name: Name of the node to compute prediction for
        structure: Static graph structure

    Returns:
        Tuple of (z_mu, pre_activation)
    """
    node_info = structure.nodes[node_name]
    node_class = get_node_class_from_type(node_info.node_type)

    # Source nodes (no incoming edges) have zero prediction
    if node_info.in_degree == 0:
        batch_size = next(iter(z_latent.values())).shape[0]
        zero_pred = jnp.zeros((batch_size, node_info.dim))
        return zero_pred, zero_pred

    # Get node's parameters
    default_empty = NodeParams(weights={}, biases={})
    node_params = params.nodes.get(node_name, default_empty)

    # Gather inputs for each slot
    in_edges_data = gather_inputs(node_info, structure, z_latent)

    # Forward pass through node
    z_mu, pre_activation = node_class.forward(
        node_params,
        in_edges_data,
        node_info.node_config,
        z_latent[node_name].shape
    )
    return z_mu, pre_activation


def compute_latent_gradients_local(
    error: Dict[str, jnp.ndarray],
    gain_mod_error: Dict[str, jnp.ndarray],
    params: GraphParams,
    z_latent: Dict[str, jnp.ndarray],
    structure: GraphStructure,
) -> Dict[str, jnp.ndarray]:
    """
    Compute gradient of energy w.r.t. latent states using local Jacobian.

    For each node i:
        grad_i = error_i - sum_j (error_j @ jacobian_{j->i})

    error_i (batch_size, dim_i)
    error_j (batch_size, dim_j)
    jacobian_{j->i} (dim_j, dim_i), del z_mu_j / del z_i

    This implements true predictive coding with local gradient computation.

    Args:
        error: Prediction errors for all nodes
        gain_mod_error: Gain-modulated errors for all nodes
        params: Model parameters
        z_latent: Current latent states keyed on node names
        structure: Graph structure

    Returns:
        Dictionary of gradients w.r.t. latent states
    """
    # Zero the latent gradients
    latent_grads = {}  # dimension Dict[node_name: (batch_size, dim_node_latent)]
    for node in structure.nodes:
        latent_grads[node] = jnp.zeros_like(z_latent[node])

    # Start with local error contribution
    for node in structure.nodes:
        latent_grads[node] = latent_grads[node] + error[node].copy()

    # Backpropagate errors to pre-synaptic nodes through Jacobians
    for node, node_info in structure.nodes.items():
        node_class = get_node_class_from_type(node_info.node_type)

        # Collect edge inputs for Jacobian computation
        edge_inputs = {}
        for in_edge_key in node_info.in_edges:
            in_edge_info = structure.edges[in_edge_key]
            edge_inputs[in_edge_key] = z_latent[in_edge_info.source]

        # Compute Jacobian for in_edges of the node
        jacobian_dict = node_class.compute_jacobian(
            params.nodes[node], edge_inputs, node_info.node_config, z_latent[node].shape
        )  #  dimension Dict[edge_name: (dim_source_latent, dim_target_latent)]

        # Backpropagate error on edges
        for edge_key, jacob in jacobian_dict.items():
            edge_info = structure.edges[edge_key]
            source_name = edge_info.source
            latent_grads[source_name] = latent_grads[source_name] - jnp.matmul(error[node], jacob)

    return latent_grads


def compute_all_projections(
    params: GraphParams,
    z_latent: Dict[str, jnp.ndarray],
    structure: GraphStructure,
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
    """
    Compute predictions for all nodes in the graph.

    Args:
        params: Model parameters
        z_latent: Current latent states
        structure: Graph structure

    Returns:
        Tuple of (z_mu_dict, pre_activation_dict)
    """
    z_mu = {}
    pre_activation = {}

    # Use node_order for efficient traversal
    for node_name in structure.node_order:
        z_mu[node_name], pre_activation[node_name] = compute_node_projection(
            params, z_latent, node_name, structure
        )

    return z_mu, pre_activation


def compute_errors(
    z_latent: Dict[str, jnp.ndarray],
    z_mu: Dict[str, jnp.ndarray],
    pre_activation: Dict[str, jnp.ndarray],
    structure: GraphStructure,
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
    """
    Compute prediction errors and gain-modulated errors.

    Args:
        z_latent: Current latent states
        z_mu: Predicted states
        pre_activation: Pre-activation values
        structure: Graph structure

    Returns:
        Tuple of (error, gain_mod_error, energy)
    """
    error = {}
    gain_mod_error = {}
    energy = {}

    for node_name in structure.nodes:
        node_info = structure.nodes[node_name]

        # Compute basic error
        err = z_latent[node_name] - z_mu[node_name]
        error[node_name] = err
        energy[node_name] = jnp.sum(err ** 2)  # TODO call the node energy functional method

        # Compute gain-modulated error
        if node_info.in_degree == 0:
            # Source nodes have no prediction
            gain_mod_error[node_name] = jnp.zeros_like(err)
        else:
            _, deriv_fn = get_activation(node_info.activation_config)
            gain = deriv_fn(pre_activation[node_name])
            gain_mod_error[node_name] = err * gain

    return error, gain_mod_error, energy


def inference_step(
    params: GraphParams,
    state: GraphState,
    clamps: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    eta_infer: float,
) -> GraphState:
    """
    Single inference step with local gradient computation.

    Args:
        params: Model parameters
        state: Current graph state
        clamps: Dictionary of clamped values
        structure: Graph structure
        eta_infer: Inference learning rate

    Returns:
        Updated graph state
    """
    # 1. Compute predictions for all nodes
    z_mu, pre_activation = compute_all_projections(params, state.z_latent, structure)

    # 2. Compute errors
    error, gain_mod_error, energy = compute_errors(
        state.z_latent, z_mu, pre_activation, structure
    )

    # 3. Compute LOCAL gradient using Jacobian
    latent_grads = compute_latent_gradients_local(
        error, gain_mod_error, params, state.z_latent, structure
    )

    # 4. Update latent states
    new_z_latent = {}
    for node_name in structure.nodes:
        if node_name in clamps:
            # Keep clamped nodes fixed
            new_z_latent[node_name] = clamps[node_name]
        else:
            # Update via gradient descent
            new_z_latent[node_name] = (
                state.z_latent[node_name] - eta_infer * latent_grads[node_name]
            )

    return GraphState(
        z_latent=new_z_latent,
        z_mu=z_mu,
        error=error,
        energy=energy,
        pre_activation=pre_activation,
        gain_mod_error=gain_mod_error,
        latent_grad=latent_grads,  # Store gradients for inspection
    )


def inference_step_parallel(
    params: GraphParams,
    state: GraphState,
    clamps: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    eta_infer: float,
) -> GraphState:
    """
    Single inference step with parallel node updates using vmap.

    This is an optimized version that updates all nodes in parallel.

    Args:
        params: Model parameters
        state: Current graph state
        clamps: Dictionary of clamped values
        structure: Graph structure
        eta_infer: Inference learning rate

    Returns:
        Updated graph state
    """
    # Stack all latent states into a single array for parallel processing
    node_names = list(structure.nodes.keys())
    batch_size = next(iter(state.z_latent.values())).shape[0]

    [print(state.z_latent[name].shape) for name in node_names]
    # Create stacked arrays
    z_latent_stacked = jnp.stack([state.z_latent[name] for name in node_names], axis=0)

    # Define per-node update function
    def update_node(node_idx, z_latent_node):
        node = node_names[node_idx]
        node_info = structure.nodes[node]

        # Compute projection for this node
        z_latent_dict = {name: z_latent_stacked[i] for i, name in enumerate(node_names)}
        z_mu_node, pre_act_node = compute_node_projection(
            params, z_latent_dict, node, structure
        )

        # Compute error
        error_node = z_latent_node - z_mu_node
        energy_node = jnp.sum(error_node ** 2)  # TODO call the node energy functional method
        # Compute gain-modulated error
        if node_info.in_degree > 0:
            _, deriv_fn = get_activation(node_info.activation_config)
            gain = deriv_fn(pre_act_node)
            gain_mod_error_node = error_node * gain
        else:
            gain_mod_error_node = jnp.zeros_like(error_node)

        return z_mu_node, pre_act_node, error_node, gain_mod_error_node, energy_node

    # Apply vmap over all nodes
    node_indices = jnp.arange(len(node_names))
    z_mu_stacked, pre_act_stacked, error_stacked, gain_mod_stacked, energy_stacked = jax.vmap(
        update_node, in_axes=(0, 0)
    )(node_indices, z_latent_stacked)

    # Convert back to dictionaries
    z_mu = {name: z_mu_stacked[i] for i, name in enumerate(node_names)}
    pre_activation = {name: pre_act_stacked[i] for i, name in enumerate(node_names)}
    error = {name: error_stacked[i] for i, name in enumerate(node_names)}
    gain_mod_error = {name: gain_mod_stacked[i] for i, name in enumerate(node_names)}
    energy = {name: energy_stacked[i] for i, name in enumerate(node_names)}

    # Compute gradients (still sequential for now, could be optimized)
    latent_grads = compute_latent_gradients_local(
        error, gain_mod_error, params, state.z_latent, structure
    )

    # Update latent states
    new_z_latent = {}
    for node_name in node_names:
        if node_name in clamps:
            new_z_latent[node_name] = clamps[node_name]
        else:
            new_z_latent[node_name] = (
                state.z_latent[node_name] - eta_infer * latent_grads[node_name]
            )

    return GraphState(
        z_latent=new_z_latent,
        z_mu=z_mu,
        error=error,
        energy=energy,
        pre_activation=pre_activation,
        gain_mod_error=gain_mod_error,
        latent_grad=latent_grads,
    )


def run_inference(
    params: GraphParams,
    initial_state: GraphState,
    clamps: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    infer_steps: int,
    eta_infer: float = 0.1,
    use_parallel: bool = False,
) -> GraphState:
    """
    Run inference for infer_steps steps to converge latent states.

    Args:
        params: Model parameters
        initial_state: Initial graph state
        clamps: Dictionary of clamped values
        structure: Graph structure
        infer_steps: Number of inference steps
        eta_infer: Inference learning rate
        use_parallel: Whether to use parallel node updates

    Returns:
        Final converged graph state
    """
    inference_fn = inference_step_parallel if use_parallel else inference_step

    def body_fn(t, state):
        return inference_fn(params, state, clamps, structure, eta_infer)

    # Use lax.fori_loop for efficiency
    final_state = jax.lax.fori_loop(0, infer_steps, body_fn, initial_state)

    return final_state