"""
Test suite for LinearAutoGradNode gradient computation.

Verifies that LinearAutoGradNode (using JAX autodiff) produces
numerically equivalent gradients to LinearNode (using manual formulas).
"""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")

import pytest
import jax
import jax.numpy as jnp

from fabricpc.core.types import NodeState, NodeParams
from fabricpc.graph.graph_net import create_pc_graph, initialize_state
from fabricpc.core.inference import compute_all_projections, compute_errors
from fabricpc.nodes import get_node_class_from_type, LinearNode, LinearAutoGradNode

jax.config.update("jax_platform_name", "cpu")


@pytest.fixture
def rng_key():
    """Fixture to provide a JAX random key."""
    return jax.random.PRNGKey(42)


def create_config(node_type: str):
    """Create a small network config with specified node type."""
    return {
        "node_list": [
            {
                "name": "input",
                "dim": 8,
                "type": node_type,
                "activation": {"type": "identity"},
            },
            {
                "name": "hidden",
                "dim": 12,
                "type": node_type,
                "activation": {"type": "tanh"},
            },
            {
                "name": "output",
                "dim": 4,
                "type": node_type,
                "activation": {"type": "sigmoid"},
            },
        ],
        "edge_list": [
            {"source_name": "input", "target_name": "hidden", "slot": "in"},
            {"source_name": "hidden", "target_name": "output", "slot": "in"},
        ],
        "task_map": {"x": "input", "y": "output"},
    }


class TestLinearAutoGradNode:
    """Test that LinearAutoGradNode produces identical gradients to LinearNode."""

    @pytest.mark.parametrize("activation", ["identity", "relu", "tanh", "sigmoid"])
    def test_gradient_equivalence_single_node(self, rng_key, activation):
        """Test gradient equivalence for different activation functions."""
        batch_size = 4
        input_dim = 6
        output_dim = 8

        rngkey_weights, rngkey_inputs, rngkey_latent = jax.random.split(rng_key, 3)

        edge_key = "src->dst:in"
        params = NodeParams(
            weights={edge_key: jax.random.normal(rngkey_weights, (input_dim, output_dim)) * 0.1},
            biases={"b": jnp.zeros((1, output_dim))}
        )
        inputs = {edge_key: jax.random.normal(rngkey_inputs, (batch_size, input_dim))}

        from fabricpc.core.types import NodeInfo, GraphStructure, EdgeInfo
        node_info = NodeInfo(
            name="dst",
            dim=output_dim,
            node_type="linear",
            activation_config={"type": activation},
            node_config={},
            in_degree=1,
            out_degree=0,
            slots={},
            in_edges=frozenset([edge_key]),
            out_edges=frozenset(),
        )
        source_info = NodeInfo(
            name="src",
            dim=input_dim,
            node_type="linear",
            activation_config={"type": "identity"},
            node_config={},
            in_degree=0,
            out_degree=1,
            slots={},
            in_edges=frozenset(),
            out_edges=frozenset([edge_key]),
        )
        structure = GraphStructure(
            nodes={"src": source_info, "dst": node_info},
            edges={edge_key: EdgeInfo(source="src", target="dst", slot="in", key=edge_key)},
            node_order=["src", "dst"],
            task_map={"x": "src"},
        )

        # Forward pass
        node_out_shape = (batch_size, output_dim)
        z_mu, pre_activation, _ = LinearNode.forward(params, inputs, node_info, node_out_shape)

        # Create initial state with random latent
        z_latent = jax.random.normal(rngkey_latent, (batch_size, output_dim))
        node_state = NodeState(
            z_latent=z_latent,
            latent_grad=jnp.zeros((batch_size, output_dim)),
            z_mu=z_mu,
            error=jnp.zeros((batch_size, output_dim)),
            energy=jnp.zeros(()),
            pre_activation=pre_activation,
            gain_mod_error=jnp.zeros((batch_size, output_dim)),
            substructure={},
        )

        # Use node class methods to compute error and energy
        node_state = LinearNode.compute_error(node_state, node_info)
        node_state = LinearNode.energy_functional(node_state, node_info)

        # Compare gradients
        grads_linear = LinearNode.compute_gradient(params, inputs, node_state, node_info, structure)
        grads_autograd = LinearAutoGradNode.compute_gradient(params, inputs, node_state, node_info, structure)

        for node_name in grads_linear:
            max_diff = jnp.max(jnp.abs(grads_linear[node_name] - grads_autograd[node_name]))
            assert max_diff < 1e-5, \
                f"Gradient mismatch for activation={activation}, node={node_name}: max diff = {max_diff}"

    def test_gradient_equivalence_full_network(self, rng_key):
        """Test gradient equivalence across a full network with inference."""
        batch_size = 8

        # Create two identical networks with different node types
        config_linear = create_config("linear")
        config_autograd = create_config("linear_autograd")

        # Use same key for identical initialization
        params_linear, structure_linear = create_pc_graph(config_linear, rng_key)
        params_autograd, structure_autograd = create_pc_graph(config_autograd, rng_key)

        # Verify params are identical
        for node_name in params_linear.nodes:
            for edge_key in params_linear.nodes[node_name].weights:
                w_linear = params_linear.nodes[node_name].weights[edge_key]
                w_autograd = params_autograd.nodes[node_name].weights[edge_key]
                assert jnp.allclose(w_linear, w_autograd), \
                    f"Params differ for {node_name}/{edge_key}"

        # Create identical input/output data
        rngkey_x, rngkey_y, rngkey_state = jax.random.split(rng_key, 3)
        x_data = jax.random.normal(rngkey_x, (batch_size, 8))
        y_data = jax.random.normal(rngkey_y, (batch_size, 4))
        clamps = {"input": x_data, "output": y_data}

        # Initialize states identically
        state_linear = initialize_state(
            structure_linear, batch_size, rngkey_state, clamps=clamps, params=params_linear
        )
        state_autograd = initialize_state(
            structure_autograd, batch_size, rngkey_state, clamps=clamps, params=params_autograd
        )

        # Compute projections
        state_linear = compute_all_projections(params_linear, state_linear, structure_linear)
        state_autograd = compute_all_projections(params_autograd, state_autograd, structure_autograd)

        # Compute errors
        state_linear = compute_errors(state_linear, structure_linear)
        state_autograd = compute_errors(state_autograd, structure_autograd)

        # Compare gradients for each non-input node
        for node_name in ["hidden", "output"]:
            node_info = structure_linear.nodes[node_name]
            node_state_linear = state_linear.nodes[node_name]
            node_state_autograd = state_autograd.nodes[node_name]

            # Gather inputs for gradient computation
            inputs = {}
            for edge_key in node_info.in_edges:
                source_name = structure_linear.edges[edge_key].source
                inputs[edge_key] = state_linear.nodes[source_name].z_latent

            # Compute gradients
            grads_linear = LinearNode.compute_gradient(
                params_linear.nodes[node_name],
                inputs,
                node_state_linear,
                node_info,
                structure_linear
            )
            grads_autograd = LinearAutoGradNode.compute_gradient(
                params_autograd.nodes[node_name],
                inputs,
                node_state_autograd,
                node_info,
                structure_autograd
            )

            # Compare
            for grad_node in grads_linear:
                max_diff = jnp.max(jnp.abs(grads_linear[grad_node] - grads_autograd[grad_node]))
                assert max_diff < 1e-5, \
                    f"Gradient mismatch at {node_name} for {grad_node}: max diff = {max_diff}"

class TestLinearAutoGradNodeRegistration:
    """Test that LinearAutoGradNode is properly registered."""

    def test_node_type_registered(self):
        """Test that linear_autograd node type is registered."""
        node_class = get_node_class_from_type("linear_autograd")
        assert node_class is LinearAutoGradNode

    def test_network_creation_with_autograd_nodes(self, rng_key):
        """Test that a network can be created using linear_autograd nodes."""
        config = create_config("linear_autograd")
        params, structure = create_pc_graph(config, rng_key)

        assert len(structure.nodes) == 3
        assert all(info.node_type == "linear_autograd" for info in structure.nodes.values())