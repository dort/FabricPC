"""
Training loop for JAX predictive coding networks.
"""

from typing import Dict, Tuple, Any
from functools import partial
import jax
import jax.numpy as jnp
import optax

from fabricpc_jax.core.types import GraphParams, GraphState, GraphStructure
from fabricpc_jax.core.inference import run_inference
from fabricpc_jax.models.graph_net import initialize_state
from fabricpc_jax.core.initialization import get_default_state_init


def train_step(
    params: GraphParams,
    opt_state: optax.OptState,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    optimizer: optax.GradientTransformation,
    T_infer: int,
    eta_infer: float = 0.1,
    state_init_config: Dict[str, Any] = None,
) -> Tuple[GraphParams, optax.OptState, float, GraphState]:
    """
    Single training step: inference + weight update.

    Note: This function should be JIT-compiled at the call site for best performance.

    Args:
        params: Current model parameters
        opt_state: Optimizer state
        batch: Batch of data with 'x' and 'y' keys (or task-specific keys)
        structure: Graph structure
        optimizer: Optax optimizer
        T_infer: Number of inference steps
        eta_infer: Inference learning rate
        state_init_config: State initialization config (uses default if None)

    Returns:
        Tuple of (updated_params, updated_opt_state, loss, final_state)
    """

    def loss_fn(params: GraphParams) -> Tuple[float, GraphState]:
        """
        Compute energy (loss) for the current batch.

        The energy is the sum of squared prediction errors after inference.
        """
        batch_size = next(iter(batch.values())).shape[0]

        # Map task names to node names using task_map
        clamps = {}
        for task_name, task_value in batch.items():
            if task_name in structure.task_map:
                node_name = structure.task_map[task_name]
                clamps[node_name] = task_value

        # Initialize state
        # Use provided config or default
        init_config = state_init_config if state_init_config is not None else get_default_state_init()
        state = initialize_state(
            structure, batch_size, clamps=clamps, state_init_config=init_config, params=params
        )

        # Run inference to convergence
        final_state = run_inference(
            params, state, clamps, structure, T_infer, eta_infer
        )

        # Compute energy (sum of squared errors)
        # Only consider errors at non-source nodes with predictions
        energy = 0.0
        for node_name, node_info in structure.nodes.items():
            if node_info.in_degree > 0:  # Skip source nodes
                energy += jnp.sum(final_state.error[node_name] ** 2)

        # Average over batch
        energy = energy / batch_size

        return energy, final_state

    # Compute loss and gradients
    (loss, final_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    # Update parameters
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss, final_state


def train_pcn(
    params: GraphParams,
    structure: GraphStructure,
    train_loader: Any,
    config: dict,
    verbose: bool = True,
) -> GraphParams:
    """
    Main training loop for predictive coding network.

    Args:
        params: Initial parameters
        structure: Graph structure
        train_loader: Data loader (iterable yielding batches)
        config: Training configuration with keys:
            - optimizer: optimizer config dict
            - num_epochs: Number of training epochs
            - T_infer: number of inference steps
            - eta_infer: inference learning rate
            - state_initialization: (optional) state init config (uses default if not provided)
        verbose: Whether to print progress

    Returns:
        Trained parameters

    Example:
        >>> params, structure = create_pc_graph(model_config, jax.random.PRNGKey(0))
        >>> train_config = {
        ...     "optimizer": {"type": "adam", "lr": 1e-3},
        ...     "T_infer": 20,
        ...     "eta_infer": 0.1,
        ...     "state_initialization": {"type": "feedforward", "fallback": {"type": "normal", "std": 0.01}}
        ... }
        >>> trained_params = train_pcn(params, structure, train_loader, train_config)
    """
    from fabricpc_jax.training.optimizers import create_optimizer

    # Create optimizer
    optimizer = create_optimizer(config.get("optimizer", {"type": "adam", "lr": 1e-3}))
    opt_state = optimizer.init(params)

    # Training hyperparameters
    T_infer = config.get("T_infer", 20)
    eta_infer = config.get("eta_infer", 0.1)
    state_init_config = config.get("state_initialization", None)

    # Create JIT-compiled training step
    # We compile it once per training run to avoid recompilation
    jit_train_step = jax.jit(
        lambda p, o, b: train_step(p, o, b, structure, optimizer, T_infer, eta_infer, state_init_config)
    )

    # Training loop
    for epoch in range(config["num_epochs"]):
        epoch_losses = []

        for batch_idx, batch_data in enumerate(train_loader):
            # Convert batch to JAX format
            # Handle different data loader formats
            if isinstance(batch_data, (list, tuple)):
                # Assume (x, y) format
                batch = {
                    "x": jnp.array(batch_data[0]),
                    "y": jnp.array(batch_data[1]),
                }
            elif isinstance(batch_data, dict):
                # Already a dictionary
                batch = {k: jnp.array(v) for k, v in batch_data.items()}
            else:
                raise ValueError(f"Unsupported batch format: {type(batch_data)}")

            # Training step
            params, opt_state, loss, _ = jit_train_step(params, opt_state, batch)

            epoch_losses.append(float(loss))

        # Compute average loss for epoch
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0

        if verbose:
            print(f"Epoch {epoch + 1}/{config["num_epochs"]}, Loss: {avg_loss:.4f}")

    return params


def evaluate_pcn(
    params: GraphParams,
    structure: GraphStructure,
    test_loader: Any,
    config: dict,
) -> Dict[str, float]:
    """
    Evaluate predictive coding network on test data.

    Args:
        params: Trained parameters
        structure: Graph structure
        test_loader: Test data loader
        config: Evaluation configuration with keys:
            - T_infer: number of inference steps
            - eta_infer: inference learning rate
            - state_initialization: (optional) state init config (uses default if not provided)

    Returns:
        Dictionary of evaluation metrics (e.g., accuracy, loss)
    """
    T_infer = config.get("T_infer", 20)
    eta_infer = config.get("eta_infer", 0.1)
    state_init_config = config.get("state_initialization", None)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_data in test_loader:
        # Convert batch
        if isinstance(batch_data, (list, tuple)):
            batch = {"x": jnp.array(batch_data[0]), "y": jnp.array(batch_data[1])}
        elif isinstance(batch_data, dict):
            batch = {k: jnp.array(v) for k, v in batch_data.items()}
        else:
            raise ValueError(f"Unsupported batch format: {type(batch_data)}")

        batch_size = next(iter(batch.values())).shape[0]

        # Map batch to clamps
        clamps = {}
        for task_name, task_value in batch.items():
            if task_name in structure.task_map:
                node_name = structure.task_map[task_name]
                if task_name == "x":  # Only clamp input during eval
                    clamps[node_name] = task_value

        # Initialize and run inference
        # Use provided config or default
        init_config = state_init_config if state_init_config is not None else get_default_state_init()
        state = initialize_state(
            structure, batch_size, clamps=clamps, state_init_config=init_config, params=params
        )
        final_state = run_inference(params, state, clamps, structure, T_infer, eta_infer)

        # Compute loss
        energy = 0.0
        for node_name, node_info in structure.nodes.items():
            if node_info.in_degree > 0:
                energy += jnp.sum(final_state.error[node_name] ** 2)
        energy = energy / batch_size
        total_loss += float(energy) * batch_size

        # Compute accuracy (if applicable)
        if "y" in structure.task_map:
            y_node = structure.task_map["y"]
            predictions = final_state.z_latent[y_node]
            targets = batch["y"]

            # Assume classification: argmax predictions
            pred_labels = jnp.argmax(predictions, axis=1)
            true_labels = jnp.argmax(targets, axis=1)
            correct = jnp.sum(pred_labels == true_labels)

            total_correct += int(correct)
            total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    return {"loss": avg_loss, "accuracy": accuracy}
