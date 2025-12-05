"""
Energy functionals for predictive coding networks.

This module provides:
- EnergyFunctional base class with abstract interface
- Built-in energy functionals (Gaussian, Bernoulli, Cross Entrpoy)
- Registry with decorator-based registration for custom energy functionals
- Entry point discovery for external packages

Energy functionals define how prediction errors are quantified into scalar energy
values, which drives both inference (latent state updates) and learning (weight updates).

User Extensibility
------------------
Users can register custom energy functionals in two ways:

1. **Decorator-based registration** (recommended for development):

    @register_energy("my_energy")
    class MyEnergyFunctional(EnergyFunctional):
        @staticmethod
        def energy(z_latent, z_mu, config=None):
            ...
        @staticmethod
        def grad_latent(z_latent, z_mu, config=None):
            ...

2. **Entry point discovery** (recommended for distribution):

    Add to pyproject.toml:
        [project.entry-points."fabricpc.energy"]
        my_energy = "my_package.energy:MyEnergyFunctional"

Configuration
-------------
Energy functionals can be configured per-node via node_config:

    {
        "name": "output",
        "shape": (10,),
        "type": "linear",
        "energy": {
            "type": "cross_entropy",  # Name of registered energy functional
            "temperature": 1.0      # Energy-specific parameters
        }
    }

If no energy config is specified, defaults to "gaussian".
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Type, List, Tuple
import sys
import warnings

import jax.numpy as jnp


# =============================================================================
# Energy Functional Base Class
# =============================================================================

class EnergyFunctional(ABC):
    """
    Abstract base class for energy functionals.

    Energy functionals define how prediction errors are converted to scalar
    energy values. The energy drives inference (minimizing E w.r.t. z_latent)
    and provides the loss signal for learning.

    All methods are static for JAX compatibility (pure functions, no state).

    Required methods:
        - energy(): Compute E(z_latent, z_mu) per sample
        - grad_latent(): Compute ∂E/∂z_latent

    Required attributes:
        - CONFIG_SCHEMA: dict specifying configuration validation

    Example implementation:
        @register_energy("my_energy")
        class MyEnergy(EnergyFunctional):
            CONFIG_SCHEMA = {
                "temperature": {"type": float, "default": 1.0}
            }

            @staticmethod
            def energy(z_latent, z_mu, config=None):
                temp = config.get("temperature", 1.0) if config else 1.0
                diff = z_latent - z_mu
                return 0.5 * jnp.sum(diff ** 2, axis=-1) / temp

            @staticmethod
            def grad_latent(z_latent, z_mu, config=None):
                temp = config.get("temperature", 1.0) if config else 1.0
                return (z_latent - z_mu) / temp
    """

    # CONFIG_SCHEMA is required - subclasses must define it
    # Use empty dict {} if no additional config parameters are needed
    CONFIG_SCHEMA: Dict[str, Dict[str, Any]]

    @staticmethod
    @abstractmethod
    def energy(
        z_latent: jnp.ndarray,
        z_mu: jnp.ndarray,
        config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """
        Compute energy E(z_latent, z_mu).

        Args:
            z_latent: Latent states, shape (batch, *dims)
            z_mu: Predicted expectations, shape (batch, *dims)
            config: Optional configuration dict for energy parameters

        Returns:
            Energy per sample, shape (batch,)

        Note:
            Should sum over all non-batch dimensions to produce per-sample energy.
        """
        pass

    @staticmethod
    @abstractmethod
    def grad_latent(
        z_latent: jnp.ndarray,
        z_mu: jnp.ndarray,
        config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """
        Compute gradient ∂E/∂z_latent.

        Args:
            z_latent: Latent states, shape (batch, *dims)
            z_mu: Predicted expectations, shape (batch, *dims)
            config: Optional configuration dict for energy parameters

        Returns:
            Gradient w.r.t. z_latent, same shape as z_latent

        Note:
            This is the signal used to update latent states during inference:
            z_latent_new = z_latent - eta * grad_latent(z_latent, z_mu)
        """
        pass


# =============================================================================
# Energy Registry
# =============================================================================

_ENERGY_REGISTRY: Dict[str, Type[EnergyFunctional]] = {}


class EnergyRegistrationError(Exception):
    """Raised when energy functional registration fails."""
    pass


def _validate_energy_class(energy_class: Type[EnergyFunctional], energy_type: str) -> None:
    """
    Validate that an energy class implements the required interface.

    Args:
        energy_class: The energy class to validate
        energy_type: The type name being registered (for error messages)

    Raises:
        EnergyRegistrationError: If required methods/attributes are missing or abstract
    """
    # Check for required CONFIG_SCHEMA attribute
    if not hasattr(energy_class, 'CONFIG_SCHEMA'):
        raise EnergyRegistrationError(
            f"Energy type '{energy_type}': missing required CONFIG_SCHEMA attribute. "
            f"Use empty dict {{}} if no additional config parameters are needed."
        )

    # Validate CONFIG_SCHEMA is a dict
    if not isinstance(energy_class.CONFIG_SCHEMA, dict):
        raise EnergyRegistrationError(
            f"Energy type '{energy_type}': CONFIG_SCHEMA must be a dict, "
            f"got {type(energy_class.CONFIG_SCHEMA).__name__}"
        )

    # Check for required methods
    required_methods = ['energy', 'grad_latent']

    for method_name in required_methods:
        method = getattr(energy_class, method_name, None)
        if method is None:
            raise EnergyRegistrationError(
                f"Energy type '{energy_type}': missing required method '{method_name}'"
            )
        # Check it's not still abstract
        if getattr(method, '__isabstractmethod__', False):
            raise EnergyRegistrationError(
                f"Energy type '{energy_type}': method '{method_name}' is abstract"
            )


def register_energy(energy_type: str):
    """
    Decorator to register an energy functional with the registry.

    Usage:
        @register_energy("bernoulli")
        class BernoulliEnergy(EnergyFunctional):
            ...

    Args:
        energy_type: Unique identifier for this energy type (case-insensitive)

    Returns:
        Decorator function

    Raises:
        EnergyRegistrationError: If registration fails (duplicate, missing methods)
    """
    def decorator(energy_class: Type[EnergyFunctional]) -> Type[EnergyFunctional]:
        type_lower = energy_type.lower()

        # Check for duplicate registration
        if type_lower in _ENERGY_REGISTRY:
            existing = _ENERGY_REGISTRY[type_lower]
            if existing is not energy_class:
                raise EnergyRegistrationError(
                    f"Energy type '{energy_type}' already registered by {existing.__name__}"
                )
            # Same class registered twice - silently allow (import idempotency)
            return energy_class

        # Validate interface
        _validate_energy_class(energy_class, energy_type)

        # Register
        _ENERGY_REGISTRY[type_lower] = energy_class

        # Store the registered name on the class for introspection
        energy_class._registered_type = type_lower

        return energy_class

    return decorator


def get_energy_class(energy_type: str) -> Type[EnergyFunctional]:
    """
    Get an energy functional class by its registered type name.

    Args:
        energy_type: The registered energy type (case-insensitive)

    Returns:
        The energy functional class

    Raises:
        ValueError: If energy type is not registered
    """
    type_lower = energy_type.lower()
    if type_lower not in _ENERGY_REGISTRY:
        available = list(_ENERGY_REGISTRY.keys())
        raise ValueError(
            f"Unknown energy type '{energy_type}'. "
            f"Available types: {available}"
        )
    return _ENERGY_REGISTRY[type_lower]


def list_energy_types() -> List[str]:
    """Return list of all registered energy types."""
    return sorted(_ENERGY_REGISTRY.keys())


def unregister_energy(energy_type: str) -> None:
    """
    Remove an energy type from the registry.
    Primarily for testing purposes.

    Args:
        energy_type: The energy type to unregister (case-insensitive)
    """
    type_lower = energy_type.lower()
    if type_lower in _ENERGY_REGISTRY:
        del _ENERGY_REGISTRY[type_lower]


def clear_energy_registry() -> None:
    """Clear all registrations. For testing only."""
    _ENERGY_REGISTRY.clear()


def validate_energy_config(
    energy_class: Type[EnergyFunctional],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate and apply defaults from energy's CONFIG_SCHEMA.

    Energy functionals can define a CONFIG_SCHEMA class attribute to specify:
    - Required fields
    - Type validation
    - Choice validation
    - Default values

    Example CONFIG_SCHEMA:
        CONFIG_SCHEMA = {
            "temperature": {"type": float, "default": 1.0},
            "reduction": {"type": str, "default": "sum", "choices": ["sum", "mean"]},
        }

    Args:
        energy_class: The energy class with CONFIG_SCHEMA
        config: The user-provided config dict

    Returns:
        Config dict with defaults applied

    Raises:
        ValueError: If required fields are missing or validation fails
    """
    schema = getattr(energy_class, 'CONFIG_SCHEMA', None)
    if schema is None:
        return config or {}

    result = dict(config) if config else {}

    for field_name, field_spec in schema.items():
        if field_name in result:
            # Validate type if specified
            expected_type = field_spec.get('type')
            if expected_type and not isinstance(result[field_name], expected_type):
                # Handle tuple of types for error message
                if isinstance(expected_type, tuple):
                    type_names = " or ".join(t.__name__ for t in expected_type)
                else:
                    type_names = expected_type.__name__
                raise ValueError(
                    f"Energy config '{field_name}' must be {type_names}, "
                    f"got {type(result[field_name]).__name__}"
                )
            # Validate choices if specified
            choices = field_spec.get('choices')
            if choices and result[field_name] not in choices:
                raise ValueError(
                    f"Energy config '{field_name}' must be one of {choices}, "
                    f"got '{result[field_name]}'"
                )
        elif field_spec.get('required', False):
            raise ValueError(
                f"Required energy config '{field_name}' missing. "
                f"Description: {field_spec.get('description', 'no description')}"
            )
        elif 'default' in field_spec:
            result[field_name] = field_spec['default']

    return result


def discover_external_energy() -> None:
    """
    Discover and register energy functionals from installed packages via entry points.

    Looks for packages with entry points in the "fabricpc.energy" group.
    Each entry point should map an energy type name to an EnergyFunctional subclass.

    Example pyproject.toml for an external package:
        [project.entry-points."fabricpc.energy"]
        poisson = "my_package.energy:PoissonEnergy"
    """
    try:
        if sys.version_info >= (3, 10):
            from importlib.metadata import entry_points
            eps = entry_points(group="fabricpc.energy")
        else:
            from importlib.metadata import entry_points
            all_eps = entry_points()
            eps = all_eps.get("fabricpc.energy", [])

        for ep in eps:
            try:
                energy_class = ep.load()
                type_name = ep.name.lower()

                # Skip if already registered (built-in takes precedence)
                if type_name in _ENERGY_REGISTRY:
                    continue

                # Validate and register
                _validate_energy_class(energy_class, type_name)
                _ENERGY_REGISTRY[type_name] = energy_class
                energy_class._registered_type = type_name

            except Exception as e:
                warnings.warn(
                    f"Failed to load energy '{ep.name}' from {ep.value}: {e}",
                    RuntimeWarning
                )
    except Exception as e:
        # Entry point discovery failed entirely - not critical
        warnings.warn(
            f"Energy entry point discovery failed: {e}",
            RuntimeWarning
        )


# =============================================================================
# Built-in Energy Functionals
# =============================================================================

@register_energy("gaussian")
class GaussianEnergy(EnergyFunctional):
    """
    Gaussian (quadratic) energy functional.

    E = (1/2σ²) * ||z - μ||²

    This is the standard MSE-based energy, equivalent to assuming Gaussian
    distributions for predictions with fixed variance.

    Config options:
        - precision: 1/σ² (default: 1.0). Higher values = sharper distributions.

    This is the DEFAULT energy functional if none is specified.
    """

    CONFIG_SCHEMA = {
        "precision": {
            "type": (int, float),
            "default": 1.0,
            "description": "Precision (1/variance) of the Gaussian. Higher = tighter fit."
        }
    }

    @staticmethod
    def energy(
        z_latent: jnp.ndarray,
        z_mu: jnp.ndarray,
        config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """
        Compute Gaussian energy: E = (precision/2) * ||z - μ||²

        Sums over all non-batch dimensions.
        """
        precision = config.get("precision", 1.0) if config else 1.0
        diff = z_latent - z_mu
        # Sum over all non-batch dimensions
        axes_to_sum = tuple(range(1, len(diff.shape)))
        return 0.5 * precision * jnp.sum(diff ** 2, axis=axes_to_sum)

    @staticmethod
    def grad_latent(
        z_latent: jnp.ndarray,
        z_mu: jnp.ndarray,
        config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """
        Compute gradient: ∂E/∂z = precision * (z - μ)
        """
        precision = config.get("precision", 1.0) if config else 1.0
        return precision * (z_latent - z_mu)


@register_energy("bernoulli")
class BernoulliEnergy(EnergyFunctional):
    """
    Bernoulli (binary cross-entropy) energy functional.

    E = -Σ[z*log(μ) + (1-z)*log(1-μ)]

    Use for binary outputs where μ represents probabilities in [0, 1].
    The target z_latent should be clamped to binary values (0 or 1).

    Config options:
        - eps: Small constant for numerical stability (default: 1e-7)
    """

    CONFIG_SCHEMA = {
        "eps": {
            "type": (int, float),
            "default": 1e-7,
            "description": "Small constant for numerical stability in log"
        }
    }

    @staticmethod
    def energy(
        z_latent: jnp.ndarray,
        z_mu: jnp.ndarray,
        config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """
        Compute Bernoulli (BCE) energy: E = -Σ[z*log(μ) + (1-z)*log(1-μ)]
        """
        eps = config.get("eps", 1e-7) if config else 1e-7
        z_mu_safe = jnp.clip(z_mu, eps, 1 - eps)

        bce = -(z_latent * jnp.log(z_mu_safe) + (1 - z_latent) * jnp.log(1 - z_mu_safe))

        # Sum over all non-batch dimensions
        axes_to_sum = tuple(range(1, len(bce.shape)))
        return jnp.sum(bce, axis=axes_to_sum)

    @staticmethod
    def grad_latent(
        z_latent: jnp.ndarray,
        z_mu: jnp.ndarray,
        config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """
        Compute gradient: ∂E/∂z = -log(μ) + log(1-μ) = log((1-μ)/μ)

        Note: In standard PC with clamped targets, this gradient is used
        to propagate errors backward. For binary targets, the gradient
        pushes z toward z_mu.
        """
        eps = config.get("eps", 1e-7) if config else 1e-7
        z_mu_safe = jnp.clip(z_mu, eps, 1 - eps)

        # ∂BCE/∂z = -log(μ) + log(1-μ)
        return -jnp.log(z_mu_safe) + jnp.log(1 - z_mu_safe)


@register_energy("cross_entropy")
class CrossEntropyEnergy(EnergyFunctional):
    """
    Categorical (cross-entropy) energy functional.

    E = -Σ z_i * log(μ_i)

    Use for classification where:
    - z_latent is one-hot encoded targets
    - z_mu is softmax probabilities (should sum to 1 along last axis)

    Config options:
        - eps: Small constant for numerical stability (default: 1e-7)
        - axis: Axis along which probabilities sum to 1 (default: -1)
    """

    CONFIG_SCHEMA = {
        "eps": {
            "type": (int, float),
            "default": 1e-7,
            "description": "Small constant for numerical stability in log"
        },
        "axis": {
            "type": int,
            "default": -1,
            "description": "Axis along which probabilities sum to 1"
        }
    }

    @staticmethod
    def energy(
        z_latent: jnp.ndarray,
        z_mu: jnp.ndarray,
        config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """
        Compute cross_entropy (CE) energy: E = -Σ z_i * log(μ_i)
        """
        eps = config.get("eps", 1e-7) if config else 1e-7
        z_mu_safe = jnp.clip(z_mu, eps, 1.0)

        ce = -z_latent * jnp.log(z_mu_safe)

        # Sum over all non-batch dimensions
        axes_to_sum = tuple(range(1, len(ce.shape)))
        return jnp.sum(ce, axis=axes_to_sum)

    @staticmethod
    def grad_latent(
        z_latent: jnp.ndarray,
        z_mu: jnp.ndarray,
        config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """
        Compute gradient: ∂E/∂z = -log(μ)

        For one-hot targets with clamped latents, this gradient is used
        to propagate classification errors backward through the network.
        """
        eps = config.get("eps", 1e-7) if config else 1e-7
        z_mu_safe = jnp.clip(z_mu, eps, 1.0)

        return -jnp.log(z_mu_safe)


@register_energy("laplacian")
class LaplacianEnergy(EnergyFunctional):
    """
    Laplacian (L1) energy functional.

    E = (1/b) * Σ|z - μ|

    More robust to outliers than Gaussian. Corresponds to assuming
    Laplace distributions for predictions.

    Config options:
        - scale: b parameter (default: 1.0). Larger = more tolerance.
    """

    CONFIG_SCHEMA = {
        "scale": {
            "type": (int, float),
            "default": 1.0,
            "description": "Scale parameter b of Laplace distribution"
        }
    }

    @staticmethod
    def energy(
        z_latent: jnp.ndarray,
        z_mu: jnp.ndarray,
        config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """
        Compute Laplacian energy: E = (1/b) * Σ|z - μ|
        """
        scale = config.get("scale", 1.0) if config else 1.0
        diff = jnp.abs(z_latent - z_mu)

        axes_to_sum = tuple(range(1, len(diff.shape)))
        return jnp.sum(diff, axis=axes_to_sum) / scale

    @staticmethod
    def grad_latent(
        z_latent: jnp.ndarray,
        z_mu: jnp.ndarray,
        config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """
        Compute gradient: ∂E/∂z = (1/b) * sign(z - μ)
        """
        scale = config.get("scale", 1.0) if config else 1.0
        return jnp.sign(z_latent - z_mu) / scale


@register_energy("huber")
class HuberEnergy(EnergyFunctional):
    """
    Huber energy functional (smooth L1).

    E = {  0.5 * (z - μ)²           if |z - μ| ≤ δ
        {  δ * (|z - μ| - 0.5*δ)    if |z - μ| > δ

    Combines advantages of L2 (smooth gradients) and L1 (robustness).

    Config options:
        - delta: Transition threshold (default: 1.0)
    """

    CONFIG_SCHEMA = {
        "delta": {
            "type": (int, float),
            "default": 1.0,
            "description": "Threshold for switching from quadratic to linear"
        }
    }

    @staticmethod
    def energy(
        z_latent: jnp.ndarray,
        z_mu: jnp.ndarray,
        config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """
        Compute Huber energy.
        """
        delta = config.get("delta", 1.0) if config else 1.0
        diff = z_latent - z_mu
        abs_diff = jnp.abs(diff)

        # Quadratic region
        quadratic = 0.5 * diff ** 2
        # Linear region
        linear = delta * (abs_diff - 0.5 * delta)

        huber = jnp.where(abs_diff <= delta, quadratic, linear)

        axes_to_sum = tuple(range(1, len(huber.shape)))
        return jnp.sum(huber, axis=axes_to_sum)

    @staticmethod
    def grad_latent(
        z_latent: jnp.ndarray,
        z_mu: jnp.ndarray,
        config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """
        Compute gradient: clipped to [-δ, δ]
        """
        delta = config.get("delta", 1.0) if config else 1.0
        diff = z_latent - z_mu

        return jnp.clip(diff, -delta, delta)


# =============================================================================
# Convenience Functions
# =============================================================================

def compute_energy(
    z_latent: jnp.ndarray,
    z_mu: jnp.ndarray,
    energy_config: Dict[str, Any] = None
) -> jnp.ndarray:
    """
    Compute energy using the specified energy functional.

    Args:
        z_latent: Latent states, shape (batch, *dims)
        z_mu: Predicted expectations, shape (batch, *dims)
        energy_config: Energy configuration dict with "type" and other params.
                      If None, uses Gaussian energy with defaults.

    Returns:
        Energy per sample, shape (batch,)

    Example:
        energy = compute_energy(z, mu, {"type": "bernoulli", "eps": 1e-6})
    """
    if energy_config is None:
        energy_config = {"type": "gaussian"}

    energy_type = energy_config.get("type", "gaussian")
    energy_class = get_energy_class(energy_type)

    # Validate and apply defaults
    config = validate_energy_config(energy_class, energy_config)

    return energy_class.energy(z_latent, z_mu, config)


def compute_energy_gradient(
    z_latent: jnp.ndarray,
    z_mu: jnp.ndarray,
    energy_config: Dict[str, Any] = None
) -> jnp.ndarray:
    """
    Compute energy gradient w.r.t. z_latent.

    Args:
        z_latent: Latent states, shape (batch, *dims)
        z_mu: Predicted expectations, shape (batch, *dims)
        energy_config: Energy configuration dict with "type" and other params.
                      If None, uses Gaussian energy with defaults.

    Returns:
        Gradient ∂E/∂z_latent, same shape as z_latent

    Example:
        grad = compute_energy_gradient(z, mu, {"type": "cross_entropy"})
    """
    if energy_config is None:
        energy_config = {"type": "gaussian"}

    energy_type = energy_config.get("type", "gaussian")
    energy_class = get_energy_class(energy_type)

    # Validate and apply defaults
    config = validate_energy_config(energy_class, energy_config)

    return energy_class.grad_latent(z_latent, z_mu, config)


def get_energy_and_gradient(
    z_latent: jnp.ndarray,
    z_mu: jnp.ndarray,
    energy_config: Dict[str, Any] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute both energy and gradient efficiently.

    Args:
        z_latent: Latent states, shape (batch, *dims)
        z_mu: Predicted expectations, shape (batch, *dims)
        energy_config: Energy configuration dict

    Returns:
        Tuple of (energy, gradient):
            - energy: per-sample energy, shape (batch,)
            - gradient: ∂E/∂z_latent, same shape as z_latent
    """
    if energy_config is None:
        energy_config = {"type": "gaussian"}

    energy_type = energy_config.get("type", "gaussian")
    energy_class = get_energy_class(energy_type)
    config = validate_energy_config(energy_class, energy_config)

    energy = energy_class.energy(z_latent, z_mu, config)
    gradient = energy_class.grad_latent(z_latent, z_mu, config)

    return energy, gradient


# Auto-discover external energy functionals on import
discover_external_energy()