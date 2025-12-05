"""
Node type registry with decorator-based registration.

This module provides a plugin architecture for custom node types:
- @register_node decorator for registering node classes
- Config schema validation for custom node parameters
- Entry point discovery for external packages
"""

from typing import Type, Dict, Any, List
import sys
import warnings

from fabricpc.nodes.base import NodeBase


# Global registry
_NODE_REGISTRY: Dict[str, Type[NodeBase]] = {}


class NodeRegistrationError(Exception):
    """Raised when node registration fails."""
    pass


def _validate_node_class(node_class: Type[NodeBase], node_type: str) -> None:
    """
    Validate that a node class implements the required interface.

    Args:
        node_class: The node class to validate
        node_type: The type name being registered (for error messages)

    Raises:
        NodeRegistrationError: If required methods/attributes are missing or abstract
    """
    # Check for required CONFIG_SCHEMA attribute
    if not hasattr(node_class, 'CONFIG_SCHEMA'):
        raise NodeRegistrationError(
            f"Node type '{node_type}': missing required CONFIG_SCHEMA attribute. "
            f"Use empty dict {{}} if no additional config parameters are needed."
        )

    # Validate CONFIG_SCHEMA is a dict
    if not isinstance(node_class.CONFIG_SCHEMA, dict):
        raise NodeRegistrationError(
            f"Node type '{node_type}': CONFIG_SCHEMA must be a dict, "
            f"got {type(node_class.CONFIG_SCHEMA).__name__}"
        )

    # Check for required DEFAULT_ENERGY_CONFIG attribute
    if not hasattr(node_class, 'DEFAULT_ENERGY_CONFIG'):
        raise NodeRegistrationError(
            f"Node type '{node_type}': missing required DEFAULT_ENERGY_CONFIG attribute. "
            f"Use {{'type': 'gaussian'}} for standard MSE energy."
        )

    # Validate DEFAULT_ENERGY_CONFIG is a dict with "type" key
    if not isinstance(node_class.DEFAULT_ENERGY_CONFIG, dict):
        raise NodeRegistrationError(
            f"Node type '{node_type}': DEFAULT_ENERGY_CONFIG must be a dict, "
            f"got {type(node_class.DEFAULT_ENERGY_CONFIG).__name__}"
        )
    if "type" not in node_class.DEFAULT_ENERGY_CONFIG:
        raise NodeRegistrationError(
            f"Node type '{node_type}': DEFAULT_ENERGY_CONFIG must have a 'type' key"
        )

    # Check for required methods
    required_methods = ['get_slots', 'initialize_params', 'forward']

    for method_name in required_methods:
        method = getattr(node_class, method_name, None)
        if method is None:
            raise NodeRegistrationError(
                f"Node type '{node_type}': missing required method '{method_name}'"
            )
        # Check it's not still abstract
        if getattr(method, '__isabstractmethod__', False):
            raise NodeRegistrationError(
                f"Node type '{node_type}': method '{method_name}' is abstract"
            )


def register_node(node_type: str):
    """
    Decorator to register a node class with the registry.

    Usage:
        @register_node("conv2d")
        class Conv2DNode(NodeBase):
            ...

    Args:
        node_type: Unique identifier for this node type (case-insensitive)

    Returns:
        Decorator function

    Raises:
        NodeRegistrationError: If registration fails (duplicate, missing methods)
    """
    def decorator(node_class: Type[NodeBase]) -> Type[NodeBase]:
        type_lower = node_type.lower()

        # Check for duplicate registration
        if type_lower in _NODE_REGISTRY:
            existing = _NODE_REGISTRY[type_lower]
            if existing is not node_class:
                raise NodeRegistrationError(
                    f"Node type '{node_type}' already registered by {existing.__name__}"
                )
            # Same class registered twice - silently allow (import idempotency)
            return node_class

        # Validate interface
        _validate_node_class(node_class, node_type)

        # Register
        _NODE_REGISTRY[type_lower] = node_class

        # Store the registered name on the class for introspection
        node_class._registered_type = type_lower

        return node_class

    return decorator


def get_node_class(node_type: str) -> Type[NodeBase]:
    """
    Get a node class by its registered type name.

    Args:
        node_type: The registered node type (case-insensitive)

    Returns:
        The node class

    Raises:
        ValueError: If node type is not registered
    """
    type_lower = node_type.lower()
    if type_lower not in _NODE_REGISTRY:
        available = list(_NODE_REGISTRY.keys())
        raise ValueError(
            f"Unknown node type '{node_type}'. "
            f"Available types: {available}"
        )
    return _NODE_REGISTRY[type_lower]


def list_node_types() -> List[str]:
    """Return list of all registered node types."""
    return sorted(_NODE_REGISTRY.keys())


def unregister_node(node_type: str) -> None:
    """
    Remove a node type from the registry.
    Primarily for testing purposes.

    Args:
        node_type: The node type to unregister (case-insensitive)
    """
    type_lower = node_type.lower()
    if type_lower in _NODE_REGISTRY:
        del _NODE_REGISTRY[type_lower]


def clear_registry() -> None:
    """Clear all registrations. For testing only."""
    _NODE_REGISTRY.clear()


def validate_node_config(node_class: Type[NodeBase], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and apply defaults from node's CONFIG_SCHEMA.

    Nodes can define a CONFIG_SCHEMA class attribute to specify:
    - Required fields
    - Type validation
    - Choice validation
    - Default values

    Example CONFIG_SCHEMA:
        CONFIG_SCHEMA = {
            "kernel_size": {"type": tuple, "required": True, "description": "Kernel dims"},
            "stride": {"type": tuple, "default": (1, 1)},
            "padding": {"type": str, "default": "valid", "choices": ["valid", "same"]},
        }

    Args:
        node_class: The node class with CONFIG_SCHEMA
        config: The user-provided config dict

    Returns:
        Config dict with defaults applied

    Raises:
        ValueError: If required fields are missing or validation fails
    """
    schema = getattr(node_class, 'CONFIG_SCHEMA', None)
    if schema is None:
        return config  # No schema, pass through as-is

    result = dict(config)

    for field_name, field_spec in schema.items():
        if field_name in result:
            # Validate type if specified
            expected_type = field_spec.get('type')
            if expected_type and not isinstance(result[field_name], expected_type):
                raise ValueError(
                    f"Config '{field_name}' must be {expected_type.__name__}, "
                    f"got {type(result[field_name]).__name__}"
                )
            # Validate choices if specified
            choices = field_spec.get('choices')
            if choices and result[field_name] not in choices:
                raise ValueError(
                    f"Config '{field_name}' must be one of {choices}, "
                    f"got '{result[field_name]}'"
                )
        elif field_spec.get('required', False):
            raise ValueError(
                f"Required config '{field_name}' missing. "
                f"Description: {field_spec.get('description', 'no description')}"
            )
        elif 'default' in field_spec:
            result[field_name] = field_spec['default']

    return result


def discover_external_nodes() -> None:
    """
    Discover and register nodes from installed packages via entry points.

    Looks for packages with entry points in the "fabricpc.nodes" group.
    Each entry point should map a node type name to a NodeBase subclass.

    Example pyproject.toml for an external package:
        [project.entry-points."fabricpc.nodes"]
        conv2d = "my_package.nodes:Conv2DNode"
    """
    try:
        if sys.version_info >= (3, 10):
            from importlib.metadata import entry_points
            eps = entry_points(group="fabricpc.nodes")
        else:
            from importlib.metadata import entry_points
            all_eps = entry_points()
            eps = all_eps.get("fabricpc.nodes", [])

        for ep in eps:
            try:
                node_class = ep.load()
                type_name = ep.name.lower()

                # Skip if already registered (built-in takes precedence)
                if type_name in _NODE_REGISTRY:
                    continue

                # Validate and register
                _validate_node_class(node_class, type_name)
                _NODE_REGISTRY[type_name] = node_class
                node_class._registered_type = type_name

            except Exception as e:
                warnings.warn(
                    f"Failed to load node '{ep.name}' from {ep.value}: {e}",
                    RuntimeWarning
                )
    except Exception as e:
        # Entry point discovery failed entirely - not critical
        warnings.warn(
            f"Entry point discovery failed: {e}",
            RuntimeWarning
        )