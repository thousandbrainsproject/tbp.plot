# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


class ArgAdder(Protocol):
    """Protocol for argument-adder callables.

    Any object conforming to this protocol must be callable with a single
    `argparse.ArgumentParser` and may add arguments in place.
    """

    def __call__(self, parser) -> None: ...


@dataclass
class PlotSpec:
    """Specification of a registered plot.

    Attributes:
        name: Unique name of the plot, used in the cli command.
        fn: Callable entry point for the plot. It should accept keyword
            arguments matching the parser's args and return an exit code.
        add_arguments: Optional callable that adds plot-specific arguments
            to an `argparse.ArgumentParser`.
        description: Description of the plot.
    """

    name: str
    fn: Callable[..., int]
    add_arguments: ArgAdder | None
    description: str | None = None


_REGISTRY: dict[str, PlotSpec] = {}


def register(name: str, description: str | None = None):
    """Decorator to register a plot function.

    Args:
        name: Unique name for the plot.
        description: Optional description for the plot.

    Returns:
        A decorator that registers a new `PlotSpec` in the registry.
    """

    def _wrap(fn: Callable[..., int]):
        _REGISTRY[name] = PlotSpec(
            name=name, fn=fn, add_arguments=None, description=description
        )
        return fn

    return _wrap


def attach_args(name: str):
    """Decorator to attach an `add_arguments` function to a registered plot.

    This allows each plot module to define a function that adds
    plot-specific CLI arguments.

    Args:
        name: The name of the plot to which the arguments function belongs.
              Must already be registered in the registry.

    Returns:
        A decorator that registers the given function as the
        ``add_arguments`` handler for the specified plot.
    """

    def _wrap(adder: ArgAdder):
        """Attach the ``adder`` to the already-registered plot.

        Raises:
            RuntimeError: If the plot ``name`` has not been registered.

        Returns:
            ArgAdder: The same ``adder`` function, unchanged.
        """
        if name not in _REGISTRY:
            raise RuntimeError(
                f"attach_args called before plot '{name}' was registered"
            )

        spec = _REGISTRY[name]
        _REGISTRY[name] = PlotSpec(
            name=spec.name,
            fn=spec.fn,
            add_arguments=adder,
            description=spec.description,
        )
        return adder

    return _wrap


def get(name: str) -> PlotSpec:
    """Retrieve a registered plot by name.

    Args:
        name: Plot name to look up.

    Returns:
        PlotSpec: The specification for the requested plot.

    Raises:
        KeyError: If the plot name is not registered. The message suggests
        running ``tbp-plot`` to list available plots.
    """
    try:
        return _REGISTRY[name]
    except KeyError as e:
        raise KeyError(
            f"Unknown plot '{name}'. Use `tbp-plot` to list available plots."
        ) from e


def all_specs() -> list[PlotSpec]:
    """Return all registered plots sorted by name."""
    return sorted(_REGISTRY.values(), key=lambda s: s.name)
