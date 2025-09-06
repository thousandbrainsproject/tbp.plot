# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Command-line interface for TBP plots.

This module discovers available plot modules, lists them, and provides
entry points for running specific plots with their plot-specific
arguments. It is defined as the `plot` project script in `pyproject.toml`.
"""

import argparse
import importlib
import pkgutil
import sys
from difflib import get_close_matches

from tbp.plot.registry import PlotSpec, all_specs, get


def _print_list() -> None:
    """Print all registered plots and their descriptions.

    Uses the global registry to display a left-aligned list of plot
    names. If the registry is empty, prints a message instead.
    """
    specs = all_specs()

    # Registry is empty
    if not specs:
        print("No plots registered.")
        return

    # ljust names for aesthetics.
    name_w = max(len(s.name) for s in specs)
    print("Available plots:\n")
    for s in specs:
        desc = f": {s.description}" if s.description else ""
        print(f"  {s.name.ljust(name_w)}{desc}")


def _build_plot_parser(spec: PlotSpec) -> argparse.ArgumentParser:
    """Create an ArgumentParser configured for a specific plot.

    Args:
        spec: The PlotSpec object describing the plot. If the spec has
            an `add_arguments` function, it is invoked to register
            custom CLI args.

    Returns:
        Parser configured with this plot's arguments.
    """
    p = argparse.ArgumentParser(
        prog=f"plot {spec.name}",
        description=f"Arguments for plot '{spec.name}'",
        add_help=True,
    )
    if spec.add_arguments:
        spec.add_arguments(p)

    return p


def main() -> int:
    """Entry point for the `plot` command-line tool.

    This function imports all plot modules to populate the registry,
    then runs specific plots based on user input:
        - With no args or `-h/--help`: show the list of available plots.
        - With `<plot> -h`: show help for the specified plot.
        - With `<plot> [args...]`: parse args and run the plot.
        - With an unknown plot: print suggestions (if any) and return exit code 2.

    Returns:
        Exit code from the invoked plot, or:
            * 0 if help/list was printed successfully,
            * 2 if an unknown plot name was requested.
    """
    # Arguments passed to `cli.py` excluding script name
    argv = sys.argv[1:]

    # Import all modules under `tbp.plot.plots` to register the plots
    pkg = importlib.import_module("tbp.plot.plots")
    for modinfo in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + "."):
        importlib.import_module(modinfo.name)

    # Display the help menu if requested
    if not argv or argv[0] in ("-h", "--help"):
        p = argparse.ArgumentParser(prog="plot", add_help=False)
        p.description = (
            "TBP plotting CLI. Run a plot by name, or call with no args to list."
        )
        _print_list()
        return 0

    plot_name, *plot_args = argv

    try:
        spec = get(plot_name)
    except KeyError:
        names = [s.name for s in all_specs()]
        matches = get_close_matches(plot_name, names, n=3, cutoff=0.5)
        print(f"Unknown plot '{plot_name}'.")
        if matches:
            print(f"Did you mean: {', '.join(matches)}?")
        print()
        _print_list()
        return 2

    plot_parser = _build_plot_parser(spec)

    # If user asked for help on a specific plot
    if plot_args and plot_args[0] in ("-h", "--help"):
        plot_parser.print_help()
        return 0

    opts = plot_parser.parse_args(plot_args)
    return spec.fn(**vars(opts))


if __name__ == "__main__":
    main()
