# tbp.plot

This is a visualization tool for plotting [tbp.monty](https://github.com/thousandbrainsproject/tbp.monty) visualizations.

- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)

## Installation

To use the tool, all you need to do is clone this repository. If you intend to do development on this repository, see the [Development](#development) section for the development setup.

### Install `uv`

On a Mac, `brew install uv` is sufficient. For additional options, see the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

### Install dependencies

```bash
uv sync
```

## Usage

To run a tool:

```bash
$ uv run interactive_object_evidence_over_time -h
usage: interactive_object_evidence_over_time [-h] [--debug] [--objects_mesh_dir OBJECTS_MESH_DIR] [-lm LEARNING_MODULE] experiment_log_dir

Creates an interactive plot of object evidence and sensor visualization.

positional arguments:
  experiment_log_dir    The directory containing the experiment log with the detailed stats file.

options:
  -h, --help            show this help message and exit
  --debug               Enable debug logging
  --objects_mesh_dir OBJECTS_MESH_DIR
                        The directory containing the mesh objects.
  -lm, --learning_module LEARNING_MODULE
                        The name of the learning module (default: "LM_0").
```

## Development

> [!NOTE]
> First, **make a fork of this repository**. Make any changes on your local fork. This repository is configured to only accept pull requests from forks.

The development of this project is managed with [uv](https://docs.astral.sh/uv/), "a single tool to replace `pip`, `pip-tools`, `pipx`, `poetry`, `pyenv`, `twine`, `virtualenv`, and more." You will need to install it.

We use `uv` as it tracks the latest PEP standards while relying on existing infrastructure like `pip`, `venv`, etc.

`uv.lock` is non-standard in Python, but as Python does not yet define a lockfile standard, any lockfile format is non-standard. The benefit of `uv.lock` is that it is cross-platform and "captures the packages that would be installed across all possible Python markers such as operating system, architecture, and Python version". This makes it safe to check-in to the repository.

### Install `uv`

On a Mac, `brew install uv` is sufficient. For additional options, see the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

### Install dependencies

```bash
uv sync
```

### Run formatter

```bash
uv run ruff format
```

### Run style checks

```bash
uv run ruff check
```

### Run dependency checks

```bash
uv run deptry src tests
```

### Run static type checks

```bash
uv run mypy
```

### Run tests

```bash
uv run pytest
```

### Build package

```bash
uv build
```
