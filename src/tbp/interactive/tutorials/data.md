
## Getting Started with `DataParser` and `DataLocator`


### The Example Data

This first section shows how to load data with a `DataParser`.
We will use a toy example of some json data in the file `toy_data/detailed_run_stats.json`.

This data is structured as shown below.
```python
{
  "episode_1": [
    { "evidence": 0.15, "pose_error": 18.0 },
    { "evidence": 0.42, "pose_error": 14.5 },
    { "evidence": 0.60, "pose_error": 11.2 }
  ],
  "episode_2": [
    { "evidence": 0.05, "pose_error": 22.1 },
    { "evidence": 0.20, "pose_error": 19.8 },
    { "evidence": 0.35, "pose_error": 16.0 },
    { "evidence": 0.50, "pose_error": 12.7 }
  ],
  "episode_3": [
    { "evidence": 0.10, "pose_error": 21.5 },
    { "evidence": 0.28, "pose_error": 17.9 }
  ]
}
```

- Top level is a dictionary keyed by episode names: `episode_1`, `episode_2`, …
- Each episode maps to a list of steps
- Each step is a dictionary with two numbers: `evidence` and `pose_error`

### Loading the Data

The `DataParser.__init__` joins the provided path with `detailed_run_stats.json`, and loads it.
The loaded structure is stored at `parser.data`, so you can directly inspect or iterate over it.

```python
from pathlib import Path

parser = DataParser(path="toy_data")  # points to the directory, not the file
data = parser.data

# Basic sanity checks
print(type(data))                    # dict
print(list(data.keys()))             # ['episode_1', 'episode_2', 'episode_3']
print(len(data["episode_2"]))        # 4 steps in episode_2
print(data["episode_1"][0])          # {'evidence': 0.15, 'pose_error': 18.0}
print(data["episode_1"][0]["evidence"])  # 0.15
```

### Accessing Data

1) Full path Locators

A full path to a single value in the toy data structure could be:
```python
root["episode_2"][3]["evidence"] # returns 0.5
```

We can encode this full path with a `DataLocator`:
```python
loc_full = DataLocator([
    DataLocatorStep.key("episode", value="episode_2"),  # dict key at top level
    DataLocatorStep.index("step", value=3),             # list index inside episode_2
    DataLocatorStep.key("field", value="evidence"),     # dict key inside the step dict
])
```
Then we can extract the data from the `DataParser`:
```python
parser.extract(loc_full) # returns 0.5
```

2) Partial Locator

Think of a `DataLocator` as a partial function over your dataset’s path.
It captures the known part of a path now, and defers the unknown part to be provided later.
When you call `DataParser.extract(locator, **kwargs)`, you pass the missing steps to complete the path.


Create a locator that fixes the episode and metric, but leaves the step index as `None`.

```python
loc_partial = DataLocator([
    DataLocatorStep.key("episode", value="episode_2"),
    DataLocatorStep.index("step"),      # missing value (index)
    DataLocatorStep.key("metric", value="evidence"),
])
```

We can now extract the evidence value at any step:
```python
parser.extract(loc_partial, step=0) # returns 0.05
parser.extract(loc_partial, step=1) # returns 0.20
parser.extract(loc_partial, step=2) # returns 0.35
parser.extract(loc_partial, step=3) # returns 0.50
```

3) Querying Valid Options

We can also use the partial locators to discover valid options in a data structure.
When a locator has missing steps, `parser.query(locator, **kwargs)` returns the available values for the first missing step.

```python
loc_partial = DataLocator([
    DataLocatorStep.key("episode"),      # missing value (key)
    DataLocatorStep.index("step"),      # missing value (index)
    DataLocatorStep.key("metric", value="evidence"),
])
```

**What episodes are available?**

```python
parser.query(loc_partial) # returns ["episode_1", "episode_2", "episode_3"]
```

**What step indices exist for a given episode?**

```python
parser.query(loc_partial, episode="episode_1") # returns [0,1,2]
parser.query(loc_partial, episode="episode_2") # returns [0,1,2,3]
```


4) Extending `DataLocator`s

For longer paths, `DataLocator`s can be easily extended.
You can compose locators by starting general and then extending them.
This keeps the code modular and allows you progressively specialize a path.

```python
episode_locator = DataLocator([DataLocatorStep.key("episode", value="episode_3")])
step_locator = episode_locator.extend([DataLocatorStep.index("step")])
metric_locator = step_locator.extend([DataLocatorStep.key("metric", value="pose_error")])

parser.extract(metric_locator, step=1) # returns 17.9
parser.extract(step_locator, step=1) # returns {"evidence": 0.28, "pose_error": 17.9}
parser.query(step_locator, step=1) # returns ["evidence", "pose_error"]
```
