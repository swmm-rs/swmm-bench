# swmm-bench

Benchmark and regression-test compatible [SWMM](https://www.epa.gov/water-research/storm-water-management-model-swmm) executables.

`swmm-bench` measures runtime and peak memory on SWMM input models. `swmm-test` runs a curated numerical regression suite, including interface-file interoperability checks. With two or more engines, both commands compare report tables and binary `.out` time series and produce terminal, JSON, and optional HTML reports.

> [!IMPORTANT]
> **EPA SWMM test coverage:** [view the latest report](http://karosc.github.io/swmm-bench/epa-swmm-coverage.html).
>
> The regression suite covers broad solver behavior, but the [largest gaps](docs/epa-coverage-analysis.md)
> are specialized hydraulic regimes and numerical boundaries: cross-section geometry, inlet and roadway
> routing, LID/groundwater states, and external-interface formats. Input validation, compatibility, and
> other error paths are also incomplete.

## Requirements

- Python 3.11+
- One or more executable files that accept:

  ```text
  ENGINE input.inp report.rpt output.out
  ```

## Install

From a checkout, install the locked project environment with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Run commands through that environment:

```bash
uv run swmm-bench --help
uv run swmm-test --help
```

## Quick start

Benchmark the bundled stress models with one engine:

```bash
uv run swmm-bench run /path/to/swmm
```

Compare two engines on your own models:

```bash
uv run swmm-bench run /path/to/swmm-a /path/to/swmm-b \
  --inp /path/to/models --recursive --name nightly
```

The run is saved under `swmm-bench-results/nightly/`. Omitting `--name` creates a timestamped directory.

## Commands

### `swmm-bench run`

Run one or more engines against input files. Omit `--inp` to use the bundled long-running benchmark models.

Run one bundled benchmark model by its category-relative path:

```bash
uv run swmm-bench run /path/to/swmm --model stress/fredericksburg.inp
```

```bash
uv run swmm-bench run /path/to/swmm-a /path/to/swmm-b \
  --inp model.inp \
  --runs 5 \
  --inp-option THREADS=4 \
  --parse-workers 4 \
  --output-parse-workers 2 \
  --report-size-mb 100 \
  --inp-option VARIABLE_STEP=0.5 \
  --timeout 600 \
  --output-dir results \
  --name baseline
```

| Option | Purpose |
| --- | --- |
| `--inp PATH` | Repeatable input file or directory. |
| `--model PATH` | Run one exact bundled benchmark model; cannot be combined with `--inp`. |
| `--recursive` | Search supplied directories recursively. |
| `--pattern TEXT` | Input-file glob; defaults to `*.inp`. |
| `--inp-option NAME=VALUE` | Override a model `[OPTIONS]` value; repeatable. Defaults to `THREADS=1`. |
| `--runs N` | Measured runs per engine/model; repeated runs are interleaved and median timing is reported. Defaults to `1`. |
| `--parse-workers N` | Processes used to parse RPT files; defaults to `4`. |
| `--output-parse-workers N` | Processes used to parse OUT files; defaults to `1` because each worker can consume substantial memory. |
| `--report-size-mb N` | Approximate HTML report size target; defaults to `100`. Complete time series are retained for the highest-distance series within this budget; chart timesteps are never sampled. |
| `--timeout SECONDS` | Per-run timeout; defaults to `300`. |
| `--output-dir PATH` | Results parent directory; defaults to `swmm-bench-results`. |
| `--name TEXT` | Name for this run directory. |
| `--html / --no-html` | Generate or disable the HTML report. |
| `--json-out PATH` | Write `results.json` to a specific path. |

### `swmm-test run`

Run the compact, feature-oriented regression suite. It covers hydrology, hydraulics, controls, routing, water quality, and interface consumers without the benchmark suite's longest simulations.

```bash
# See available categories and model paths.
uv run swmm-test list

# Run all bundled regression models.
uv run swmm-test run /path/to/swmm-a /path/to/swmm-b

# Narrow the run to a category or exact model path.
uv run swmm-test run /path/to/swmm-a /path/to/swmm-b --category hydrology
uv run swmm-test run /path/to/swmm-a /path/to/swmm-b \
  --model water-quality/waterquality-events_example.inp
```

`swmm-test run` accepts repeatable `--inp-option NAME=VALUE` overrides, plus `--parse-workers`, `--output-parse-workers`, `--report-size-mb`, `--timeout`, `--output-dir`, `--name`, `--html / --no-html`, and `--json-out`. Its default results parent is `swmm-test-results`.

### `swmm-test interface`

Create interface files with a source engine, then consume them with explicit target engines. By default, it covers rainfall, runoff, hotstart, RDII, and routing interfaces.

```bash
uv run swmm-test interface /path/to/source-swmm \
  /path/to/target-swmm-a /path/to/target-swmm-b

# Limit the run; repeat --family to select more than one.
uv run swmm-test interface /path/to/source-swmm /path/to/target-swmm \
  --family rainfall --family hotstart
```

The command accepts repeatable `--inp-option NAME=VALUE` overrides for every engine run, plus `--parse-workers` and `--output-parse-workers` for comparison parsing and `--report-size-mb` for chart retention. It retains generated interfaces and records each path, size, and SHA-256 in `results.json`. A failed engine, missing interface, or empty interface produces a nonzero exit after reports are written.

### Rebuild or render reports

Recalculate report comparisons from retained run artifacts without executing SWMM:

```bash
uv run swmm-bench rebuild swmm-bench-results/nightly
```

Add `--outputs` to also recalculate binary-output comparisons and retain their chart data. Add `--all-comparisons` to retain and render chart data even when the overall distance is 1% or less. Rebuild also accepts `--parse-workers`, `--output-parse-workers`, and `--report-size-mb`.

Use `--output-typical-weight` (default `0.75`) on `run`, `rebuild`, and `swmm-test` commands to balance the output-distance score. A value of `1` scores only typical-over-time disagreement; `0` scores only event disagreement. The report retains both component scores while using their weighted composite to sort and filter.

OUT parsing returns large DataFrames from each worker process. Increase `--output-parse-workers` only when enough RAM is available; `2` or `3` is a practical starting point for large artifacts.

Render a saved JSON result as HTML:

```bash
uv run swmm-bench report results.json --output report.html
```

Use `--engine-order` to put one or more engines first in the regenerated
report's performance chart, table, and distance controls. Repeat the option to
set the preferred order; unlisted engines retain their saved order:

```bash
uv run swmm-bench report results.json --output report.html \
  --engine-order runswmm \
  --engine-order epaswmm
```

## Results

Each run directory contains:

- `results.json` — machine-readable run metadata, measurements, and comparisons.
- `report.html` — interactive report when HTML output is enabled.
- Per-engine, per-model artifacts — copied inputs, stdout, stderr, `result.rpt`, and `result.out`.

A single engine records measurements and artifacts. Two or more engines additionally produce pairwise report-table and output-time-series comparisons. Report and output distances are intentionally reported as separate scores.

### Report distance

Report tables are aligned on the union of their row and column labels. Each aligned cell contributes a score from `0` to `1`:

- equal values, including paired nulls: `0`
- values present on only one side, unequal booleans, unequal text, or non-finite numbers: `1`
- finite numbers and durations: `min(abs(A - B) / max(abs(A), abs(B)), 1)`

Each table's score is the mean of its cell scores; the overall report distance is the cell-count-weighted mean across all parsed tables.

### Output distance

For finite values paired at the same timestamp, the binary-output comparator retains two bounded components:

```text
typical distance = min(mean(abs(A - B)) / max(abs(A), abs(B)), 1)
event distance = min(RMSE(A - B) / max(RMS(A), RMS(B)), 1)
```

The primary score is their configurable composite. The default gives 75% weight to typical-over-time disagreement and 25% to event disagreement, so a brief missed spike remains visible without being treated as a wholly different series.

One-sided timestamps within the common output horizon, one-sided null or invalid values, and series emitted by only one engine count as missing. Paired nulls are neutral. Timestamps after the earlier output's final timestamp are recorded as trailing timeline coverage and do not affect value distance:

```text
series distance = missing fraction + (1 - missing fraction) * (
  typical weight * typical distance + event weight * event distance
)
```

The overall output distance is the equal-weight mean across semantic series. The generated HTML report includes the same definition and identifies legacy schema-2/3 scores separately.

## Acknowledgements

Some bundled regression models are copied or adapted from [SWMMEnablement/1729-SWMM5-Models](https://github.com/SWMMEnablement/1729-SWMM5-Models/) and [pyswmm/swmm-nrtestsuite](https://github.com/pyswmm/swmm-nrtestsuite). Per-model source notes live in `swmm_bench/resources/regression-suite/README.md` and `CITATION.txt`.

The bundled `stress/fredericksburg.inp` benchmark adapts [Shahed Behrouz and Ahmadi's Fredericksburg model](https://doi.org/10.4211/hs.2c4a324dbe0d487690b7b79eb0bfd618), distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Its `[TITLE]` section records the source checksum and benchmark-specific changes.

## Development

```bash
uv run pytest
uv run ruff check .
uv run swmm-bench --help
uv run swmm-test --help
```
