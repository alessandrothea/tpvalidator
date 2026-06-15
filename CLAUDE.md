# tpvalidator — Claude context

## What this project is
DUNE trigger-primitive validation package. Analyses simulated trigger primitives (TPs)
and trigger activities (TAs) from the DUNE vertical-drift (VD) and horizontal-drift (HD)
detectors. Core workflows live in Jupyter notebooks; the `tpvalidator` package provides
the library backing them.

## Toolchain
- **Python 3.13** (`.python-version`)
- **uv** for environment and dependency management — always use `uv` instead of pip
  - `uv sync` to install/update the environment
  - `uv run <cmd>` or activate `.venv` to run scripts
- **pytest** for tests: `pytest test/`
- No linter/formatter is configured yet

## Package layout (`src/tpvalidator/`)
| Module | Purpose |
|---|---|
| `workspace.py` | Top-level analysis objects (`TriggerAnalysisWorkspace`, etc.) |
| `rootio/` | ROOT file I/O — `NtupleReader` (reader.py) and `NtupleWriter` (writer.py) |
| `algo/` | TP/TA algorithms — DBSCAN clustering, TPG emulator, numba-JIT code |
| `viz/` | Plotting — histograms, TPC geometry display (`geo.py`), backtracker |
| `datasets/` | Dataset helpers (`miniprod.py`) |
| `cli/` | `ta-finder` CLI entry point |
| `report/` | PDF report generation |
| `detector_geometry.py` | Detector geometry constants |
| `tpc_angles.py` | Wire-angle / drift-direction calculations |

## Data
- `data/` is **git-ignored** — never commit data files
- Geometry JSONs live under `data/vd/geo/` and `data/hd/geo/`

## Notebooks
- `notebooks/devel/` — exploratory / debugging notebooks
- `notebooks/studies/` — physics analysis notebooks
- `notebooks/archive/` — legacy, kept for reference

## NtupleWriter API (`rootio/writer.py`)
- Path and mode are constructor arguments: `NtupleWriter(path, key_columns, ...)`
- Supports multiple TTrees per file via optional `tree_name` on `write()` / `extend()`
- Use as context manager or call `.close()` explicitly

## Key conventions
- ROOT files are read/written with **uproot** + **awkward** arrays
- Heavy loops use **numba** JIT — keep numba-compiled functions in dedicated files
- Physics quantities are in **cm** (positions) and **ADC counts** (charge)
- VD drift direction is **x**; HD drift direction is **y**

## Test conventions
- Every assert must be preceded by a `print()` that shows the computed value(s) and
  expected value(s), so developers can run `pytest -vs` to inspect results without
  adding debug statements on the fly.
- Prefer f-strings: `print(f"\n<label>: {got}, expected: {expected}")`
- Skip prints only for `pytest.raises` blocks (nothing to show before the exception).
- Run tests with `uv run pytest test/ -vs` to see all print output.
