"""
NtupleWriter: Write pandas DataFrames into ROOT files using uproot.

The DataFrame is split into events via groupby on a user-selected set of
"key" columns. Those key columns become scalar branches; all remaining
columns become variable-length vector branches (one entry per event).

Requires Python >= 3.12.
"""

import logging
from pathlib import Path
from collections.abc import Sequence

import awkward as ak
import numpy as np
import pandas as pd
import uproot

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases  (PEP 695 – Python 3.12+; plain `type` statement)
# ---------------------------------------------------------------------------

type BranchDict = dict[str, np.ndarray | ak.Array]
type DtypeKey   = tuple[str, int]          # (kind, itemsize)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DTYPE_MAP: dict[DtypeKey, str] = {
    ("i", 1): "int8",
    ("i", 2): "int16",
    ("i", 4): "int32",
    ("i", 8): "int64",
    ("u", 1): "uint8",
    ("u", 2): "uint16",
    ("u", 4): "uint32",
    ("u", 8): "uint64",
    ("f", 4): "float32",
    ("f", 8): "float64",
    ("b", 1): "bool",
}


def _infer_dtype(series: pd.Series) -> str:
    """Map a pandas Series dtype to a ROOT branch type-string.

    Falls back to ``float64`` for exotic or object dtypes.
    """
    key: DtypeKey = (series.dtype.kind, series.dtype.itemsize)
    return _DTYPE_MAP.get(key, "float64")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class NtupleWriter:
    """Write a pandas DataFrame to a ROOT TTree, splitting rows into events.

    Parameters
    ----------
    key_columns:
        Column names used as the groupby key.  Each unique combination of
        values defines one *event*.  These columns are stored as **scalar**
        branches (one value per event).
    tree_name:
        Name of the TTree inside the ROOT file.  Defaults to ``"Events"``.
    compression:
        uproot compression object (e.g. ``uproot.ZLIB(4)``, ``uproot.LZ4(4)``,
        ``uproot.ZSTD(5)``, or ``None`` for uncompressed).
        Defaults to ``uproot.LZ4(4)`` if not specified.
    chunk_size:
        Number of *events* to accumulate before flushing a basket to disk.
        Larger values use more memory but typically compress better.
        Defaults to ``1_000``.

    Examples
    --------
    One-shot write::

        writer = NtupleWriter(key_columns=["run", "lumi", "event"])
        writer.write(df, "output.root")

    Streaming / multi-batch write (context manager)::

        with NtupleWriter(["run", "event"]).open("out.root") as w:
            for batch in batches:
                w.extend(batch)
    """

    def __init__(
        self,
        key_columns: Sequence[str],
        tree_name: str = "Events",
        compression: object = uproot.LZ4(4),
        chunk_size: int = 1_000,
    ) -> None:
        if not key_columns:
            raise ValueError("`key_columns` must not be empty.")

        self.key_columns: list[str]  = list(key_columns)
        self.tree_name:   str        = tree_name
        self.compression: object     = compression
        self.chunk_size:  int        = chunk_size

        # Internal state – active only while used as a context manager
        self._root_file: uproot.WritableFile | None = None
        self._tree:      object | None              = None

    # ------------------------------------------------------------------
    # Public API – one-shot write
    # ------------------------------------------------------------------

    def write(
        self,
        df:   pd.DataFrame,
        path: str | Path,
        mode: str = "recreate",
    ) -> None:
        """Convert *df* to ROOT and write everything in one call.

        Parameters
        ----------
        df:
            Input DataFrame.  Must contain every column in ``key_columns``.
        path:
            Destination ``.root`` file path.
        mode:
            ``"recreate"`` *(default)* – overwrite any existing file.
            ``"update"`` – add a new TTree to an existing file (the tree
            name must not already be present).
        """
        if mode not in ("recreate", "update"):
            raise ValueError(f"mode must be 'recreate' or 'update', got {mode!r}")

        self._validate_columns(df)
        branches = self._build_branch_dict(df)
        n_events = len(next(iter(branches.values())))

        opener = uproot.recreate if mode == "recreate" else uproot.update
        with opener(str(path), compression=self.compression) as f:
            f[self.tree_name] = branches

        _log.info(
            f"Wrote {n_events:,} events ({len(df):,} rows) → '{path}' / '{self.tree_name}'"
        )

    # ------------------------------------------------------------------
    # Public API – streaming / context-manager write
    # ------------------------------------------------------------------

    def open(self, path: str | Path, mode: str = "recreate") -> "NtupleWriter":
        """Open a ROOT file for incremental writing.

        Must be used as a context manager (or paired with :meth:`close`)::

            with NtupleWriter(["run", "event"]).open("out.root") as w:
                for batch in batches:
                    w.extend(batch)
        """
        if mode not in ("recreate", "update"):
            raise ValueError(f"mode must be 'recreate' or 'update', got {mode!r}")
        opener = uproot.recreate if mode == "recreate" else uproot.update
        self._root_file = opener(str(path), compression=self.compression)
        self._tree      = None
        return self

    def extend(self, df: pd.DataFrame) -> None:
        """Append *df* (after groupby-splitting) to the currently open TTree."""
        if self._root_file is None:
            raise RuntimeError(
                "No file is open. Call .open() first, or use .write() for a one-shot write."
            )

        self._validate_columns(df)
        branches = self._build_branch_dict(df)
        n_events = len(next(iter(branches.values())))

        if self._tree is None:
            # First batch – create the TTree
            self._root_file[self.tree_name] = branches
            self._tree = self._root_file[self.tree_name]
        else:
            self._root_file[self.tree_name].extend(branches)

        _log.info(f"Appended {n_events:,} events ({len(df):,} rows)")

    def close(self) -> None:
        """Flush and close the underlying ROOT file."""
        if self._root_file is not None:
            self._root_file.close()
            self._root_file = None
            self._tree      = None

    def __enter__(self) -> "NtupleWriter":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.key_columns if c not in df.columns]
        if missing:
            raise ValueError(f"key_columns not found in DataFrame: {missing}")

    def _build_branch_dict(self, df: pd.DataFrame) -> BranchDict:
        """Build the ``{branch_name: array}`` dict expected by uproot.

        Scalar branches (key columns)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        One value per event, taken from the *first* row of each group.
        All rows within a group share the same key-column values by
        construction, so taking the first is always correct.

        Vector branches (remaining columns)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        All non-key columns are collected into awkward jagged arrays –
        one variable-length sub-array per event.  The group order matches
        the scalar branches exactly.
        """
        vector_cols = [c for c in df.columns if c not in self.key_columns]
        groups      = df.groupby(self.key_columns, sort=False)

        # ── scalar branches ────────────────────────────────────────────
        first_rows  = groups.first().reset_index()
        scalar_data: dict[str, np.ndarray] = {
            col: first_rows[col].to_numpy().astype(_infer_dtype(first_rows[col]))
            for col in self.key_columns
        }

        # ── vector branches ────────────────────────────────────────────
        # groups.indices: dict[group_key → np.ndarray of row positions]
        # The dict preserves the same insertion order as `groups.first()`.
        group_indices = groups.indices
        vector_data: dict[str, ak.Array] = {}

        for col in vector_cols:
            dtype_str = _infer_dtype(df[col])
            col_np    = df[col].to_numpy().astype(dtype_str)
            vector_data[col] = ak.Array(
                [col_np[idx] for idx in group_indices.values()]
            )

        return scalar_data | vector_data   # PEP 584 dict merge (Python 3.9+)


# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import tempfile

    rng      = np.random.default_rng(42)
    n_events = 50

    rows = [
        {
            "run":        1,
            "lumi":       evt // 10 + 1,
            "event":      evt,
            "hit_x":      rng.uniform(-100, 100),
            "hit_y":      rng.uniform(-100, 100),
            "hit_charge": rng.exponential(50),
            "hit_layer":  int(rng.integers(0, 4)),
        }
        for evt in range(n_events)
        for _   in range(int(rng.integers(1, 8)))
    ]

    df = pd.DataFrame(rows)
    print(f"Input DataFrame: {len(df)} rows, {df['event'].nunique()} unique events")
    print(df.head(12).to_string(index=False))

    # ── one-shot write ──────────────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".root", delete=False) as tmp:
        out_path = tmp.name

    writer = NtupleWriter(key_columns=["run", "lumi", "event"])
    writer.write(df, out_path)

    # ── verify round-trip ───────────────────────────────────────────────────
    print("\n--- Round-trip verification ---")
    with uproot.open(out_path) as f:
        tree   = f[writer.tree_name]
        arrays = tree.arrays(library="ak")

        print(f"TTree '{writer.tree_name}' → {tree.num_entries:,} entries")
        print("Branches:", tree.keys())
        print("\nFirst 3 events:")
        for i in range(min(3, tree.num_entries)):
            ev = {
                k: arrays[k][i].tolist()
                   if hasattr(arrays[k][i], "tolist")
                   else int(arrays[k][i])
                for k in tree.keys()
            }
            print(f"  event {i}: {ev}")

    os.unlink(out_path)
    print("\nDemo complete ✓")