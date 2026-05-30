#!/usr/bin/env python


from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import duckdb
import pandas as pd


@dataclass(frozen=True)
class ParquetCachePool:
    root: Path

    def dataset_dir(self, name: str) -> Path:
        return self.root / "parquet" / name

    def meta_path(self, name: str) -> Path:
        return self.dataset_dir(name) / "_meta.json"

    def exists(self, name: str) -> bool:
        d = self.dataset_dir(name)
        return d.exists() and any(d.glob("*.parquet"))

    def write_df(
        self,
        name: str,
        df: pd.DataFrame,
        *,
        partition_by: Optional[list[str]] = None,
        overwrite: bool = True,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        """
        Write a DataFrame as a Parquet *dataset* (directory of parquet files).
        Optionally partition by columns for faster downstream filtering.
        """
        d = self.dataset_dir(name)
        d.parent.mkdir(parents=True, exist_ok=True)

        if overwrite and d.exists():
            # Remove old dataset
            for p in d.rglob("*"):
                if p.is_file():
                    p.unlink()
            for p in sorted(d.rglob("*"), reverse=True):
                if p.is_dir():
                    p.rmdir()

        d.mkdir(parents=True, exist_ok=True)

        con = duckdb.connect()
        con.register("df", df)

        if partition_by:
            part_cols = ", ".join([f"'{c}'" for c in partition_by])
            con.execute(
                f"""
                COPY df TO '{d.as_posix()}'
                (FORMAT PARQUET, PARTITION_BY ({part_cols}), OVERWRITE_OR_IGNORE TRUE);
                """
            )
        else:
            # write a single parquet file inside the dataset dir
            out_file = d / "part-000.parquet"
            con.execute(
                f"""
                COPY df TO '{out_file.as_posix()}'
                (FORMAT PARQUET, OVERWRITE_OR_IGNORE TRUE);
                """
            )

        meta = {
            "name": name,
            "created_unix": time.time(),
            "n_rows": int(len(df)),
            "columns": list(df.columns),
            "partition_by": partition_by or [],
            "user_metadata": dict(metadata or {}),
        }
        self.meta_path(name).write_text(json.dumps(meta, indent=2))
        return d

    def query_df(self, sql: str, *, params: Optional[tuple[Any, ...]] = None) -> pd.DataFrame:
        """
        Run a DuckDB SQL query and return a pandas DataFrame.
        """
        con = duckdb.connect()
        if params is None:
            return con.execute(sql).df()
        return con.execute(sql, params).df()

    def table_ref(self, name: str) -> str:
        """
        Return a DuckDB FROM reference to this dataset.
        DuckDB can query Parquet datasets by pointing to the directory or glob.
        """
        d = self.dataset_dir(name)
        # directory dataset: DuckDB reads all parquet files inside
        return f"'{d.as_posix()}'"

    def read_meta(self, name: str) -> dict[str, Any]:
        p = self.meta_path(name)
        return json.loads(p.read_text()) if p.exists() else {}
