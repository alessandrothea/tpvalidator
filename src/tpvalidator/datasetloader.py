from pathlib import Path
import json
import logging
from typing import Optional
from pydantic import BaseModel, field_validator
import tpvalidator
import tpvalidator.mcprod.workspace as workspace

_package_root = Path(tpvalidator.__file__).parents[2]
from tpvalidator.utilities import temporary_log_level
from rich import print

_log = logging.getLogger(__name__)


class DatasetsConfig(BaseModel):
    dataset_path: str
    dataset_info: dict
    datasets_spec: dict[str, str]
    tp_cut: Optional[str] = None
    tp_info_update: Optional[dict] = None

    @field_validator("datasets_spec")
    @classmethod
    def datasets_spec_not_empty(cls, v):
        if not v:
            raise ValueError("datasets_spec must not be empty")
        return v

def load(dataset_dir: str):

    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = _package_root / dataset_dir

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist")
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"'{dataset_dir}' is not a directory")

    cfg_file = dataset_dir / "datasets.json"
    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file '{cfg_file}' does not exist")
    if not cfg_file.is_file():
        raise ValueError(f"'{cfg_file}' is not a regular file")

    try:
        with open(cfg_file) as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse '{cfg_file}': {e}")

    cfg = DatasetsConfig.model_validate(raw)

    dataset_path = dataset_dir / cfg.dataset_path
    datasets = {}
    for name, filename in cfg.datasets_spec.items():
        with temporary_log_level(workspace.TriggerPrimitivesWorkspace._log, logging.INFO):
            ws = workspace.TriggerPrimitivesWorkspace(dataset_path / filename, extra_info=cfg.dataset_info)
        print(f"Dataset '{name}': {ws.num_events} events")
        print(ws.info)
        datasets[name] = ws

    if cfg.tp_cut:
        for ws in datasets.values():
            ws.tps.query(cfg.tp_cut, inplace=True)

    if cfg.tp_info_update:
        for ws in datasets.values():
            ws.tps.extra_info.update(cfg.tp_info_update)

    return datasets
