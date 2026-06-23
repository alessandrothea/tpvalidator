from pathlib import Path
import logging
import yaml
from typing import Optional, List
from pydantic import BaseModel, field_validator
import tpvalidator
import tpvalidator.workspace as workspace

_package_root = Path(tpvalidator.__file__).parents[2]
from tpvalidator.utils import temporary_log_level
from rich import print

_log = logging.getLogger(__name__)


class DatasetEntry(BaseModel):
    trg_file: str
    rawadc_file: Optional[str] = None
    label: Optional[str] = None
    first_entry: Optional[int] = None
    last_entry: Optional[int] = None



class DataCatalogue(BaseModel):
    dataset_path: str
    dataset_info: dict
    datasets_spec: dict[str, DatasetEntry]
    tp_cut: Optional[str] = None
    tp_info_update: Optional[dict] = None

    @field_validator("datasets_spec")
    @classmethod
    def datasets_spec_not_empty(cls, v):
        if not v:
            raise ValueError("datasets_spec must not be empty")
        return v

def _resolve_dir(dataset_dir: str) -> Path:
    p = Path(dataset_dir)
    if not p.is_absolute():
        p = _package_root / p
    if not p.exists():
        raise FileNotFoundError(f"Dataset directory '{p}' does not exist")
    if not p.is_dir():
        raise NotADirectoryError(f"'{p}' is not a directory")
    return p


def parse(dataset_dir: str) -> DataCatalogue:
    """Find and parse the datacatalogue.yaml in *dataset_dir*, return a DataCatalogue."""
    d = _resolve_dir(dataset_dir)
    cfg_file = d / "datacatalogue.yaml"
    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file '{cfg_file}' does not exist")
    if not cfg_file.is_file():
        raise ValueError(f"'{cfg_file}' is not a regular file")
    try:
        with open(cfg_file) as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse '{cfg_file}': {e}")
    return DataCatalogue.model_validate(raw)


def list_datasets(dataset_dir: str) -> List[str]:
    """Return the names of datasets declared in the catalogue."""
    return list(parse(dataset_dir).datasets_spec.keys())


def load_datasets(dataset_dir: str, load_rawadc:bool=True, selection: Optional[List[str]] = None) -> dict:
    """Load workspace objects for each dataset (optionally filtered by *selection*)."""
    d = _resolve_dir(dataset_dir)
    cfg = parse(dataset_dir)
    dataset_path = d / cfg.dataset_path
    datasets = {}
    for name, entry in cfg.datasets_spec.items():
        if selection and name not in selection:
            print(f"Workspace {name} skipped")
            continue

        print(f"Loading {name}")
        ws = workspace.TriggerPrimitivesWorkspace(dataset_path / entry.trg_file,
                                                first_entry=entry.first_entry,
                                                last_entry=entry.last_entry,
                                                extra_info=cfg.dataset_info
                                                )
        if entry.rawadc_file and load_rawadc:
            print(f"Adding {entry.rawadc_file}")
            ws.add_rawdigits(str(dataset_path / entry.rawadc_file))
        print(f"Dataset '{name}': {ws.num_entries} events")
        print(ws.info)
        datasets[name] = ws

    if cfg.tp_cut:
        for ws in datasets.values():
            ws.tps.query(cfg.tp_cut, inplace=True)

    if cfg.tp_info_update:
        for ws in datasets.values():
            ws.tps.extra_info.update(cfg.tp_info_update)

    return datasets


def load(dataset_dir: str, selection: Optional[List[str]] = None, load_rawadc:bool=True ) -> dict:
    """Load all datasets from a catalogue directory (convenience wrapper)."""
    return load_datasets(dataset_dir, load_rawadc, selection)


def iterdataset_xp(dataset_dir, dataset_name, num_entries, load_rawadc:bool=False):

    d = _resolve_dir(dataset_dir)
    cfg = parse(dataset_dir)
    print(cfg)
    dataset_path = d / cfg.dataset_path
    if dataset_name not in cfg.datasets_spec:
        raise KeyError(f'Dataset {dataset_name} not found in {dataset_dir}')

    dataset = cfg.datasets_spec[dataset_name]
    ws = workspace.TriggerPrimitivesWorkspace(dataset_path / dataset.trg_file)

    total_num_entries = ws.num_entries
    del ws
    print(f"Found {total_num_entries} entries")

    first_entry = dataset.first_entry if dataset.first_entry is not None else 0
    last_entry = dataset.last_entry if dataset.last_entry is not None else total_num_entries

    for i in range(first_entry, last_entry, num_entries):
        print(i, i+num_entries)
        yield workspace.TriggerPrimitivesWorkspace(dataset_path / dataset.trg_file, first_entry=i, last_entry=i+num_entries)
    return None
