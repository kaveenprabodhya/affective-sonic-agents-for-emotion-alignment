"""Locate the project root and load YAML config from it.

The root is found by walking upward until a directory containing both `config/`
and `src/` is seen, so any module can load config without fragile relative paths.
"""
from pathlib import Path
import yaml


def find_root(start=None) -> Path:
    p = Path(start or __file__).resolve()
    for parent in [p, *p.parents]:
        if (parent / "config").is_dir() and (parent / "src").is_dir():
            return parent
    raise RuntimeError("Project root (a dir with both config/ and src/) not found.")


ROOT = find_root()
CONFIG = ROOT / "config"
LOGS = ROOT / "logs"


def load(name: str):
    """Load a YAML file from config/ by filename, e.g. load('personas.yaml')."""
    return yaml.safe_load((CONFIG / name).read_text())


def read_prompt(rel: str) -> str:
    """Read a prompt template, e.g. read_prompt('prompts/audience_system.txt')."""
    return (CONFIG / rel).read_text()