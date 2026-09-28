"""Shared fixtures. BLEnD-dependent tests are skipped when the BLEnD data is not found."""
import os
from pathlib import Path

import pytest

from blend.config import COUNTRIES, Paths
from blend.data import BlendDataset

ROOT = Path(__file__).resolve().parent.parent


def _blend_dir():
    for c in (os.environ.get("BLEND_DIR"), ROOT / "external" / "BLEnD", ROOT.parent / "BLEnD"):
        if c and (Path(c) / "data" / "annotations").exists():
            return Path(c)
    return None


@pytest.fixture(scope="session")
def blend_dir():
    d = _blend_dir()
    if d is None:
        pytest.skip("BLEnD data not found (set BLEND_DIR=../BLEnD)")
    return d


@pytest.fixture(scope="session")
def iran(blend_dir):
    return BlendDataset(COUNTRIES["Iran"], blend_dir)


@pytest.fixture(scope="session")
def azerbaijan(blend_dir):
    return BlendDataset(COUNTRIES["Azerbaijan"], blend_dir)


@pytest.fixture
def tmp_paths(tmp_path, blend_dir):
    return Paths(blend_dir=blend_dir, results=tmp_path / "results")
