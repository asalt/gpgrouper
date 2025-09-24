"""Tests for SequenceModi cleaning helpers in gpgrouper."""

from importlib import util
from pathlib import Path
from types import ModuleType
import sys

import pandas as pd


def _load_gpgrouper():
    """Load gpgrouper module with minimal stubs for external deps."""

    root = Path(__file__).resolve().parents[1]

    # Ensure we import the in-repo package rather than any installed copy.
    sys.modules.pop("gpgrouper", None)
    pkg = ModuleType("gpgrouper")
    pkg.__path__ = [str(root)]
    sys.modules["gpgrouper"] = pkg

    # Stub RefProtDB dependency required during import.
    refprot = ModuleType("RefProtDB")
    utils = ModuleType("RefProtDB.utils")
    utils.fasta_dict_from_file = lambda *args, **kwargs: {}
    refprot.utils = utils
    sys.modules.setdefault("RefProtDB", refprot)
    sys.modules.setdefault("RefProtDB.utils", utils)

    spec = util.spec_from_file_location("gpgrouper._clean_test", root / "gpgrouper.py")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gpg = _load_gpgrouper()


def test_clean_modified_sequence_replaces_known_labels():
    series = pd.Series(["M1(Carbamidomethyl);S2(Phospho);K3(Label:13C(6)+GlyGly)"])
    cleaned = gpg.clean_modified_sequence(series).iloc[0]

    assert "car" in cleaned
    assert "pho" in cleaned
    assert "lab" in cleaned
    assert "gg" in cleaned


def test_clean_modified_sequence_preserves_unimod21():
    series = pd.Series(["S2(UniMod:21);S3(Phospho)"])
    cleaned = gpg.clean_modified_sequence(series).iloc[0]

    assert "UniMod:21" in cleaned
    assert "pho" in cleaned  # Phospho should still abbreviate


def test_set_modifications_retains_unimod21_in_sequence():
    df = pd.DataFrame(
        {
            "Sequence": ["SSSS"],
            "Modifications": ["S2(UniMod:21)"]
        }
    )

    result = gpg.set_modifications(df.copy())
    assert result.loc[0, "SequenceModi"] == "SS(UniMod:21)SS"
    assert result.loc[0, "SequenceModiCount"] == 1
