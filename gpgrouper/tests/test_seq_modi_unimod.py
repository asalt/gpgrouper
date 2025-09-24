"""Tests for UniMod handling in seq_modi."""

from importlib import util
from pathlib import Path


def _load_subfuncts():
    module_path = Path(__file__).resolve().parents[1] / "subfuncts.py"
    spec = util.spec_from_file_location("gpgrouper_subfuncts", module_path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


subfuncts = _load_subfuncts()


def test_seq_modi_preserves_unimod_tag():
    sequence, seqmodi, count, label = subfuncts.seq_modi(
        "SSSS", "S2(UniMod:21)"
    )

    assert sequence == "SSSS"
    assert seqmodi == "SS(UniMod:21)SS"
    assert count == 1
    assert label == 0
