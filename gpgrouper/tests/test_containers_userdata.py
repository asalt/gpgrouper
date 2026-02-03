"""Focused tests for containers.UserData behaviour."""

from importlib import util
from pathlib import Path
from types import ModuleType
import sys

import pandas as pd
import pytest


def _load_containers_module(monkeypatch):
    """Load containers.py with minimal package scaffolding."""

    root = Path(__file__).resolve().parents[1]

    # Ensure we import modules from the working tree, not an installed package.
    monkeypatch.delitem(sys.modules, "gpgrouper", raising=False)
    package = ModuleType("gpgrouper")
    package.__path__ = [str(root)]
    monkeypatch.setitem(sys.modules, "gpgrouper", package)

    version_spec = util.spec_from_file_location("gpgrouper._version", root / "_version.py")
    version_module = util.module_from_spec(version_spec)
    version_spec.loader.exec_module(version_module)
    monkeypatch.setitem(sys.modules, "gpgrouper._version", version_module)

    spec = util.spec_from_file_location("gpgrouper.containers_test", root / "containers.py")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def containers_module():
    monkeypatch = pytest.MonkeyPatch()
    baseline_modules = set(sys.modules)
    try:
        yield _load_containers_module(monkeypatch)
    finally:
        monkeypatch.undo()
        for name in set(sys.modules) - baseline_modules:
            if name.startswith("gpgrouper."):
                sys.modules.pop(name, None)


@pytest.fixture
def UserData(containers_module):
    return containers_module.UserData


def test_userdata_basename_handles_missing_datafile(tmp_path, UserData):
    ud = UserData(
        recno="123",
        datafile=None,
        indir=str(tmp_path),
        outdir=str(tmp_path / "out"),
    )

    assert ud.basename == "123"
    assert ud.output_name() == "123_1_1_labelnone.tsv"


def test_categorical_assign_sets_category_dtype(tmp_path, UserData):
    ud = UserData(
        recno="1",
        datafile=None,
        indir=str(tmp_path),
        outdir=str(tmp_path / "out"),
    )
    ud.df = pd.DataFrame({"dummy": [0, 1]})

    ud.categorical_assign("NewColumn", "value")

    assert str(ud.df["NewColumn"].dtype) == "category"
    assert ud.df["NewColumn"].iloc[0] == "value"


def test_read_csv_missing_file_sets_error(tmp_path, UserData):
    ud = UserData(
        recno="1",
        datafile="missing.tsv",
        indir=str(tmp_path),
        outdir=str(tmp_path / "out"),
    )

    result = ud.read_csv(sep="\t")

    assert result == 1
    assert ud.EXIT_CODE == 1
    assert isinstance(ud.ERROR, str) and "Traceback" in ud.ERROR
    assert Path(ud.LOGFILE).exists()


def test_temp_file_properties_are_singleton_lists(tmp_path, UserData):
    ud = UserData(
        recno="1",
        datafile=None,
        indir=str(tmp_path),
        outdir=str(tmp_path / "out"),
    )

    first_e2g = ud.e2g_files
    second_e2g = ud.e2g_files
    assert first_e2g is second_e2g

    first_psm = ud.psm_files
    second_psm = ud.psm_files
    assert first_psm is second_psm

    first_msf = ud.msf_files
    second_msf = ud.msf_files
    assert first_msf is second_msf


def test_clean_removes_registered_temp_files(tmp_path, UserData):
    output_dir = tmp_path / "out"
    ud = UserData(
        recno="1",
        datafile=None,
        indir=str(tmp_path),
        outdir=str(output_dir),
    )

    e2g_path = output_dir / "temp_e2g.tsv"
    psm_path = output_dir / "temp_psm.tsv"
    msf_path = output_dir / "temp_msf.tsv"
    output_dir.mkdir(parents=True, exist_ok=True)
    e2g_path.write_text("test")
    psm_path.write_text("test")
    msf_path.write_text("test")

    ud.e2g_files.append(str(e2g_path))
    ud.psm_files.append(str(psm_path))
    ud.msf_files.append(str(msf_path))

    ud.clean()

    assert not e2g_path.exists()
    assert not psm_path.exists()
    assert not msf_path.exists()
    assert ud.e2g_files == []
    assert ud.psm_files == []
    assert ud.msf_files == []


def test_filterstamp_reflects_flags(tmp_path, UserData):
    ud = UserData(
        recno="1",
        datafile=None,
        indir=str(tmp_path),
        outdir=str(tmp_path / "out"),
    )
    ud.filtervalues = {
        "ion_score": 12,
        "qvalue": 0.05,
        "pep": 0.01,
        "idg": 3,
        "zmin": 1,
        "zmax": 6,
        "modi": 2,
        "ion_score_bins": (10, 20, 30),
    }

    assert ud.filterstamp.startswith("psmscore12_qv0.05_pep0.01_idg3_z1to6_mo2_psmscore_bins(10, 20, 30)")

    ud.phospho = True
    assert ud.filterstamp.endswith("_phospho")

    ud.acetyl = True
    assert ud.filterstamp.endswith("_phospho_acetyl")



def test_ensure_filter_defaults_populates_missing_values(tmp_path, UserData):
    ud = UserData(
        recno="1",
        datafile=None,
        indir=str(tmp_path),
        outdir=str(tmp_path / "out"),
    )
    ud.filtervalues = {"ion_score": 5}

    result = ud.ensure_filter_defaults()

    for key in ("qvalue", "pep", "idg", "zmin", "zmax", "modi", "ion_score_bins"):
        assert key in result
    assert result["ion_score"] == 5
