import pytest
import pandas as pd
import numpy as np
from collections import defaultdict
from unittest.mock import MagicMock

import gpgrouper.match as match

# Fixtures
@pytest.fixture
def sample_database():
    return pd.DataFrame(
        {
            "geneid": ["1", "2"],
            "sequence": ["PEPTIDE", "MPEPTIDEM"],
        }
    )

@pytest.fixture
def sample_usrdatas():
    """Creates a sample user data structure for testing."""
    class UserData:
        def __init__(self):
            self.df = pd.DataFrame({"Sequence": ["PEPTIDE", "MPEPTIDEM"]})
            self.searchdb = None

    return [UserData(), UserData()]

@pytest.fixture
def mock_load_fasta(monkeypatch, sample_database):
    def _mock_load_fasta(_refseq_file):
        return sample_database.copy()

    monkeypatch.setattr(match, "load_fasta", _mock_load_fasta)

@pytest.fixture
def mock_protease(monkeypatch):
    def _protease(sequence, minlen, rule, miscuts, semi_tryptic, semi_tryptic_iter):
        return [sequence], len(sequence)

    monkeypatch.setattr(match, "protease", _protease)


@pytest.fixture
def mock_calculate_breakup_size(monkeypatch):
    monkeypatch.setattr(match, "calculate_breakup_size", MagicMock(return_value=999))


@pytest.fixture
def noop_extract_peptideinfo(monkeypatch):
    monkeypatch.setattr(match, "extract_peptideinfo", MagicMock())

# Tests
def test_match_function(
    mock_load_fasta,
    mock_protease,
    mock_calculate_breakup_size,
    noop_extract_peptideinfo,
    sample_usrdatas,
):
    """Test that the _match function processes data correctly."""
    refseq_file = "mock_refseq_file.fasta"
    result = match.do_match(
        usrdatas=sample_usrdatas,
        refseq_file=refseq_file,
        miscuts=2,
        enzyme="trypsin/P",
        semi_tryptic=True,
        semi_tryptic_iter=3,
        min_pept_len=3,
        num_workers=1
    )
    # Ensure result is a DataFrame and has 'capacity' column populated
    assert isinstance(result, pd.DataFrame)
    assert "capacity" in result.columns
    assert not result["capacity"].isna().all(), "Capacity should not be entirely NaN."

    # Check if peptide matching was applied to user data
    for usrdata in sample_usrdatas:
        assert "metadatainfo" in usrdata.df.columns
        assert usrdata.df["metadatainfo"].str.endswith("|").all()
        assert usrdata.searchdb == refseq_file, "User data searchdb attribute should be set to refseq_file."

@pytest.mark.parametrize("enzyme, expected_rule, expected_miscuts", [
    ("trypsin/P", "[KR]", 2),
    ("noenzyme", ".", 50),
])
def test_enzyme_rule_application(
    enzyme,
    expected_rule,
    expected_miscuts,
    mock_load_fasta,
    mock_calculate_breakup_size,
    noop_extract_peptideinfo,
    monkeypatch,
    sample_usrdatas,
):
    protease_mock = MagicMock(return_value=([], 0))
    monkeypatch.setattr(match, "protease", protease_mock)

    match.do_match(
        usrdatas=sample_usrdatas,
        refseq_file="mock_refseq_file.fasta",
        miscuts=2,
        enzyme=enzyme,
        semi_tryptic=False,
        semi_tryptic_iter=6,
        min_pept_len=7,
        num_workers=1
    )

    assert protease_mock.call_args.kwargs["rule"] == expected_rule
    assert protease_mock.call_args.kwargs["miscuts"] == expected_miscuts


def test_batch_processing_logic(sample_usrdatas):
    fragment_mapping = defaultdict(list, {"PEPTIDE": [0], "MPEPTIDEM": [1]})
    match._process_batch(sample_usrdatas, fragment_mapping)

    for usrdata in sample_usrdatas:
        assert usrdata.df["metadatainfo"].tolist() == ["0|", "1|"]
