# test_coverage.py
import pytest

from gpgrouper.gpgrouper import _calc_coverage

# Define a series of tests
@pytest.mark.parametrize("seqs, pepts, expected", [
    (["ACDEFGHIKLMNPQRSTVWY", "ABCDEFGHIJKLMN", "QRSTVWXYZ"], ["CDEF", "IJKL", "QRST", "XYZ"], 0.583),
    (["AAAA", "BBBB", "CCCC"], ["AAA", "BBB", "CCC"], 0.75),
    (["AAAA", "BBBB", "CCCC"], ["DDD", "EEE"], 0),
    (["ACACAC", "BBBBBB"], ["ACA", "BBB"], 1.0),
    (["XYZ"], ["XYZ", "XYZ"], 1.0),
    (["ABCDEFG"], ["XYZ"], 0)
])
def test_calc_coverage(seqs, pepts, expected):
    result = _calc_coverage(seqs, pepts)
    assert pytest.approx(result) == expected, "Test failed for input: {}, {}, expected: {}, got: {}".format(seqs, pepts, expected, result)

# Test for checking error or warning handling (if applicable)
def test_no_coverage_warning():
    seqs = ["ABCDEFG"]
    pepts = ["XYZ"]
    with pytest.raises(Exception):  # Change Exception to your specific expected exception or use pytest.warns if a warning is expected
        _calc_coverage(seqs, pepts)
