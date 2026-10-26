import pytest

from detector.dataset import parse_label_file, write_label_file


def label(tmp_path, text):
    p = tmp_path / "x.txt"
    p.write_text(text, encoding="utf-8")
    return p


def test_valid_label(tmp_path):
    r = parse_label_file(label(tmp_path, "0 0.5 0.5 0.4 0.3\n"))
    assert not r.issues and r.boxes == [(0, 0.5, 0.5, 0.4, 0.3)]


def test_empty_file_is_valid_negative(tmp_path):
    r = parse_label_file(label(tmp_path, ""))
    assert not r.issues and r.boxes == []


@pytest.mark.parametrize("text,fragment", [
    ("0 0.5 0.5 0.4\n", "expected 5 values"),
    ("0 a 0.5 0.4 0.3\n", "non-numeric"),
    ("1 0.5 0.5 0.4 0.3\n", "invalid class id"),
    ("0.5 0.5 0.5 0.4 0.3\n", "invalid class id"),
    ("-1 0.5 0.5 0.4 0.3\n", "invalid class id"),
    ("0 1.2 0.5 0.4 0.3\n", "outside [0,1]"),
    ("0 0.5 0.5 0.0 0.3\n", "non-positive"),
    ("0 0.5 0.5 0.4 -0.1\n", "outside [0,1]"),
    ("0 0.95 0.5 0.4 0.3\n", "beyond the image"),
    ("0 nan 0.5 0.4 0.3\n", "NaN"),
])
def test_invalid_labels_reported(tmp_path, text, fragment):
    r = parse_label_file(label(tmp_path, text))
    assert r.issues and r.boxes == []
    assert fragment in r.issues[0]


def test_good_lines_survive_next_to_bad_ones(tmp_path):
    r = parse_label_file(label(tmp_path, "0 0.5 0.5 0.4 0.3\n0 2 2 2 2\n"))
    assert len(r.boxes) == 1 and len(r.issues) == 1


def test_unreadable_label(tmp_path):
    r = parse_label_file(tmp_path / "missing.txt")
    assert r.issues


def test_write_roundtrip(tmp_path):
    p = tmp_path / "sub" / "a.txt"
    write_label_file(p, [(0, 0.5, 0.5, 0.2, 0.2)])
    assert parse_label_file(p).boxes == [(0, 0.5, 0.5, 0.2, 0.2)]
    write_label_file(p, [])
    assert p.read_text() == ""
