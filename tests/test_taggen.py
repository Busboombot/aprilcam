"""Tests for the unified taggen module."""

import pytest
from pathlib import Path

from aprilcam.taggen import parse_ids, render_tag, generate_pdf


class TestParseIds:
    def test_single_id(self):
        assert parse_ids("5") == [5]

    def test_range(self):
        assert parse_ids("0-3") == [0, 1, 2, 3]

    def test_comma_separated(self):
        assert parse_ids("1,3,5") == [1, 3, 5]

    def test_mixed_ranges_and_singles(self):
        assert parse_ids("0-2,7,10-12") == [0, 1, 2, 7, 10, 11, 12]

    def test_duplicates_removed(self):
        assert parse_ids("1,1,2-3,3") == [1, 2, 3]

    def test_empty_string(self):
        assert parse_ids("") == []

    def test_whitespace_tolerance(self):
        assert parse_ids(" 1 - 3 , 5 ") == [1, 2, 3, 5]


class TestRenderTag:
    def test_apriltag_36h11(self):
        img = render_tag(0, family="36h11", size=100)
        assert img.shape == (100, 100)
        assert img.dtype.name == "uint8"

    def test_aruco_4x4(self):
        img = render_tag(0, family="aruco4x4", size=100)
        assert img.shape == (100, 100)
        assert img.dtype.name == "uint8"

    def test_unknown_family_raises(self):
        with pytest.raises(ValueError, match="Unknown family"):
            render_tag(0, family="bogus")


class TestGeneratePdf:
    def test_creates_pdf_file(self, tmp_path):
        out = tmp_path / "test.pdf"
        result = generate_pdf([0, 1, 2], family="36h11", size=100, output_path=out)
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_aruco_pdf(self, tmp_path):
        out = tmp_path / "aruco.pdf"
        result = generate_pdf([0, 1], family="aruco4x4", size=100, output_path=out)
        assert result == out
        assert out.exists()


class TestCliPngMode:
    def test_png_output(self, tmp_path):
        from aprilcam.taggen import main

        out_dir = tmp_path / "pngs"
        rc = main(["0-2", "--png", "-o", str(out_dir)])
        assert rc == 0
        pngs = list(out_dir.glob("*.png"))
        assert len(pngs) == 3

    def test_aruco_png_output(self, tmp_path):
        from aprilcam.taggen import main

        out_dir = tmp_path / "aruco_pngs"
        rc = main(["-A", "0-1", "--png", "-o", str(out_dir)])
        assert rc == 0
        pngs = list(out_dir.glob("*.png"))
        assert len(pngs) == 2
