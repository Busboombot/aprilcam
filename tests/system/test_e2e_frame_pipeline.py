"""End-to-end integration tests for the frame pipeline.

These tests exercise the full pipeline across multiple MCP tools in
sequence, using the static test image at ``tests/data/playfield_cam3.jpg``.
"""

import json
import os

import pytest

from aprilcam.mcp_server import (
    create_frame_from_image,
    frame_registry,
    get_frame_image,
    list_frames,
    process_frame,
    release_frame,
    save_frame,
)

TEST_IMAGE = os.path.join(
    os.path.dirname(__file__), os.pardir, "data", "playfield_cam3.jpg"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_frame_registry():
    """Ensure the frame registry is empty before and after each test."""
    frame_registry.clear()
    yield
    frame_registry.clear()


# ---------------------------------------------------------------------------
# Test 1: Full workflow — load, detect, inspect, save, release
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_load_detect_inspect_save(tmp_path):
    # 1. Create frame from image
    result = await create_frame_from_image(TEST_IMAGE)
    data = json.loads(result[0].text)
    assert "frame_id" in data
    frame_id = data["frame_id"]
    assert frame_id.startswith("frm_")

    # 2. Process with multiple operations
    result = await process_frame(
        frame_id, ["detect_aruco", "detect_tags", "detect_lines"]
    )
    data = json.loads(result[0].text)
    assert data["frame_id"] == frame_id
    results = data["results"]

    # 3. Verify detect_aruco found ArUco markers (IDs 0-3)
    aruco = results["detect_aruco"]
    assert isinstance(aruco, list)
    assert len(aruco) >= 1
    found_ids = {m["id"] for m in aruco}
    assert len(found_ids & {0, 1, 2, 3}) >= 1

    # 4. Verify detect_tags returned results
    tags = results["detect_tags"]
    assert isinstance(tags, list)

    # 5. Verify detect_lines returned results
    lines = results["detect_lines"]
    assert isinstance(lines, list)

    # 6. Get original frame image
    result = await get_frame_image(frame_id, stage="original")
    assert len(result) > 0
    assert result[0].type == "image"

    # 7. Save frame to disk
    output_dir = str(tmp_path / "e2e_output")
    result = await save_frame(frame_id, output_dir)
    data = json.loads(result[0].text)
    assert data["path"] == output_dir
    assert "original.jpg" in data["files"]
    assert "metadata.json" in data["files"]

    # 8. Verify metadata.json has correct fields
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    assert meta["frame_id"] == frame_id
    assert meta["source"] == f"file:{TEST_IMAGE}"
    assert "operations_applied" in meta
    assert "detect_aruco" in meta["operations_applied"]
    assert "detect_tags" in meta["operations_applied"]
    assert "detect_lines" in meta["operations_applied"]
    assert "timestamp" in meta

    # 9. Release the frame
    result = await release_frame(frame_id)
    data = json.loads(result[0].text)
    assert data["released"] is True
    assert data["frame_id"] == frame_id

    # 10. Verify frame is truly gone
    result = await get_frame_image(frame_id)
    data = json.loads(result[0].text)
    assert "error" in data


# ---------------------------------------------------------------------------
# Test 2: Create with inline operations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_create_with_operations():
    # 1. Create frame with operations run inline
    result = await create_frame_from_image(
        TEST_IMAGE, operations=["detect_aruco", "detect_lines"]
    )
    data = json.loads(result[0].text)

    # 2. Verify results returned inline with frame_id
    assert "frame_id" in data
    frame_id = data["frame_id"]
    assert "results" in data
    assert "detect_aruco" in data["results"]
    assert "detect_lines" in data["results"]

    # 3. Verify frame is in registry and has operations_applied set
    entry = frame_registry.get(frame_id)
    assert "detect_aruco" in entry.operations_applied
    assert "detect_lines" in entry.operations_applied


# ---------------------------------------------------------------------------
# Test 3: Multiple frames in ring buffer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_multiple_frames_in_ring_buffer():
    # 1. Create 5 frames from the test image
    frame_ids = []
    for _ in range(5):
        result = await create_frame_from_image(TEST_IMAGE)
        data = json.loads(result[0].text)
        frame_ids.append(data["frame_id"])

    # 2. List frames — verify 5 frames listed
    result = await list_frames()
    data = json.loads(result[0].text)
    assert isinstance(data, list)
    assert len(data) == 5

    listed_ids = {entry["frame_id"] for entry in data}
    for fid in frame_ids:
        assert fid in listed_ids

    # 3. Access frame 2 and frame 4 by ID
    entry2 = frame_registry.get(frame_ids[1])
    entry4 = frame_registry.get(frame_ids[3])
    assert entry2.frame_id == frame_ids[1]
    assert entry4.frame_id == frame_ids[3]

    # 4. Process frame 2 with detect_lines
    result = await process_frame(frame_ids[1], ["detect_lines"])
    data2 = json.loads(result[0].text)
    assert data2["frame_id"] == frame_ids[1]
    assert "detect_lines" in data2["results"]

    # 5. Process frame 4 with detect_contours
    result = await process_frame(frame_ids[3], ["detect_contours"])
    data4 = json.loads(result[0].text)
    assert data4["frame_id"] == frame_ids[3]
    assert "detect_contours" in data4["results"]

    # 6. Verify both frames have different operations_applied
    assert entry2.operations_applied == ["detect_lines"]
    assert entry4.operations_applied == ["detect_contours"]

    # And different results keys
    assert "detect_lines" in entry2.results
    assert "detect_lines" not in entry4.results
    assert "detect_contours" in entry4.results
    assert "detect_contours" not in entry2.results


# ---------------------------------------------------------------------------
# Test 4: Family field in tag detection results
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_family_field_in_detection_results():
    # 1. Create frame from test image
    result = await create_frame_from_image(TEST_IMAGE)
    data = json.loads(result[0].text)
    frame_id = data["frame_id"]

    # 2. Process with detect_tags
    result = await process_frame(frame_id, ["detect_tags"])
    data = json.loads(result[0].text)
    tags = data["results"]["detect_tags"]
    assert isinstance(tags, list)

    # 3-4. Verify each tag has a family field with a known value
    # The test image may or may not have AprilTags; if it does,
    # verify the family field is present and valid.
    if len(tags) > 0:
        known_families = {"36h11", "25h9", "16h5", "tag36h11", "tag25h9", "tag16h5"}
        for tag in tags:
            assert "family" in tag, f"Tag {tag.get('id')} missing 'family' field"
            assert tag["family"] in known_families, (
                f"Tag {tag.get('id')} has unexpected family '{tag['family']}'"
            )

    # Also verify via ArUco detection that the image has detectable markers
    result = await process_frame(frame_id, ["detect_aruco"])
    data = json.loads(result[0].text)
    aruco = data["results"]["detect_aruco"]
    assert len(aruco) >= 1, "Test image should contain ArUco markers"


# ---------------------------------------------------------------------------
# Test 5: Save frame — all stages present
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_save_frame_all_stages(tmp_path):
    # 1. Create frame from test image
    result = await create_frame_from_image(TEST_IMAGE)
    data = json.loads(result[0].text)
    frame_id = data["frame_id"]

    # 2. Save it
    output_dir = str(tmp_path / "all_stages")
    result = await save_frame(frame_id, output_dir)
    data = json.loads(result[0].text)

    expected_files = ["original.jpg", "deskewed.jpg", "processed.jpg", "metadata.json"]
    for fname in expected_files:
        assert fname in data["files"], f"Missing {fname} in save output"

    # 3. Verify original.jpg exists
    assert os.path.isfile(os.path.join(output_dir, "original.jpg"))

    # 4. Verify deskewed.jpg exists (same as original since no deskew applied)
    assert os.path.isfile(os.path.join(output_dir, "deskewed.jpg"))

    # 5. Verify processed.jpg exists
    assert os.path.isfile(os.path.join(output_dir, "processed.jpg"))

    # 6. Verify metadata.json is valid JSON with expected keys
    meta_path = os.path.join(output_dir, "metadata.json")
    assert os.path.isfile(meta_path)
    with open(meta_path) as f:
        meta = json.load(f)

    assert "frame_id" in meta
    assert meta["frame_id"] == frame_id
    assert "source" in meta
    assert "timestamp" in meta
    assert "operations_applied" in meta
    assert "is_deskewed" in meta
    assert "results" in meta

    # Since no operations were run, operations_applied should be empty
    assert meta["operations_applied"] == []
    # No deskew was applied
    assert meta["is_deskewed"] is False
