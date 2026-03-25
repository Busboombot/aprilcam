"""Integration tests for frame lifecycle MCP tools."""

import json
import os

import pytest

from aprilcam.mcp_server import (
    create_frame_from_image,
    frame_registry,
    get_frame_image,
    list_frames,
    release_frame,
    save_frame,
)

TEST_IMAGE = os.path.join(os.path.dirname(__file__), "data", "playfield_cam3.jpg")


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
# create_frame_from_image tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_frame_from_image():
    result = await create_frame_from_image(TEST_IMAGE)
    data = json.loads(result[0].text)
    assert "frame_id" in data
    assert data["frame_id"].startswith("frm_")
    assert data["source"] == f"file:{TEST_IMAGE}"


@pytest.mark.asyncio
async def test_create_frame_invalid_path():
    result = await create_frame_from_image("/nonexistent/path/image.jpg")
    data = json.loads(result[0].text)
    assert "error" in data
    assert "not found" in data["error"].lower()


# ---------------------------------------------------------------------------
# get_frame_image tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_frame_image_original():
    # Create a frame first
    create_result = await create_frame_from_image(TEST_IMAGE)
    frame_id = json.loads(create_result[0].text)["frame_id"]

    # Get the original stage
    result = await get_frame_image(frame_id, stage="original")
    assert len(result) > 0
    assert result[0].type == "image"


@pytest.mark.asyncio
async def test_get_frame_image_invalid_stage():
    create_result = await create_frame_from_image(TEST_IMAGE)
    frame_id = json.loads(create_result[0].text)["frame_id"]

    result = await get_frame_image(frame_id, stage="nonexistent")
    data = json.loads(result[0].text)
    assert "error" in data
    assert "invalid stage" in data["error"].lower()


@pytest.mark.asyncio
async def test_get_frame_image_missing_frame():
    result = await get_frame_image("frm_999")
    data = json.loads(result[0].text)
    assert "error" in data


# ---------------------------------------------------------------------------
# save_frame tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_frame(tmp_path):
    create_result = await create_frame_from_image(TEST_IMAGE)
    frame_id = json.loads(create_result[0].text)["frame_id"]

    output_dir = str(tmp_path / "saved_frame")
    result = await save_frame(frame_id, output_dir)
    data = json.loads(result[0].text)

    assert data["path"] == output_dir
    assert "original.jpg" in data["files"]
    assert "deskewed.jpg" in data["files"]
    assert "processed.jpg" in data["files"]
    assert "metadata.json" in data["files"]

    # Verify files exist on disk
    for fname in data["files"]:
        assert os.path.isfile(os.path.join(output_dir, fname))

    # Verify metadata content
    with open(os.path.join(output_dir, "metadata.json")) as f:
        meta = json.load(f)
    assert meta["frame_id"] == frame_id
    assert meta["source"] == f"file:{TEST_IMAGE}"


# ---------------------------------------------------------------------------
# release_frame tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_release_frame():
    create_result = await create_frame_from_image(TEST_IMAGE)
    frame_id = json.loads(create_result[0].text)["frame_id"]

    # Release the frame
    result = await release_frame(frame_id)
    data = json.loads(result[0].text)
    assert data["released"] is True
    assert data["frame_id"] == frame_id

    # Trying to get it should fail
    result = await get_frame_image(frame_id)
    data = json.loads(result[0].text)
    assert "error" in data


@pytest.mark.asyncio
async def test_release_frame_missing():
    result = await release_frame("frm_999")
    data = json.loads(result[0].text)
    assert "error" in data


# ---------------------------------------------------------------------------
# list_frames tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_frames():
    # Create two frames
    await create_frame_from_image(TEST_IMAGE)
    await create_frame_from_image(TEST_IMAGE)

    result = await list_frames()
    data = json.loads(result[0].text)
    assert isinstance(data, list)
    assert len(data) == 2

    # Each entry should have expected keys
    for entry in data:
        assert "frame_id" in entry
        assert "source" in entry
        assert "timestamp" in entry
        assert "operations_applied" in entry
        assert "is_deskewed" in entry


@pytest.mark.asyncio
async def test_list_frames_empty():
    result = await list_frames()
    data = json.loads(result[0].text)
    assert data == []
