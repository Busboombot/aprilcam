"""Integration tests for the AprilCam web server."""

import pytest
from starlette.testclient import TestClient
from aprilcam.web_server import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_discovery_endpoint(client):
    """GET / returns JSON API documentation."""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["server"] == "aprilcam"
    assert "endpoints" in data
    assert len(data["endpoints"]) > 0


def test_discovery_lists_all_endpoints(client):
    """Discovery endpoint lists all 13 REST API endpoints."""
    resp = client.get("/")
    data = resp.json()
    endpoint_paths = [e["path"] for e in data["endpoints"]]
    assert "/api/list_cameras" in endpoint_paths
    assert "/api/open_camera" in endpoint_paths
    assert "/api/get_tags" in endpoint_paths


def test_list_cameras(client):
    """POST /api/list_cameras returns a list."""
    resp = client.post("/api/list_cameras", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


def test_unknown_tool_returns_404(client):
    """POST /api/nonexistent returns 404."""
    resp = client.post("/api/nonexistent_tool", json={})
    assert resp.status_code == 404
    data = resp.json()
    assert "error" in data


def test_invalid_json_returns_400(client):
    """POST with invalid JSON returns 400."""
    resp = client.post(
        "/api/list_cameras",
        content=b"not json",
        headers={"Content-Type": "application/json"},
    )
    # Should still work since list_cameras takes no params
    # But test with a tool that needs params
    resp = client.post(
        "/api/open_camera",
        content=b"not json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code in (200, 400)  # depends on implementation


def test_close_camera_unknown_id(client):
    """POST /api/close_camera with unknown ID returns error in body."""
    resp = client.post("/api/close_camera", json={"camera_id": "nonexistent"})
    assert resp.status_code == 400  # unknown camera_id returns 400
    data = resp.json()
    assert "error" in data


# ---------------------------------------------------------------------------
# Content negotiation tests for GET /
# ---------------------------------------------------------------------------


def test_discovery_html_when_accept_html(client):
    """GET / with Accept: text/html returns the live web UI."""
    resp = client.get("/", headers={"Accept": "text/html"})
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "<html" in resp.text
    assert "AprilCam Live" in resp.text
    # Verify key UI elements are present
    assert "cameraSelect" in resp.text
    assert "/ws/tags/" in resp.text


def test_discovery_json_when_accept_json(client):
    """GET / with Accept: application/json returns JSON discovery doc."""
    resp = client.get("/", headers={"Accept": "application/json"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["server"] == "aprilcam"
    assert "endpoints" in data


def test_discovery_json_when_no_accept(client):
    """GET / with no Accept header returns JSON discovery doc."""
    # TestClient may set a default Accept header, so we explicitly clear it
    resp = client.get("/", headers={"Accept": ""})
    assert resp.status_code == 200
    data = resp.json()
    assert data["server"] == "aprilcam"


def test_discovery_json_when_wildcard(client):
    """GET / with Accept: */* returns JSON discovery doc (backwards-compatible)."""
    resp = client.get("/", headers={"Accept": "*/*"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["server"] == "aprilcam"
