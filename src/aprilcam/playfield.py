from __future__ import annotations

import argparse
import math
from typing import Tuple

import cv2 as cv
import numpy as np


def _draw_marker_img(dictionary: cv.aruco.Dictionary, marker_id: int, side: int) -> np.ndarray:
    # Generate a crisp marker image using OpenCV ArUco/AprilTag
    if hasattr(cv.aruco, "generateImageMarker"):
        img = cv.aruco.generateImageMarker(dictionary, int(marker_id), int(side))
    else:
        img = cv.aruco.drawMarker(dictionary, int(marker_id), int(side))
    # Return RGB for pygame
    return cv.cvtColor(img, cv.COLOR_GRAY2RGB)


def _np_to_surface_rgb(arr_hwc_rgb: np.ndarray):
    import pygame

    # pygame.surfarray expects (W,H,3); convert from (H,W,3)
    return pygame.surfarray.make_surface(np.transpose(arr_hwc_rgb, (1, 0, 2)))


def _make_padded_marker_surface(dictionary: cv.aruco.Dictionary, marker_id: int, side: int, quiet_ratio: float = 2.0/7.0):
    """Create a pygame Surface of a marker with a white quiet zone around it.
    quiet_ratio is fraction of marker side added on each side (default 2/7 ≈ 0.2857).
    """
    import pygame

    side = int(side)
    # Compute white quiet zone padding in pixels per edge
    if float(quiet_ratio) <= 0.0:
        pad = 0
    else:
        pad = max(1, int(round(float(side) * float(quiet_ratio))))
    marker = _draw_marker_img(dictionary, int(marker_id), int(side))  # RGB
    h, w = marker.shape[:2]
    canvas = np.full((h + 2 * pad, w + 2 * pad, 3), 255, dtype=np.uint8)
    canvas[pad:pad + h, pad:pad + w] = marker
    return _np_to_surface_rgb(canvas), (w + 2 * pad, h + 2 * pad)


def place_fiducials(screen, width: int, height: int, fid_side: int, quiet_ratio: float = 2.0/7.0) -> None:
    import pygame

    # IDs follow CORNER_ID_MAP: 0=upper_left, 1=upper_right, 2=lower_left, 3=lower_right
    four = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    # Increase margin to account for quiet zone
    margin = 20
    # Padded size (approx) for centering math
    _surf, (pad_w, pad_h) = _make_padded_marker_surface(four, 0, int(fid_side), quiet_ratio)
    positions = {
        0: (margin + pad_w // 2, margin + pad_h // 2),
        1: (width - margin - pad_w // 2, margin + pad_h // 2),
        3: (width - margin - pad_w // 2, height - margin - pad_h // 2),
        2: (margin + pad_w // 2, height - margin - pad_h // 2),
    }
    for cid, center in positions.items():
        surf, _ = _make_padded_marker_surface(four, cid, int(fid_side), quiet_ratio)
        rect = surf.get_rect(center=center)
        screen.blit(surf, rect)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="playfield",
        description="Pygame window with 4x4 corner fiducials and three moving AprilTags (36h11)",
    )
    parser.add_argument("--width", type=int, default=1200, help="Window width in pixels (default 1200)")
    parser.add_argument("--height", type=int, default=800, help="Window height in pixels (default 800)")
    parser.add_argument("--tag-size", type=int, default=48, help="Moving AprilTag side length in pixels (default 48)")
    parser.add_argument("--fiducial-size", type=int, default=48, help="Corner fiducial side length in pixels (default 48)")
    parser.add_argument("--fps", type=float, default=60.0, help="Target frame rate (default 60)")
    parser.add_argument(
        "--quiet-ratio",
        type=float,
        default=2.0/7.0,
        help=(
            "White quiet zone thickness as a fraction of the tag side added on each edge. "
            "Example: 0.285714 (~2/7) for two tag modules. Use 0 to disable."
        ),
    )
    args = parser.parse_args(argv)

    # Lazy import pygame to avoid dependency on non-playfield workflows
    try:
        import pygame
    except Exception as e:
        print("This program requires pygame. Install with: pip install pygame")
        return 2

    pygame.init()
    screen = pygame.display.set_mode((int(args.width), int(args.height)))
    pygame.display.set_caption("AprilCam Playfield")
    clock = pygame.time.Clock()

    # Tag dictionaries
    apriltag_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_APRILTAG_36h11)

    # Pre-build moving tags (IDs 1, 2, 3) with quiet zone padding for robust detection when rotated
    tag_ids = [1, 2, 3]
    tag_surfs = []
    tag_dims = []
    for tid in tag_ids:
        # White quiet zone thickness controlled by --quiet-ratio (fraction of side per edge)
        surf, (pw, ph) = _make_padded_marker_surface(apriltag_dict, tid, int(args.tag_size), quiet_ratio=float(args.quiet_ratio))
        tag_surfs.append(surf)
        tag_dims.append((pw, ph))

    # Motion parameters
    W, H = int(args.width), int(args.height)
    side = int(args.tag_size)
    pad = max(10, int(0.6 * side))

    # Left-to-right: start near left edge, middle height
    lr_pos = [pad, H * 0.35]
    lr_v = [200.0, 0.0]

    # Up-to-down: start near top, quarter width
    ud_pos = [W * 0.25, pad]
    ud_v = [0.0, 200.0]

    # Circular: center and radius
    circ_center = (W * 0.65, H * 0.55)
    circ_r = min(W, H) * 0.20
    circ_ang = 0.0
    circ_w = math.radians(40.0)  # 40 deg/s

    running = True
    prev_t = pygame.time.get_ticks() / 1000.0
    # Global speed scale (modifiable via arrow keys)
    speed = 1.0

    while running:
        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_UP:
                    # Increase speed by 10%, cap to a reasonable max
                    speed = min(8.0, speed * 1.10)
                    print(f"Speed: {speed:.2f}x")
                elif event.key == pygame.K_DOWN:
                    # Decrease speed by ~9% (inverse of +10%), but not below 0.05x
                    speed = max(0.05, speed / 1.10)
                    print(f"Speed: {speed:.2f}x")

        # Timing
        now = pygame.time.get_ticks() / 1000.0
        dt = max(1e-3, min(0.1, now - prev_t))
        prev_t = now

        # Update positions
        lr_pos[0] += lr_v[0] * dt * speed
        if lr_pos[0] < pad or lr_pos[0] > W - pad:
            lr_v[0] = -lr_v[0]
            lr_pos[0] = max(pad, min(W - pad, lr_pos[0]))

        ud_pos[1] += ud_v[1] * dt * speed
        if ud_pos[1] < pad or ud_pos[1] > H - pad:
            ud_v[1] = -ud_v[1]
            ud_pos[1] = max(pad, min(H - pad, ud_pos[1]))

        circ_ang = (circ_ang + circ_w * dt * speed) % (2 * math.pi)
        cx = circ_center[0] + circ_r * math.cos(circ_ang)
        cy = circ_center[1] + circ_r * math.sin(circ_ang)

        # Draw
        screen.fill((24, 24, 24))
        # Fiducials in corners
        # Corner fiducials use the same quiet zone ratio
        place_fiducials(screen, W, H, int(args.fiducial_size), quiet_ratio=float(args.quiet_ratio))

        # Moving tags (rotate slightly for realism)
        for i, (pos, angle_deg) in enumerate([
                (lr_pos, 0.0),
                (ud_pos, 90.0),
                ((cx, cy), math.degrees(circ_ang)),
            ]):
            surf = tag_surfs[i]
            # Rotate; pygame rotates with interpolation, but the added white quiet zone keeps edges robust
            rotated = pygame.transform.rotozoom(surf, -angle_deg, 1.0)
            rect = rotated.get_rect(center=(int(pos[0]), int(pos[1])))
            screen.blit(rotated, rect)

        pygame.display.flip()
        clock.tick(float(args.fps))

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
