---
status: pending
---

# Image Model & Processing Refactor

Major refactoring to introduce a robust image model and cleanly separate
concerns between image data, image processing, camera operations, and
playfield tracking.

## Image Model

- Create an `ImageFrame` class that represents a single captured image
- Holds the raw image data (NumPy array)
- Holds metadata: source, timestamp, resolution, capture parameters
- Holds homography matrix and calibration metadata (if applicable)
- Holds a transformed/processed image variant with metadata describing
  what transformations were applied
- Stores detected ArUco corner tag locations (fiducial/homography tags)
- Stores a list of all detected AprilTags

## AprilTag Model Improvements

- AprilTag class includes: family, ID number, pixel location, orientation,
  and velocity
- Clean separation between per-frame detection data and tracked flow data

## Processing Functions Operate on NumPy Arrays

- All basic image processing functions (detect_contours, detect_lines,
  detect_circles, etc.) take a NumPy array as input, not a camera handle
- Higher-level functions operate on the ImageFrame model, transforming
  one image to another or adding metadata
- This enables testing with static images — no camera required

## Camera Separation

- Camera operations are fully separated from image processing
- You can take an ImageFrame and run all processing on a static image
  from any source (file, camera, screen capture)
- Testing uses static test images, not live cameras

## Playfield Responsibilities

- Playfield maintains tag flow over time (history of tag positions)
- Playfield calculates tag velocities from position history — images
  themselves do not compute velocities
- ArUco corner detection for playfield creation does not need to happen
  every frame — corners are assumed stationary, camera is assumed fixed
- Playfield creation is a one-time setup step

## Testing Strategy

- Use existing test images (playfield_cam3.jpg, playfield_cam3_moved.jpg)
  for initial testing
- Record additional test data with moving tags for velocity/flow testing
- End-to-end tests use recorded data with fixed and moving tags
- All detection and processing testable without a live camera
