"""3D BBox Kalman filter smoothing for tracked objects."""

import numpy as np


class BBox3DKalmanFilter:
    """Kalman filter for smoothing 3D bounding box trajectories.

    State: [cx, cy, cz, dx, dy, dz, vx, vy, vz] (9D)
        - cx, cy, cz: 3D center in camera coordinates
        - dx, dy, dz: 3D dimensions (width, height, length)
        - vx, vy, vz: velocity of center

    Observation: [cx, cy, cz, dx, dy, dz] (6D)
    """

    def __init__(
        self,
        process_noise_pos=0.5,
        process_noise_dim=0.1,
        process_noise_vel=1.0,
        measurement_noise_pos=1.0,
        measurement_noise_dim=0.5,
        dt=1.0 / 30,
    ):
        self.dt = dt
        self.dim_x = 9  # state dimension
        self.dim_z = 6  # observation dimension

        # State transition matrix (constant velocity model)
        self.F = np.eye(self.dim_x)
        self.F[0, 6] = dt  # cx += vx * dt
        self.F[1, 7] = dt  # cy += vy * dt
        self.F[2, 8] = dt  # cz += vz * dt

        # Observation matrix (observe position + dimensions)
        self.H = np.zeros((self.dim_z, self.dim_x))
        self.H[:6, :6] = np.eye(6)

        # Process noise
        self.Q = np.diag([
            process_noise_pos,
            process_noise_pos,
            process_noise_pos,
            process_noise_dim,
            process_noise_dim,
            process_noise_dim,
            process_noise_vel,
            process_noise_vel,
            process_noise_vel,
        ]) ** 2

        # Measurement noise
        self.R = np.diag([
            measurement_noise_pos,
            measurement_noise_pos,
            measurement_noise_pos,
            measurement_noise_dim,
            measurement_noise_dim,
            measurement_noise_dim,
        ]) ** 2

        # State and covariance (initialized on first observation)
        self.x = None
        self.P = None
        self.initialized = False

    def init_state(self, observation):
        """Initialize state from first observation.

        Args:
            observation: numpy array (6,) [cx, cy, cz, dx, dy, dz]
        """
        self.x = np.zeros(self.dim_x)
        self.x[:6] = observation
        # velocity initialized to zero

        self.P = np.eye(self.dim_x)
        self.P[:6, :6] *= 1.0   # moderate uncertainty on position/dims
        self.P[6:, 6:] *= 10.0  # high uncertainty on velocity
        self.initialized = True

    def predict(self):
        """Predict step (no observation available)."""
        if not self.initialized:
            return
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, observation):
        """Update step with observation.

        Args:
            observation: numpy array (6,) [cx, cy, cz, dx, dy, dz]
        """
        if not self.initialized:
            self.init_state(observation)
            return

        # Predict first
        self.predict()

        # Innovation
        y = observation - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update
        self.x = self.x + K @ y
        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H) @ self.P

    def get_state(self):
        """Get current smoothed state.

        Returns:
            numpy array (6,) [cx, cy, cz, dx, dy, dz] or None
        """
        if not self.initialized:
            return None
        return self.x[:6].copy()


def smooth_rotation(rotations, alpha=0.5):
    """Smooth rotation values using exponential moving average.

    For the 4-value rotation output from SAM3_3D, we use simple
    component-wise EMA since the rotation representation is continuous.

    Args:
        rotations: list of numpy arrays (4,) or None for missing frames
        alpha: smoothing factor (0=no smoothing, 1=full previous)

    Returns:
        List of smoothed numpy arrays (4,) or None
    """
    smoothed = []
    prev = None

    for rot in rotations:
        if rot is None:
            smoothed.append(prev.copy() if prev is not None else None)
            continue

        if prev is None:
            prev = rot.copy()
            smoothed.append(rot.copy())
            continue

        # EMA smoothing
        smoothed_rot = alpha * prev + (1 - alpha) * rot
        # Normalize if quaternion-like
        norm = np.linalg.norm(smoothed_rot)
        if norm > 0:
            smoothed_rot = smoothed_rot / norm * np.linalg.norm(rot)

        prev = smoothed_rot.copy()
        smoothed.append(smoothed_rot)

    return smoothed


def smooth_tracks(tracked_results, n_frames, kf_params=None):
    """Apply Kalman filter smoothing to all tracks.

    Args:
        tracked_results: list[n_frames] of list[detections_per_frame],
            where each detection is a dict with track_id, box_3d, etc.
        n_frames: total number of frames
        kf_params: dict of Kalman filter parameters (optional)

    Returns:
        Dict mapping track_id -> list[n_frames] of smoothed box_3d (10,)
            or None for frames where track is not visible.
    """
    if kf_params is None:
        from . import config
        kf_params = {
            "process_noise_pos": config.KF_PROCESS_NOISE_POS,
            "process_noise_dim": config.KF_PROCESS_NOISE_DIM,
            "process_noise_vel": config.KF_PROCESS_NOISE_VEL,
            "measurement_noise_pos": config.KF_MEASUREMENT_NOISE_POS,
            "measurement_noise_dim": config.KF_MEASUREMENT_NOISE_DIM,
        }

    # Collect per-track observations across frames
    track_observations = {}  # track_id -> list[(frame_idx, box_3d)]
    for frame_idx, frame_dets in enumerate(tracked_results):
        for det in frame_dets:
            tid = det["track_id"]
            if tid not in track_observations:
                track_observations[tid] = []
            track_observations[tid].append((frame_idx, det["box_3d"]))

    # Apply KF per track
    smoothed = {}

    for track_id, observations in track_observations.items():
        kf = BBox3DKalmanFilter(
            process_noise_pos=kf_params["process_noise_pos"],
            process_noise_dim=kf_params["process_noise_dim"],
            process_noise_vel=kf_params["process_noise_vel"],
            measurement_noise_pos=kf_params["measurement_noise_pos"],
            measurement_noise_dim=kf_params["measurement_noise_dim"],
        )

        # Build frame -> box_3d mapping
        frame_to_box = {fi: b3d for fi, b3d in observations}

        # Collect raw rotations for smoothing
        raw_rotations = []
        frame_indices_with_obs = []

        track_frames = [None] * n_frames

        # Find first and last frame with observation
        first_frame = observations[0][0]
        last_frame = observations[-1][0]

        for frame_idx in range(first_frame, last_frame + 1):
            if frame_idx in frame_to_box:
                box_3d = frame_to_box[frame_idx]
                obs = box_3d[:6]  # cx, cy, cz, dx, dy, dz
                kf.update(obs)
                raw_rotations.append(box_3d[6:10])
                frame_indices_with_obs.append(frame_idx)
            else:
                kf.predict()
                raw_rotations.append(None)
                frame_indices_with_obs.append(frame_idx)

            state = kf.get_state()
            if state is not None:
                track_frames[frame_idx] = state

        # Smooth rotations
        smoothed_rots = smooth_rotation(
            raw_rotations,
            alpha=kf_params.get(
                "rotation_smooth_alpha",
                0.5,
            ),
        )

        # Combine smoothed position/dims with smoothed rotation
        rot_idx = 0
        for frame_idx in range(first_frame, last_frame + 1):
            if track_frames[frame_idx] is not None:
                rot = smoothed_rots[rot_idx]
                if rot is not None:
                    box_3d_smoothed = np.zeros(10)
                    box_3d_smoothed[:6] = track_frames[frame_idx]
                    box_3d_smoothed[6:10] = rot
                    track_frames[frame_idx] = box_3d_smoothed
                else:
                    track_frames[frame_idx] = None
            rot_idx += 1

        smoothed[track_id] = track_frames

    return smoothed
