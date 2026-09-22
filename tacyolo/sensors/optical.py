from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from tacyolo.sensors.sim import SceneSimulator
from tacyolo.sensors.sync import HardwareTimeSource, StampedPacket, SyncMode


class OpticalSource:
    """File, webcam index, image folder, numpy frames, or the coupled scene simulator."""

    def __init__(
        self,
        source: str | int | Path | np.ndarray | list[np.ndarray] | SceneSimulator = "synthetic",
        loop: bool = False,
        width: int = 640,
        height: int = 360,
        n_targets: int = 2,
        scene: SceneSimulator | None = None,
    ) -> None:
        self.loop = loop
        self._index = 0
        self._cap: cv2.VideoCapture | None = None
        self._frames: list[np.ndarray] | None = None
        self.scene = scene
        self.fps = 30.0
        self.step_scene = False

        if isinstance(source, SceneSimulator):
            self.scene = source
            self.step_scene = True
            return

        if isinstance(source, np.ndarray):
            if source.ndim == 4:
                self._frames = [np.ascontiguousarray(f) for f in source]
            elif source.ndim == 3:
                self._frames = [np.ascontiguousarray(source)]
            else:
                raise ValueError("numpy source must be HxWxC or NxHxWxC")
            return

        if isinstance(source, list):
            self._frames = [np.ascontiguousarray(f) for f in source]
            return

        if source in ("synthetic", "sim", None):
            self.scene = scene or SceneSimulator(width=width, height=height, n_targets=n_targets)
            self.step_scene = True
            return

        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            self._cap = cv2.VideoCapture(int(source))
            if not self._cap.isOpened():
                raise FileNotFoundError(f"Cannot open camera index {source}")
            self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
            return

        path = Path(str(source))
        if path.is_dir():
            files = sorted(
                p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            )
            if not files:
                raise FileNotFoundError(f"No images in {path}")
            self._frames = [cv2.imread(str(p)) for p in files]
            self._frames = [f for f in self._frames if f is not None]
            return

        if not path.exists():
            raise FileNotFoundError(f"Optical source not found: {path}")
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video {path}")
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    def __iter__(self) -> Iterator[np.ndarray]:
        return self

    def __next__(self) -> np.ndarray:
        frame = self.read()
        if frame is None:
            raise StopIteration
        return frame

    def read(self) -> np.ndarray | None:
        if self.scene is not None:
            if self.step_scene:
                self.scene.step()
            frame = self.scene.render_frame()
            self._index += 1
            return frame

        if self._frames is not None:
            if self._index >= len(self._frames):
                if not self.loop:
                    return None
                self._index = 0
            frame = self._frames[self._index]
            self._index += 1
            return frame.copy()

        if self._cap is not None:
            ok, frame = self._cap.read()
            if not ok:
                if not self.loop:
                    return None
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
                if not ok:
                    return None
            self._index += 1
            return frame
        return None

    def read_stamped(self, time_source: HardwareTimeSource | None = None) -> StampedPacket[np.ndarray] | None:
        frame = self.read()
        if frame is None:
            return None
        clock = time_source or getattr(self, "_clock", None)
        if clock is None:
            self._clock = HardwareTimeSource()
            clock = self._clock
        ts_ns = clock.now_ns()
        return StampedPacket(
            sensor_id="optical",
            seq=self._index,
            timestamp_ns=ts_ns,
            data=frame,
            pps_locked=(clock.mode == SyncMode.PPS),
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

