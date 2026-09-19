"""Shared 2D scene used to generate optical frames and synthetic radar."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class SimObject:
    obj_id: int
    x: float
    y: float
    vx: float
    vy: float
    w: float
    h: float
    color: tuple[int, int, int]
    label: str
    rcs: float = 1.0


@dataclass
class SceneSimulator:
    width: int = 640
    height: int = 360
    n_targets: int = 2
    seed: int = 0
    objects: list[SimObject] = field(default_factory=list)
    t: int = 0
    radar_origin: tuple[float, float] = (320.0, 360.0)

    def __post_init__(self) -> None:
        if not self.objects:
            rng = np.random.default_rng(self.seed)
            palette = [
                ((40, 40, 220), "red"),
                ((220, 40, 40), "blue"),
                ((40, 180, 40), "green"),
            ]
            count = max(1, int(self.n_targets))
            for i in range(count):
                color, label = palette[i % len(palette)]
                self.objects.append(
                    SimObject(
                        obj_id=i + 1,
                        x=float(80 + i * 180 + rng.integers(0, 40)),
                        y=float(80 + rng.integers(0, 80)),
                        vx=float(rng.choice([-1, 1]) * rng.uniform(2.2, 4.0)),
                        vy=float(rng.choice([-1, 1]) * rng.uniform(1.2, 2.6)),
                        w=48.0,
                        h=36.0,
                        color=color,
                        label=label,
                        rcs=float(0.8 + 0.4 * i),
                    )
                )

    def step(self) -> None:
        self.t += 1
        for obj in self.objects:
            obj.x += obj.vx
            obj.y += obj.vy
            if obj.x < obj.w * 0.5 or obj.x > self.width - obj.w * 0.5:
                obj.vx *= -1
                obj.x = float(np.clip(obj.x, obj.w * 0.5, self.width - obj.w * 0.5))
            if obj.y < obj.h * 0.5 or obj.y > self.height - obj.h * 0.5:
                obj.vy *= -1
                obj.y = float(np.clip(obj.y, obj.h * 0.5, self.height - obj.h * 0.5))

    def render_frame(self, draw_objects: bool = True) -> np.ndarray:
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = (18, 18, 22)
        cv2.rectangle(frame, (0, self.height - 18), (self.width, self.height), (32, 32, 38), -1)
        if draw_objects:
            for obj in self.objects:
                x1 = int(obj.x - obj.w * 0.5)
                y1 = int(obj.y - obj.h * 0.5)
                x2 = int(obj.x + obj.w * 0.5)
                y2 = int(obj.y + obj.h * 0.5)
                cv2.rectangle(frame, (x1, y1), (x2, y2), obj.color, -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (240, 240, 240), 1)
        return frame

    def truth_boxes(self) -> list[dict]:
        out = []
        for obj in self.objects:
            out.append(
                {
                    "id": obj.obj_id,
                    "label": obj.label,
                    "bbox": (
                        obj.x - obj.w * 0.5,
                        obj.y - obj.h * 0.5,
                        obj.x + obj.w * 0.5,
                        obj.y + obj.h * 0.5,
                    ),
                    "vx": obj.vx,
                    "vy": obj.vy,
                    "rcs": obj.rcs,
                }
            )
        return out

    def radar_truth(self) -> list[dict]:
        ox, oy = self.radar_origin
        out = []
        for obj in self.objects:
            dx = obj.x - ox
            dy = oy - obj.y
            rng = float(np.hypot(dx, dy))
            bearing = float(np.arctan2(dx, dy))
            radial_v = float((obj.vx * dx + (-obj.vy) * dy) / (rng + 1e-6))
            out.append(
                {
                    "id": obj.obj_id,
                    "label": obj.label,
                    "range": rng,
                    "bearing": bearing,
                    "doppler": radial_v,
                    "rcs": obj.rcs,
                    "x": obj.x,
                    "y": obj.y,
                }
            )
        return out
