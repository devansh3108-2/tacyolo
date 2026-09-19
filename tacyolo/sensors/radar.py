from __future__ import annotations

from pathlib import Path

import numpy as np

from tacyolo.sensors.sim import SceneSimulator
from tacyolo.types import RadarPeak


class RadarSource:
    def next(self) -> dict:
        raise NotImplementedError

    def close(self) -> None:
        return None


class SyntheticRadar(RadarSource):
    """Range-Doppler simulator driven by a shared SceneSimulator when available."""

    def __init__(
        self,
        n_range: int = 128,
        n_doppler: int = 64,
        snr_db: float = 12.0,
        clutter_power: float = 8.0,
        n_targets: int = 2,
        seed: int = 1,
        scene: SceneSimulator | None = None,
        max_range: float = 420.0,
        max_doppler: float = 8.0,
    ) -> None:
        self.n_range = int(n_range)
        self.n_doppler = int(n_doppler)
        self.snr_db = float(snr_db)
        self.clutter_power = float(clutter_power)
        self.max_range = float(max_range)
        self.max_doppler = float(max_doppler)
        self.rng = np.random.default_rng(seed)
        self.scene = scene
        self._standalone = SceneSimulator(n_targets=n_targets, seed=seed) if scene is None else None
        self.t = 0

    def next(self) -> dict:
        if self._standalone is not None:
            self._standalone.step()
            truth = self._standalone.radar_truth()
        elif self.scene is not None:
            truth = self.scene.radar_truth()
        else:
            truth = []

        rd = self._render_rd(truth)
        iq = np.fft.ifft(rd, axis=0)
        self.t += 1
        return {
            "rd_map": np.abs(rd).astype(np.float32),
            "iq": iq.astype(np.complex64),
            "truth": truth,
            "peaks_gt": [
                RadarPeak(
                    range_bin=t["range"] / self.max_range * (self.n_range - 1),
                    doppler_bin=(t["doppler"] / (2 * self.max_doppler) + 0.5) * (self.n_doppler - 1),
                    energy=t["rcs"],
                    bearing=t["bearing"],
                )
                for t in truth
            ],
        }

    def _render_rd(self, truth: list[dict]) -> np.ndarray:
        rd = np.zeros((self.n_range, self.n_doppler), dtype=np.complex64)
        noise_std = 10 ** (-self.snr_db / 20.0)
        # Clutter ridge near zero Doppler.
        clutter_col = self.n_doppler // 2
        for r in range(self.n_range):
            spread = np.exp(-0.5 * ((np.arange(self.n_doppler) - clutter_col) / 2.2) ** 2)
            amp = self.clutter_power * (0.15 + 0.85 * (r / max(self.n_range - 1, 1)))
            phase = self.rng.uniform(0, 2 * np.pi, size=self.n_doppler)
            rd[r] += (amp * spread * np.exp(1j * phase)).astype(np.complex64)

        for t in truth:
            r_bin = np.clip(t["range"] / self.max_range * (self.n_range - 1), 0, self.n_range - 1)
            d_bin = np.clip(
                (t["doppler"] / (2 * self.max_doppler) + 0.5) * (self.n_doppler - 1),
                0,
                self.n_doppler - 1,
            )
            rr = int(round(r_bin))
            dd = int(round(d_bin))
            for dr in range(-2, 3):
                for ddlt in range(-2, 3):
                    r2 = rr + dr
                    d2 = dd + ddlt
                    if 0 <= r2 < self.n_range and 0 <= d2 < self.n_doppler:
                        w = np.exp(-0.5 * (dr * dr + ddlt * ddlt))
                        rd[r2, d2] += t["rcs"] * 6.0 * w * np.exp(1j * self.rng.uniform(0, 2 * np.pi))

        rd += (self.rng.normal(scale=noise_std, size=rd.shape) + 1j * self.rng.normal(scale=noise_std, size=rd.shape)).astype(
            np.complex64
        )
        return rd


class NpyRadar(RadarSource):
    """Load a recorded range-Doppler cube from .npy (T, R, D) or (R, D)."""

    def __init__(self, path: str | Path) -> None:
        cube = np.load(path)
        if cube.ndim == 2:
            cube = cube[None, ...]
        if cube.ndim != 3:
            raise ValueError("Radar npy must be (R,D) or (T,R,D)")
        self.cube = np.abs(cube).astype(np.float32)
        self.i = 0

    def next(self) -> dict:
        if self.i >= len(self.cube):
            raise StopIteration
        rd = self.cube[self.i]
        self.i += 1
        return {"rd_map": rd, "iq": rd.astype(np.complex64), "truth": [], "peaks_gt": []}


def make_radar(mode: str = "synthetic", scene: SceneSimulator | None = None, **kwargs) -> RadarSource:
    if mode in ("synthetic", "sim", None, ""):
        return SyntheticRadar(scene=scene, **kwargs)
    path = Path(str(mode))
    if path.suffix == ".npy" and path.exists():
        return NpyRadar(path)
    raise FileNotFoundError(f"Unknown radar source: {mode}")
