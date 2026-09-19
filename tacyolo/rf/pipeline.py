from __future__ import annotations

import time

import numpy as np

from tacyolo.rf.clutter import subspace_residual
from tacyolo.rf.compressed_sensing import recover_range_profile
from tacyolo.rf.scale_space import stable_peaks
from tacyolo.rf.tda import tda_summary
from tacyolo.types import RadarPeak, RFResult


class RFPipeline:
    def __init__(
        self,
        cs_lambda: float = 0.02,
        cs_iters: int = 40,
        cs_sample_frac: float = 0.35,
        gauss_sigmas: list[float] | None = None,
        clutter_rank: int = 3,
        tda_eps: float = 3.0,
        energy_gate: float = 2.5,
    ) -> None:
        self.cs_lambda = cs_lambda
        self.cs_iters = cs_iters
        self.cs_sample_frac = cs_sample_frac
        self.gauss_sigmas = gauss_sigmas or [1.0, 2.0, 4.0]
        self.clutter_rank = clutter_rank
        self.tda_eps = tda_eps
        self.energy_gate = energy_gate

    def cheap_energy(self, rd_map: np.ndarray) -> float:
        mag = np.abs(rd_map)
        return float(mag.mean() + mag.max() * 0.05)

    def process(self, rd_map: np.ndarray, full: bool = True) -> RFResult:
        t0 = time.perf_counter()
        mag = np.abs(rd_map).astype(np.float32)
        energy = float(mag.mean())
        if not full:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return RFResult(rd_map=mag, residual=mag, recovered=mag.mean(axis=1), energy=energy, elapsed_ms=elapsed)

        recovered = recover_range_profile(
            mag,
            sample_frac=self.cs_sample_frac,
            lam=self.cs_lambda,
            n_iter=self.cs_iters,
        )
        residual = subspace_residual(mag, rank=self.clutter_rank)
        peaks_raw = stable_peaks(residual, sigmas=self.gauss_sigmas)
        peaks = [
            RadarPeak(range_bin=float(r), doppler_bin=float(d), energy=float(s))
            for r, d, s in peaks_raw
        ]
        betti0, persistence = tda_summary(peaks_raw, eps=self.tda_eps)
        features = _rf_features(recovered, residual, peaks, betti0)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return RFResult(
            rd_map=mag,
            residual=residual,
            recovered=recovered,
            peaks=peaks,
            energy=float(residual.mean()),
            betti0=betti0,
            persistence=persistence,
            features=features,
            elapsed_ms=elapsed,
        )


def _rf_features(recovered: np.ndarray, residual: np.ndarray, peaks: list[RadarPeak], betti0: int) -> np.ndarray:
    peak_e = np.array([p.energy for p in peaks], dtype=np.float32) if peaks else np.zeros(1, np.float32)
    r_bins = np.array([p.range_bin for p in peaks], dtype=np.float32) if peaks else np.zeros(1, np.float32)
    d_bins = np.array([p.doppler_bin for p in peaks], dtype=np.float32) if peaks else np.zeros(1, np.float32)
    feat = np.array(
        [
            float(np.max(np.abs(recovered))) if recovered.size else 0.0,
            float(np.argmax(np.abs(recovered))) / max(len(recovered), 1) if recovered.size else 0.0,
            float(residual.mean()),
            float(residual.max()),
            float(len(peaks)),
            float(peak_e.max()),
            float(r_bins.mean() / max(residual.shape[0], 1)),
            float((d_bins.mean() / max(residual.shape[1], 1)) if residual.ndim == 2 else 0.0),
            float(betti0),
        ],
        dtype=np.float32,
    )
    return feat
