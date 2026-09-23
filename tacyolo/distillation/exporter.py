"""Stage 5: Zero-Overhead Edge Export & Latency Benchmarking.

Exports the trained student model to TensorRT FP16 engine format for NVIDIA Jetson
deployments and provides a high-precision CUDA-event latency benchmarking suite
measuring Mean FPS, P95, and P99 latency across warm iterations.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class LatencyBenchmarkResult:
    """Latency and throughput metrics from warm benchmark iterations."""
    model_path: Path
    device: str
    iterations: int
    mean_latency_ms: float
    median_latency_ms: float
    p90_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    fps: float

    def summary_table(self) -> str:
        """Renders formatted metrics table."""
        return (
            f"| Metric | Value |\n"
            f"|:---|:---|\n"
            f"| **Model** | `{self.model_path.name}` |\n"
            f"| **Device** | `{self.device}` |\n"
            f"| **Iterations** | {self.iterations} |\n"
            f"| **Mean Latency** | **{self.mean_latency_ms:.2f} ms** |\n"
            f"| **Median (P50)** | {self.median_latency_ms:.2f} ms |\n"
            f"| **P95 Latency** | {self.p95_latency_ms:.2f} ms |\n"
            f"| **P99 Latency** | **{self.p99_latency_ms:.2f} ms** |\n"
            f"| **Mean Throughput** | **{self.fps:.1f} FPS** |\n"
        )


class EdgeModelExporter:
    """Exports student YOLO models to TensorRT FP16 and benchmarks edge inference latency."""

    def __init__(self, device: str | None = None) -> None:
        if device is not None:
            self.device = str(device)
        else:
            self.device = "0" if torch.cuda.is_available() else "cpu"

    def export_tensorrt(
        self,
        weights_path: str | Path,
        imgsz: int = 640,
        batch_size: int = 1,
        half: bool = True,
        workspace_gb: int = 4,
    ) -> Path:
        """Exports YOLO weights to TensorRT .engine with FP16 half-precision."""
        src_path = Path(weights_path).resolve()
        if not src_path.exists():
            raise FileNotFoundError(f"Source weights not found: {src_path}")

        print(f"📦 Exporting {src_path.name} to TensorRT FP16 (batch={batch_size}, imgsz={imgsz})...")
        model = YOLO(str(src_path))

        try:
            exported_path_str = model.export(
                format="engine",
                imgsz=imgsz,
                half=half,
                batch=batch_size,
                device=self.device,
                workspace=workspace_gb,
                dynamic=False,
                verbose=True,
            )
            exported_path = Path(exported_path_str)
            print(f"⚡ TensorRT Engine Export Success: {exported_path}")
            return exported_path

        except Exception as err:
            print(f"⚠️ Direct TensorRT engine export encountered: {err}")
            print(f"🔄 Generating optimized ONNX model as fallback for Jetson trtexec compilation...")
            onnx_path_str = model.export(
                format="onnx",
                imgsz=imgsz,
                half=half,
                batch=batch_size,
                device=self.device,
                dynamic=False,
                simplify=True,
            )
            onnx_path = Path(onnx_path_str)
            engine_target = onnx_path.with_suffix(".engine")
            print(f"📄 ONNX Fallback Exported: {onnx_path}")
            print(f"💡 To compile on NVIDIA Jetson, run:\n"
                  f"   trtexec --onnx={onnx_path} --saveEngine={engine_target} --fp16 --memPoolSize=workspace:{workspace_gb*1024}MiB")
            return onnx_path

    def benchmark_latency(
        self,
        model_path: str | Path,
        imgsz: int = 640,
        batch_size: int = 1,
        warmup_iters: int = 20,
        bench_iters: int = 100,
    ) -> LatencyBenchmarkResult:
        """Benchmarks inference latency using high-precision CUDA events across 100 warm iterations."""
        path = Path(model_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        print(f"⏱️ Benchmarking {path.name} ({warmup_iters} warmup, {bench_iters} timed iterations) on {self.device}...")
        model = YOLO(str(path))

        use_cuda = torch.cuda.is_available() and self.device not in {"cpu", "-1"}
        dummy_input = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)

        # Warmup iterations
        for _ in range(warmup_iters):
            _ = model(dummy_input, device=self.device, verbose=False)

        latencies_ms: list[float] = []

        if use_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            for _ in range(bench_iters):
                start_event.record()
                _ = model(dummy_input, device=self.device, verbose=False)
                end_event.record()
                torch.cuda.synchronize()
                elapsed = start_event.elapsed_time(end_event)  # returns milliseconds
                latencies_ms.append(float(elapsed))
        else:
            for _ in range(bench_iters):
                t0 = time.perf_counter()
                _ = model(dummy_input, device=self.device, verbose=False)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(float(elapsed_ms))

        lat_arr = np.array(latencies_ms, dtype=np.float64)
        mean_ms = float(np.mean(lat_arr))
        median_ms = float(np.median(lat_arr))
        p90_ms = float(np.percentile(lat_arr, 90))
        p95_ms = float(np.percentile(lat_arr, 95))
        p99_ms = float(np.percentile(lat_arr, 99))
        fps = float(1000.0 / mean_ms) if mean_ms > 0 else 0.0

        res = LatencyBenchmarkResult(
            model_path=path,
            device=self.device,
            iterations=bench_iters,
            mean_latency_ms=mean_ms,
            median_latency_ms=median_ms,
            p90_latency_ms=p90_ms,
            p95_latency_ms=p95_ms,
            p99_latency_ms=p99_ms,
            fps=fps,
        )

        print("\n" + res.summary_table())
        return res
