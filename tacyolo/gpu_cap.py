"""Keep the GPU cool for long training. Do not run the card at 100%."""

from __future__ import annotations

import subprocess
import time

TARGET_UTIL = 70
MAX_UTIL = 80
MAX_TEMP_C = 72
POWER_WATTS = 140
MAX_CLOCK_MHZ = 1350


def _nvidia_smi(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", *args],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=8,
        ).strip()
    except Exception as exc:
        return f"ERR {exc}"


def gpu_stats() -> dict[str, float]:
    raw = _nvidia_smi(
        "--query-gpu=temperature.gpu,utilization.gpu,power.draw,power.limit,clocks.gr",
        "--format=csv,noheader,nounits",
    )
    parts = [p.strip() for p in raw.split(",")]
    out = {"temp": 0.0, "util": 0.0, "power": 0.0, "limit": 0.0, "clock": 0.0}
    try:
        out["temp"] = float(parts[0])
        out["util"] = float(parts[1])
        out["power"] = float(parts[2])
        out["limit"] = float(parts[3])
        out["clock"] = float(parts[4])
    except (ValueError, IndexError):
        pass
    return out


def gpu_util() -> int | None:
    s = gpu_stats()
    return int(s["util"]) if s["util"] else None


def cap_hardware(power_watts: int = POWER_WATTS, max_clock_mhz: int = MAX_CLOCK_MHZ) -> None:
    """Cut power and clocks so a 24h run cannot cook a 3060 Ti."""
    info = _nvidia_smi("--query-gpu=power.min_limit,power.max_limit", "--format=csv,noheader,nounits")
    watts = power_watts
    try:
        lo, hi = [float(x.strip()) for x in info.split(",")[:2]]
        watts = int(min(max(power_watts, lo), hi))
    except Exception:
        pass
    print(_nvidia_smi("-pl", str(watts)))
    print(_nvidia_smi("-lgc", f"210,{max_clock_mhz}"))
    print(f"gpu cap applied: {watts}W, graphics clock max {max_clock_mhz} MHz, temp target < {MAX_TEMP_C}C")
    print("stats", gpu_stats())


def throttle_if_busy(target: int = TARGET_UTIL, ceiling: int = MAX_UTIL) -> None:
    s = gpu_stats()
    if s["temp"] >= MAX_TEMP_C + 6:
        time.sleep(0.35)
    elif s["temp"] >= MAX_TEMP_C or s["util"] >= ceiling:
        time.sleep(0.16)
    elif s["util"] >= target:
        time.sleep(0.07)


def attach_ultralytics(model) -> None:
    cap_hardware()

    def _on_batch(_trainer=None):
        throttle_if_busy()

    model.add_callback("on_train_batch_end", _on_batch)
    print(f"gpu throttle: util < {MAX_UTIL}%  temp < {MAX_TEMP_C}C")


if __name__ == "__main__":
    cap_hardware()
