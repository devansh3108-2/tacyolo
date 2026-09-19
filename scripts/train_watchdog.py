"""TACYOLO train watchdog + terminal health."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")
except Exception:
    IST = timezone.utc

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runs" / "train_watch_state.json"
LOG = ROOT / "runs" / "train_watch.jsonl"
RESULTS = ROOT / "runs" / "train" / "tactical" / "results.csv"
ARGS = ROOT / "runs" / "train" / "tactical" / "args.yaml"
TRAIN_DIR = ROOT / "runs" / "train" / "tactical"
STDERR = ROOT / "runs" / "train_watch_stderr.log"
STDOUT = ROOT / "runs" / "train_watch_stdout.log"
STALL_MINUTES = 25


def now_ist() -> datetime:
    return datetime.now(IST)


def log(event: str, **extra):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts": now_ist().isoformat(), "event": event, **extra}
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8-sig"))
    return {
        "queue": [
            "thermal", "kaggle", "seraphim", "aerial", "military",
            "aircraft", "open_images", "visdrone", "hf_drone", "tank", "coco128",
        ],
        "next_index": 0,
        "deadline_ist": "2026-09-19T23:59:59+05:30",
    }


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=2), encoding="utf-8")


def past_deadline(st: dict) -> bool:
    raw = st.get("deadline_ist") or "2026-09-19T23:59:59+05:30"
    try:
        dl = datetime.fromisoformat(raw)
    except ValueError:
        dl = datetime(2026, 9, 19, 23, 59, 59, tzinfo=IST)
    if dl.tzinfo is None:
        dl = dl.replace(tzinfo=IST)
    return now_ist() >= dl


def train_pids() -> list[str]:
    try:
        out = subprocess.check_output(
            [
                "powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                "Where-Object { $_.CommandLine -match 'tacyolo' -and $_.CommandLine -match 'train' "
                "-and $_.CommandLine -notmatch 'train_watchdog' } | "
                "Select-Object -ExpandProperty ProcessId",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return [x.strip() for x in out.splitlines() if x.strip()]
    except Exception as exc:
        log("pid_check_failed", error=str(exc))
        return []


def related_pids() -> list[str]:
    """Train + dataset prepare helpers (terminal-adjacent activity)."""
    try:
        out = subprocess.check_output(
            [
                "powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                "Where-Object { $_.CommandLine -match 'tacyolo' -and $_.CommandLine -notmatch 'train_watchdog' } | "
                "Select-Object -ExpandProperty ProcessId",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return [x.strip() for x in out.splitlines() if x.strip()]
    except Exception:
        return []


def train_running() -> bool:
    return bool(train_pids()) or bool(related_pids())


def read_terminal_progress() -> dict:
    info = {
        "results_exists": RESULTS.exists(),
        "epoch": None,
        "epochs_total": None,
        "mtime_age_sec": None,
        "last_line": None,
        "metrics": {},
        "stall": False,
        "warming_up": False,
    }
    if ARGS.exists():
        for line in ARGS.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("epochs:"):
                try:
                    info["epochs_total"] = int(line.split(":", 1)[1].strip().strip("'\""))
                except Exception:
                    pass
                break
    if RESULTS.exists():
        mtime = RESULTS.stat().st_mtime
        info["mtime_age_sec"] = int(time.time() - mtime)
        lines = [ln for ln in RESULTS.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
        if len(lines) >= 2:
            header = [h.strip() for h in lines[0].split(",")]
            last = lines[-1]
            info["last_line"] = last
            cols = [c.strip() for c in last.split(",")]
            try:
                info["epoch"] = int(float(cols[0]))
            except Exception:
                pass

            def col(name: str):
                if name in header and header.index(name) < len(cols):
                    return cols[header.index(name)]
                return None

            info["metrics"] = {
                "time": col("time"),
                "train/box_loss": col("train/box_loss"),
                "map50": col("metrics/mAP50(B)"),
                "map5095": col("metrics/mAP50-95(B)"),
            }
        if info["mtime_age_sec"] is not None and info["mtime_age_sec"] > STALL_MINUTES * 60:
            info["stall"] = True
    else:
        # Ultralytics writes results.csv after epoch 1; use newest train-dir file.
        info["warming_up"] = True
        newest = None
        if TRAIN_DIR.exists():
            files = [f for f in TRAIN_DIR.rglob("*") if f.is_file()]
            if files:
                newest = max(f.stat().st_mtime for f in files)
                info["mtime_age_sec"] = int(time.time() - newest)
                if info["mtime_age_sec"] > STALL_MINUTES * 60:
                    info["stall"] = True
        elif ARGS.exists():
            info["mtime_age_sec"] = int(time.time() - ARGS.stat().st_mtime)
            if info["mtime_age_sec"] > STALL_MINUTES * 60:
                info["stall"] = True
    return info


def next_source(st: dict) -> str:
    q = st.get("queue") or ["thermal", "kaggle", "seraphim"]
    i = int(st.get("next_index") or 0) % len(q)
    src = q[i]
    st["next_index"] = (i + 1) % len(q)
    st["last_source"] = src
    st["last_action"] = now_ist().isoformat()
    save_state(st)
    return src


def best_weights():
    for c in [
        ROOT / "runs" / "train" / "tactical" / "weights" / "best.pt",
        ROOT / "weights" / "tactical_yolo11s.pt",
        ROOT / "yolo11s.pt",
    ]:
        if c.exists():
            return c
    return None


def start_train(source: str) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    STDOUT.parent.mkdir(parents=True, exist_ok=True)
    prep = [sys.executable, "-m", "tacyolo", "train", "--prepare-only", "--sources", source]
    log("prepare_start", source=source, cmd=prep)
    prep_code = subprocess.call(prep, cwd=str(ROOT), env=env)
    log("prepare_done", source=source, code=prep_code)
    model = best_weights()
    cmd = [sys.executable, "-m", "tacyolo", "train", "--epochs", "40", "--batch", "4", "--device", "auto"]
    if model:
        cmd += ["--model", str(model)]
    log("train_start", source=source, cmd=cmd)
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    with STDOUT.open("a", encoding="utf-8") as out, STDERR.open("a", encoding="utf-8") as err:
        out.write(f"\n===== start {now_ist().isoformat()} source={source} =====\n")
        out.flush()
        subprocess.Popen(
            cmd, cwd=str(ROOT), env=env, creationflags=creationflags, stdout=out, stderr=err
        )
    time.sleep(4)
    return 0 if train_running() else 1


def main() -> int:
    st = load_state()
    if past_deadline(st):
        log("deadline_reached_stop")
        print("DEADLINE")
        return 2

    prog = read_terminal_progress()
    st["last_epoch"] = prog.get("epoch")
    st["last_results_age_sec"] = prog.get("mtime_age_sec")
    save_state(st)

    if train_running():
        if prog.get("stall"):
            log(
                "terminal_stall",
                epoch=prog.get("epoch"),
                epochs_total=prog.get("epochs_total"),
                mtime_age_sec=prog.get("mtime_age_sec"),
                metrics=prog.get("metrics"),
            )
            print("STALL")
            return 3
        log(
            "still_training",
            epoch=prog.get("epoch"),
            epochs_total=prog.get("epochs_total"),
            mtime_age_sec=prog.get("mtime_age_sec"),
            metrics=prog.get("metrics"),
            warming_up=prog.get("warming_up"),
            pids=related_pids(),
        )
        ep = prog.get("epoch")
        tot = prog.get("epochs_total") or "?"
        age = prog.get("mtime_age_sec")
        warm = " warming_up" if prog.get("warming_up") else ""
        print(f"TRAINING epoch={ep}/{tot} results_age_sec={age}{warm}")
        return 0

    if STDERR.exists():
        tail = STDERR.read_text(encoding="utf-8", errors="ignore")[-2000:]
        if any(x in tail.lower() for x in ("traceback", "cuda out of memory", "error:", "exception")):
            log("prior_train_error_tail", tail=tail[-800:])

    src = next_source(st)
    log("idle_restart", source=src, last_progress=prog)
    code = start_train(src)
    print("RESTARTED" if code == 0 else "RESTART_FAILED")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
