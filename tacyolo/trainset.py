"""Build a merged YOLO dataset in stages: people/vehicles, drones, tanks, aircraft."""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen, urlretrieve

import yaml

from tacyolo.weights import ROOT

CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "airplane",
    "drone",
    "tank",
    "armored_car",
    "helicopter",
]

DATA_ROOT = ROOT / "data" / "tactical"
RAW_ROOT = ROOT / "datasets"
STATE_PATH = DATA_ROOT / "stages.json"

VISDRONE_MAP = {
    0: 0,  # pedestrian -> person
    1: 0,  # people -> person
    2: 1,  # bicycle
    3: 2,  # car
    4: 2,  # van -> car
    5: 5,  # truck
    8: 4,  # bus
    9: 3,  # motor -> motorcycle
}

COCO_MAP = {
    0: 0,  # person
    1: 1,  # bicycle
    2: 2,  # car
    3: 3,  # motorcycle
    4: 6,  # airplane
    5: 4,  # bus
    7: 5,  # truck
}

KEYWORD_MAP = [
    (("soldier", "person", "people", "pedestrian", "human"), 0),
    (("bicycle", "bike"), 1),
    (("car", "van", "taxi", "sedan"), 2),
    (("motorcycle", "motorbike", "motor"), 3),
    (("bus",), 4),
    (("truck", "lorry"), 5),
    (("drone", "uav", "quadcopter", "quadrocopter"), 7),
    (("tank",), 8),
    (("apc", "armored", "armoured", "ifv"), 9),
    (("helicopter", "heli", "chopper", "rotorcraft"), 10),
    (("fighter", "aircraft", "airplane", "aeroplane", "jet", "plane", "bomber", "warplane"), 6),
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

OI_CLASS_CSV = "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"
OI_BOX_URLS = {
    "validation": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
    "test": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
}
OI_IMAGE = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"
OI_NAME_TO_CLS = {
    "Person": 0,
    "Bicycle": 1,
    "Car": 2,
    "Motorcycle": 3,
    "Bus": 4,
    "Truck": 5,
    "Airplane": 6,
    "Aircraft": 6,
    "Helicopter": 10,
    "Tank": 8,
}
OI_CAPS = {
    0: 2000,
    1: 600,
    2: 2000,
    3: 600,
    4: 600,
    5: 1200,
    6: 3500,
    8: 4000,
    10: 3500,
}

SERAPHIM_REPO = "lgrzybowski/seraphim-drone-detection-dataset"


def _load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save_state(update: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state = _load_state()
    state.update(update)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _rewrite_label(
    src: Path,
    dst: Path,
    class_map: dict[int, int],
    default_cls: int | None = None,
) -> int:
    if not src.exists():
        dst.write_text("", encoding="utf-8")
        return 0
    lines_out = []
    kept = 0
    for raw in src.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
        except ValueError:
            continue
        if cls in class_map:
            parts[0] = str(class_map[cls])
        elif default_cls is not None:
            parts[0] = str(default_cls)
        else:
            continue
        lines_out.append(" ".join(parts))
        kept += 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")
    return kept


def _import_pairs(
    pairs: list[tuple[Path, Path]],
    split: str,
    class_map: dict[int, int],
    prefix: str,
    default_cls: int | None = None,
    keep_empty: bool = True,
) -> int:
    n = 0
    img_dir = DATA_ROOT / "images" / split
    lbl_dir = DATA_ROOT / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for img, label in pairs:
        stem = f"{prefix}_{img.stem}"
        dest_img = img_dir / f"{stem}{img.suffix.lower()}"
        dest_lbl = lbl_dir / f"{stem}.txt"
        kept = _rewrite_label(label, dest_lbl, class_map, default_cls=default_cls)
        if kept == 0 and not keep_empty and not label.exists():
            if dest_lbl.exists():
                dest_lbl.unlink()
            continue
        _link_or_copy(img, dest_img)
        n += 1
    return n


def _pairs_from_yolo_dirs(image_dir: Path, label_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    if not image_dir.exists():
        return pairs
    for img in image_dir.rglob("*"):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        label = label_dir / f"{img.stem}.txt"
        if not label.exists():
            nested = label_dir / img.parent.name / f"{img.stem}.txt"
            if nested.exists():
                label = nested
            else:
                walked = next(label_dir.rglob(f"{img.stem}.txt"), None)
                label = walked if walked is not None else label
        pairs.append((img, label))
    return pairs


def _pairs_from_flat(root: Path) -> list[tuple[Path, Path]]:
    """Kaggle dumps that keep image.jpg and image.txt in the same folder."""
    pairs = []
    if not root.exists():
        return pairs
    for img in root.rglob("*"):
        if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
            continue
        label = img.with_suffix(".txt")
        if label.exists():
            pairs.append((img, label))
    return pairs


def _find_split_dirs(root: Path) -> list[tuple[str, Path, Path]]:
    found: list[tuple[str, Path, Path]] = []
    candidates = [
        ("train", root / "images" / "train", root / "labels" / "train"),
        ("val", root / "images" / "val", root / "labels" / "val"),
        ("val", root / "images" / "valid", root / "labels" / "valid"),
        ("train", root / "images" / "train2017", root / "labels" / "train2017"),
        ("val", root / "images" / "val2017", root / "labels" / "val2017"),
        ("train", root / "train" / "images", root / "train" / "labels"),
        ("val", root / "valid" / "images", root / "valid" / "labels"),
        ("val", root / "val" / "images", root / "val" / "labels"),
        ("val", root / "test" / "images", root / "test" / "labels"),
        ("train", root / "images", root / "labels"),
    ]
    seen: set[str] = set()
    for split, images, labels in candidates:
        if images.exists() and labels.exists():
            key = str(images.resolve())
            if key in seen:
                continue
            seen.add(key)
            found.append((split, images, labels))
    if found:
        return found
    for labels in root.rglob("labels"):
        if not labels.is_dir():
            continue
        images = labels.parent / "images"
        if not images.exists():
            continue
        key = str(images.resolve())
        if key in seen:
            continue
        parent = labels.parent.name.lower()
        if parent in {"val", "valid", "validation", "test"}:
            split = "val"
        else:
            split = "train"
        seen.add(key)
        found.append((split, images, labels))
    return found


def _norm_name(name: str) -> str:
    return name.lower().replace("-", " ").replace("_", " ").strip()


def _map_from_names(names: list[str]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for i, name in enumerate(names):
        n = _norm_name(str(name))
        if n in {"bird", "birds", "other", "background", "negative", "dontcare", "don't care", "ignore", "flame", "solar", "panel"}:
            continue
        matched = None
        for keys, cls in KEYWORD_MAP:
            if any(k == n or k in n.split() or k in n for k in keys):
                matched = cls
                break
        if matched is not None:
            mapping[i] = matched
    return mapping


def _load_native_names(root: Path) -> list[str] | None:
    yaml_files = list(root.glob("*.yaml")) + list(root.glob("*.yml"))
    yaml_files += list(root.rglob("data.yaml"))
    for yml in yaml_files:
        try:
            data = yaml.safe_load(yml.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        names = data.get("names")
        if isinstance(names, dict):
            try:
                return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
            except Exception:
                return [str(v) for v in names.values()]
        if isinstance(names, list):
            return [str(x) for x in names]
    return None


def _extract_zips(root: Path) -> None:
    if not root.exists():
        return
    for zpath in list(root.rglob("*.zip")):
        marker = zpath.with_suffix(zpath.suffix + ".extracted")
        if marker.exists():
            continue
        dest = zpath.parent / zpath.stem
        dest.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(dest)
            marker.write_text("ok", encoding="utf-8")
            try:
                zpath.unlink()
            except OSError:
                pass
            print(f"extracted {zpath.name} -> {dest}")
        except zipfile.BadZipFile:
            print(f"skip bad zip {zpath}")


def download_visdrone() -> Path:
    from ultralytics.data.utils import check_det_dataset

    info = check_det_dataset("VisDrone.yaml")
    return Path(info["path"])


def download_coco128() -> Path:
    from ultralytics.data.utils import check_det_dataset

    info = check_det_dataset("coco128.yaml")
    return Path(info["path"])


def download_hf(repo_id: str, dest: Path) -> Path:
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(dest))
    _extract_zips(dest)
    return dest


def _import_root(root: Path, prefix: str, class_map: dict[int, int] | None = None, default_cls: int | None = None) -> int:
    names = _load_native_names(root)
    if names:
        guessed = _map_from_names(names)
        if guessed:
            class_map = guessed
    if class_map is None:
        class_map = {} if default_cls is not None else {0: 7}
    imported = 0
    search_roots = [root, root / "dataset", root / "data"]
    for search in search_roots:
        if not search.exists():
            continue
        for split, images, labels in _find_split_dirs(search):
            mapped_split = "train" if split == "train" else "val"
            imported += _import_pairs(
                _pairs_from_yolo_dirs(images, labels),
                mapped_split,
                class_map,
                prefix,
                default_cls=default_cls,
            )
    if imported == 0:
        pairs = _pairs_from_flat(root)
        if pairs:
            imported += _import_pairs(pairs, "train", class_map, prefix, default_cls=default_cls)
    return imported


def write_yaml() -> Path:
    cfg = {
        "path": str(DATA_ROOT.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
        "nc": len(CLASS_NAMES),
    }
    yaml_path = DATA_ROOT / "tactical.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.dump(cfg, sort_keys=False), encoding="utf-8")
    return yaml_path


def _ensure_val_split() -> None:
    train_imgs = list((DATA_ROOT / "images" / "train").glob("*"))
    val_dir = DATA_ROOT / "images" / "val"
    val_dir.mkdir(parents=True, exist_ok=True)
    if any(val_dir.iterdir()):
        return
    for img in train_imgs[::10]:
        _link_or_copy(img, val_dir / img.name)
        src_lbl = DATA_ROOT / "labels" / "train" / f"{img.stem}.txt"
        dst_lbl = DATA_ROOT / "labels" / "val" / f"{img.stem}.txt"
        if src_lbl.exists():
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_lbl, dst_lbl)


def _count_split(split: str) -> int:
    folder = DATA_ROOT / "images" / split
    if not folder.exists():
        return 0
    return sum(1 for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def _http_bytes(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "tacyolo/0.1"})
    with urlopen(req, timeout=120) as resp:
        return resp.read()


def _download_file(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return True
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = Request(url, headers={"User-Agent": "tacyolo/0.1"})
        with urlopen(req, timeout=60) as resp, open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh)
        tmp.replace(dest)
        return True
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return False


def source_visdrone() -> int:
    vis = download_visdrone()
    n = 0
    for split, images, labels in _find_split_dirs(vis):
        n += _import_pairs(_pairs_from_yolo_dirs(images, labels), split, VISDRONE_MAP, f"vd_{split}")
    return n


def source_coco128() -> int:
    coco = download_coco128()
    n = 0
    for split, images, labels in _find_split_dirs(coco):
        mapped_split = "train" if split == "train" else "val"
        n += _import_pairs(_pairs_from_yolo_dirs(images, labels), mapped_split, COCO_MAP, f"coco_{mapped_split}")
    return n


def source_hf_drone() -> int:
    dest = RAW_ROOT / "hf" / "Taranichilamkoti15__drone-detection"
    download_hf("Taranichilamkoti15/drone-detection", dest)
    return _import_root(dest, "drone", {0: 7, 1: 7}, default_cls=7)


def source_tank() -> int:
    dest = RAW_ROOT / "hf" / "Simuletic__UAV-Aerial-View-Battle-Tank-Detection-Dataset"
    download_hf("Simuletic/UAV-Aerial-View-Battle-Tank-Detection-Dataset", dest)
    return _import_root(dest, "tank", {0: 8, 1: 9}, default_cls=8)


def source_aerial() -> int:
    dest = RAW_ROOT / "hf" / "Tanishjain9__AerialObjects-2025"
    download_hf("Tanishjain9/AerialObjects-2025", dest)
    return _import_root(dest, "aerial", {1: 7, 2: 6, 3: 7, 4: 7})


def source_military() -> int:
    dest = RAW_ROOT / "hf" / "llama-farm__military-labeled-yolo"
    download_hf("llama-farm/military-labeled-yolo", dest)
    fallback = {
        0: 0,  # soldier
        1: 8,  # tank
        2: 9,  # apc
        5: 5,  # military_truck
        6: 10,  # helicopter
        7: 6,  # aircraft
        10: 2,  # car
        11: 5,  # truck
    }
    return _import_root(dest, "mil", fallback)


def source_aircraft() -> int:
    dest = RAW_ROOT / "hf" / "Ahnuf__Military_Aircraft"
    download_hf("Ahnuf/Military_Aircraft_Detection_Classification_Image_Dataset", dest)
    return _import_root(dest, "ac", {}, default_cls=6)


def source_seraphim() -> int:
    from huggingface_hub import hf_hub_download, list_repo_files

    dest = RAW_ROOT / "seraphim"
    dest.mkdir(parents=True, exist_ok=True)
    files = list_repo_files(SERAPHIM_REPO, repo_type="dataset")
    zips = sorted(f for f in files if f.endswith(".zip"))
    imported = 0
    state = _load_state()
    done = set(state.get("seraphim_zips", []))
    for rel in zips:
        if rel in done:
            print(f"seraphim already have {rel}")
        else:
            print(f"seraphim downloading {rel}")
            local = hf_hub_download(repo_id=SERAPHIM_REPO, repo_type="dataset", filename=rel)
            target_dir = dest / Path(rel).parent
            target_dir.mkdir(parents=True, exist_ok=True)
            out_zip = target_dir / Path(rel).name
            if Path(local).resolve() != out_zip.resolve():
                shutil.copy2(local, out_zip)
            extract_to = target_dir
            if rel.endswith(".zip"):
                marker = out_zip.with_suffix(out_zip.suffix + ".extracted")
                if not marker.exists():
                    with zipfile.ZipFile(out_zip) as zf:
                        zf.extractall(extract_to)
                    marker.write_text("ok", encoding="utf-8")
                    try:
                        out_zip.unlink()
                    except OSError:
                        pass
            done.add(rel)
            _save_state({"seraphim_zips": sorted(done)})
        n = _import_root(dest, "sph", {0: 7}, default_cls=7)
        imported = n
        print(f"seraphim running total images imported={n} after {rel}")
    return imported


def source_open_images() -> int:
    dest = RAW_ROOT / "open_images"
    img_dir = dest / "images"
    lbl_dir = dest / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = _http_bytes(OI_CLASS_CSV).decode("utf-8", errors="ignore")
    mid_to_cls: dict[str, int] = {}
    for row in csv.reader(io.StringIO(raw_csv)):
        if len(row) < 2:
            continue
        mid, name = row[0], row[1]
        if name in OI_NAME_TO_CLS:
            mid_to_cls[mid] = OI_NAME_TO_CLS[name]
    print(f"open-images matched classes: {len(mid_to_cls)}")

    boxes_by_image: dict[tuple[str, str], list[tuple[int, float, float, float, float]]] = defaultdict(list)
    for split, url in OI_BOX_URLS.items():
        print(f"open-images reading {split} boxes")
        text = _http_bytes(url).decode("utf-8", errors="ignore")
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            cls = mid_to_cls.get(row.get("LabelName", ""))
            if cls is None:
                continue
            try:
                xmin = float(row["XMin"])
                xmax = float(row["XMax"])
                ymin = float(row["YMin"])
                ymax = float(row["YMax"])
            except (KeyError, ValueError):
                continue
            xc = (xmin + xmax) / 2.0
            yc = (ymin + ymax) / 2.0
            w = max(0.0, xmax - xmin)
            h = max(0.0, ymax - ymin)
            if w <= 0 or h <= 0:
                continue
            boxes_by_image[(split, row["ImageID"])].append((cls, xc, yc, w, h))

    priority = [8, 10, 6, 5, 4, 3, 1, 2, 0]
    selected: set[tuple[str, str]] = set()
    class_counts: dict[int, int] = defaultdict(int)
    for cls in priority:
        cap = OI_CAPS.get(cls, 500)
        for key, boxes in boxes_by_image.items():
            if key in selected:
                continue
            if not any(b[0] == cls for b in boxes):
                continue
            if class_counts[cls] >= cap:
                continue
            selected.add(key)
            for c in {b[0] for b in boxes}:
                class_counts[c] += 1
    print(f"open-images selected images={len(selected)} class_counts={dict(class_counts)}")

    def _one(item: tuple[str, str]) -> bool:
        split, image_id = item
        dest_img = img_dir / f"{split}_{image_id}.jpg"
        dest_lbl = lbl_dir / f"{split}_{image_id}.txt"
        url = OI_IMAGE.format(split=split, image_id=image_id)
        if not _download_file(url, dest_img):
            return False
        lines = [f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for cls, xc, yc, w, h in boxes_by_image[item]]
        dest_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return True

    ok = 0
    items = list(selected)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(_one, item) for item in items]
        for i, fut in enumerate(as_completed(futs), 1):
            if fut.result():
                ok += 1
            if i % 200 == 0:
                print(f"open-images downloaded {i}/{len(items)} ok={ok}")
    identity = {i: i for i in range(len(CLASS_NAMES))}
    n = _import_pairs(_pairs_from_yolo_dirs(img_dir, lbl_dir), "train", identity, "oi")
    print(f"open-images imported={n} downloaded_ok={ok}")
    return n


def download_kaggle(dataset_ref: str) -> Path:
    import kagglehub

    path = Path(kagglehub.dataset_download(dataset_ref))
    _extract_zips(path)
    return path


KAGGLE_DATASETS = [
    # ref, prefix, fallback map, default_cls
    ("sshikamaru/drone-yolo-detection", "kg_drone1", {0: 7}, 7),
    ("muki2003/yolo-drone-detection-dataset", "kg_drone2", {0: 7}, 7),
    ("sudipchakrabarty/kiit-mita", "kg_mita", {0: 8, 1: 7, 2: 0, 3: 6, 4: 10, 5: 9}, None),
    ("stealthknight/bird-vs-drone", "kg_bvd", {1: 7}, None),
    ("troykueh/multi-class-drone-detection-dataset-yolov8-ready", "kg_mcd", {0: 7}, 7),
    ("caferfatihgltekin/air-defense-object-detection-dataset-yolov8", "kg_ad", {0: 6, 1: 10, 2: 7, 3: 6}, None),
    ("simuletic/uav-and-aerial-view-battle-tank-detection-dataset", "kg_tank", {0: 8, 1: 9}, 8),
]

KAGGLE_THERMAL = [
    ("pandrii000/hituav-a-highaltitude-infrared-thermal-dataset", "th_hit", {0: 0, 1: 2, 2: 1, 3: 5}, None),
    ("kausthubkannan/thermal-image-people-detection", "th_ppl", {0: 0}, 0),
    ("sikdermdsaiful/thermal-images-for-human-detection", "th_hum", {0: 0}, 0),
    ("mustafayngl/kaist-dataset-yolo26-early-fusion-preview-set", "th_kaist", {0: 0, 1: 2, 2: 1}, None),
    ("gaweshgomes/llvip-rgb-thermal-yolo-format", "th_llvip", {0: 0}, 0),
    ("niteshc7r/datasets-for-object-detection-night-and-thermal", "th_night", {0: 0, 1: 2, 2: 5}, None),
    ("animeshmahajan/thermal-image-dataset", "th_ani", {0: 0}, 0),
]


def source_kaggle() -> int:
    total = 0
    for ref, prefix, fallback, default_cls in KAGGLE_DATASETS:
        try:
            print(f"=== kaggle {ref} ===")
            dest = download_kaggle(ref)
            n = _import_root(dest, prefix, fallback, default_cls=default_cls)
            print(f"kaggle {ref} imported={n}")
            total += n
        except Exception as exc:  # noqa: BLE001
            print(f"kaggle {ref} FAILED {exc}")
    return total


def source_thermal() -> int:
    """Labeled thermal / IR frames (from thermal video and stills)."""
    total = 0
    for ref, prefix, fallback, default_cls in KAGGLE_THERMAL:
        try:
            print(f"=== thermal {ref} ===")
            dest = download_kaggle(ref)
            n = _import_root(dest, prefix, fallback, default_cls=default_cls)
            print(f"thermal {ref} imported={n}")
            total += n
        except Exception as exc:  # noqa: BLE001
            print(f"thermal {ref} FAILED {exc}")
    return total


SOURCE_FNS = {
    "visdrone": source_visdrone,
    "coco128": source_coco128,
    "hf_drone": source_hf_drone,
    "tank": source_tank,
    "aerial": source_aerial,
    "military": source_military,
    "aircraft": source_aircraft,
    "seraphim": source_seraphim,
    "open_images": source_open_images,
    "kaggle": source_kaggle,
    "thermal": source_thermal,
}


def prepare(sources: list[str] | None = None) -> Path:
    wanted = list(sources or ["visdrone", "coco128", "hf_drone", "tank"])
    stats: dict[str, int | str] = {}
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for name in wanted:
        fn = SOURCE_FNS.get(name)
        if fn is None:
            stats[name] = "UNKNOWN_SOURCE"
            continue
        try:
            print(f"=== preparing source {name} ===")
            stats[name] = fn()
        except Exception as exc:  # noqa: BLE001
            stats[name] = f"FAILED {exc}"
            print(f"source {name} failed: {exc}")
    _ensure_val_split()
    yaml_path = write_yaml()
    n_train = _count_split("train")
    n_val = _count_split("val")
    print("dataset stats:", stats)
    print(f"train images={n_train} val images={n_val} yaml={yaml_path}")
    _save_state({"last_stats": stats, "train": n_train, "val": n_val})
    if n_train < 20:
        raise RuntimeError(f"Not enough training images ({n_train}). Dataset download likely failed.")
    return yaml_path
