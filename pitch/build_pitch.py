"""Boardroom partner pitch for TACYOLO. Run: python pitch/build_pitch.py"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.colors import Color, HexColor, white
from reportlab.lib.pagesizes import landscape
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parent / "TACYOLO_Company_Pitch.pdf"
W, H = 13.333 * inch, 7.5 * inch

NAVY = HexColor("#0B1F33")
NAVY2 = HexColor("#143047")
INK = HexColor("#1A2330")
MUTED = HexColor("#5C6B7A")
GOLD = HexColor("#B0894F")
GOLD2 = HexColor("#C9A56A")
PAPER = HexColor("#F3F0E8")
CARD = HexColor("#FFFCF7")
LINE = HexColor("#D8D2C6")
TEAL = HexColor("#2F6F5E")
CLAY = HexColor("#8A4B2A")
SOFT = HexColor("#E7E1D4")


def _register_fonts() -> tuple[str, str, str]:
    candidates = [
        (
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/segoeuii.ttf",
        ),
        (
            "C:/Windows/Fonts/calibrib.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "C:/Windows/Fonts/calibrii.ttf",
        ),
    ]
    for bold, regular, italic in candidates:
        if Path(bold).exists() and Path(regular).exists():
            pdfmetrics.registerFont(TTFont("Pitch", regular))
            pdfmetrics.registerFont(TTFont("Pitch-Bold", bold))
            if Path(italic).exists():
                pdfmetrics.registerFont(TTFont("Pitch-Italic", italic))
                return "Pitch", "Pitch-Bold", "Pitch-Italic"
            return "Pitch", "Pitch-Bold", "Pitch"
    return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"


FONT, FONT_B, FONT_I = _register_fonts()


def wrap(c: canvas.Canvas, text: str, font: str, size: float, max_w: float) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        trial = (cur + " " + word).strip()
        if c.stringWidth(trial, font, size) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def draw_bg(c: canvas.Canvas) -> None:
    c.setFillColor(PAPER)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(NAVY)
    c.rect(0, 0, 0.18 * inch, H, fill=1, stroke=0)


def footer(c: canvas.Canvas, page: int, total: int = 10) -> None:
    c.setFillColor(LINE)
    c.rect(0.55 * inch, 0.38 * inch, W - 0.9 * inch, 0.6, fill=1, stroke=0)
    c.setFillColor(MUTED)
    c.setFont(FONT, 8)
    c.drawString(
        0.55 * inch,
        0.18 * inch,
        "CONFIDENTIAL  ·  Partner briefing  ·  Detection, classification & tracking only  ·  Not a fire-control or weapons system",
    )
    c.setFont(FONT_B, 8)
    c.drawRightString(W - 0.4 * inch, 0.18 * inch, f"{page:02d}  /  {total:02d}")


def wordmark(c: canvas.Canvas, x: float, y: float, scale: float = 1.0) -> None:
    s = 10 * scale
    c.setFillColor(GOLD)
    c.rect(x, y + 2 * scale, 3.2 * scale, 11 * scale, fill=1, stroke=0)
    c.setFillColor(NAVY)
    c.setFont(FONT_B, 13 * scale)
    c.drawString(x + 8 * scale, y, "TACYOLO")
    c.setFont(FONT, 7.2 * scale)
    c.setFillColor(MUTED)
    c.drawString(x + 8 * scale + c.stringWidth("TACYOLO", FONT_B, 13 * scale) + 8 * scale, y + 2 * scale, "SOFTWARE")


def heading(c: canvas.Canvas, kicker: str, title: str, y: float = 6.85 * inch) -> None:
    c.setFillColor(GOLD)
    c.setFont(FONT_B, 8.5)
    c.drawString(0.55 * inch, y, kicker.upper())
    c.setFillColor(NAVY)
    c.setFont(FONT_B, 26)
    c.drawString(0.55 * inch, y - 0.38 * inch, title)


def card(c: canvas.Canvas, x: float, y: float, w: float, h: float) -> None:
    c.setFillColor(white)
    c.setStrokeColor(LINE)
    c.setLineWidth(0.8)
    c.roundRect(x, y, w, h, 8, fill=1, stroke=1)


def para(c: canvas.Canvas, text: str, x: float, y: float, max_w: float, size: float = 11, color=INK, font: str | None = None, leading: float | None = None) -> float:
    font = font or FONT
    leading = leading or (size + 5)
    c.setFillColor(color)
    c.setFont(font, size)
    for line in wrap(c, text, font, size, max_w):
        c.drawString(x, y, line)
        y -= leading
    return y


def page_cover(c: canvas.Canvas) -> None:
    c.setFillColor(NAVY)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(HexColor("#0E283F"))
    c.rect(W * 0.62, 0, W * 0.38, H, fill=1, stroke=0)
    c.setFillColor(GOLD)
    c.rect(W * 0.62, 0, 0.07 * inch, H, fill=1, stroke=0)

    c.setFillColor(GOLD)
    c.rect(0.7 * inch, 5.85 * inch, 0.28 * inch, 0.08 * inch, fill=1, stroke=0)
    c.setFillColor(GOLD2)
    c.setFont(FONT_B, 9)
    c.drawString(0.7 * inch, 6.15 * inch, "PARTNER BRIEFING  ·  SEPTEMBER 2026")

    c.setFillColor(white)
    c.setFont(FONT_B, 48)
    c.drawString(0.7 * inch, 5.05 * inch, "TACYOLO")
    c.setFont(FONT, 18)
    c.setFillColor(HexColor("#C9D6E2"))
    y = para(
        c,
        "Software-defined detection and tracking for cameras — and a radar path when you have one. Built to run on hardware you already own.",
        0.7 * inch,
        4.55 * inch,
        6.6 * inch,
        size=14,
        color=HexColor("#C9D6E2"),
        leading=20,
    )
    c.setFillColor(GOLD)
    c.setFont(FONT_B, 11)
    c.drawString(0.7 * inch, 2.55 * inch, "Seeking a company to productize, field, and own the first SKU.")
    c.setFillColor(HexColor("#8AA0B3"))
    c.setFont(FONT, 9)
    c.drawString(0.7 * inch, 0.45 * inch, "Confidential. Dual-use computer vision. Not a targeting or weapons-release system.")

    items = [
        ("01", "Detect & classify", "People, vehicles, drones, armor, aircraft"),
        ("02", "Track in time", "EKF tracks with radar range/bearing when present"),
        ("03", "Ship on existing silicon", "Windows prototype now · Jetson next · no new PCB"),
    ]
    y = 5.7 * inch
    for num, title, sub in items:
        c.setFillColor(GOLD)
        c.setFont(FONT_B, 10)
        c.drawString(W * 0.62 + 0.45 * inch, y, num)
        c.setFillColor(white)
        c.setFont(FONT_B, 13)
        c.drawString(W * 0.62 + 0.45 * inch, y - 0.26 * inch, title)
        c.setFillColor(HexColor("#A9BCCB"))
        c.setFont(FONT, 10)
        para(c, sub, W * 0.62 + 0.45 * inch, y - 0.48 * inch, 4.0 * inch, size=10, color=HexColor("#A9BCCB"), leading=14)
        y -= 1.35 * inch
    c.showPage()


def page_problem(c: canvas.Canvas) -> None:
    draw_bg(c)
    heading(c, "01  —  The gap", "Cameras got cheap. Attention did not.")
    para(
        c,
        "Small UAS, mixed traffic, and 24/7 towers produce more pixels than operators can watch. Off-the-shelf YOLO demos detect a class list. They do not decide when to spend GPU, keep identity, or fuse a second sensor.",
        0.55 * inch,
        6.15 * inch,
        12.2 * inch,
        size=12.5,
        leading=18,
    )
    boxes = [
        ("The operator problem", "Idle scenes still run full inference. Small movers are missed. Tracks break behind cover. A second sensor (radar, IR) is bolted on as a separate product."),
        ("The integrator problem", "Defense and security OEMs do not want another GitHub notebook. They want a software stack that fits existing cameras, existing GPUs, and an existing C2 bus."),
        ("The honesty problem", "The market is full of 99% claims, thermal magic, and “AI fire control.” Serious buyers discount that immediately. We designed this briefing the opposite way."),
    ]
    x = 0.55 * inch
    for title, body in boxes:
        card(c, x, 1.55 * inch, 3.9 * inch, 3.85 * inch)
        c.setFillColor(GOLD)
        c.rect(x, 5.32 * inch, 3.9 * inch, 0.08 * inch, fill=1, stroke=0)
        c.setFillColor(NAVY)
        c.setFont(FONT_B, 13)
        c.drawString(x + 0.22 * inch, 4.95 * inch, title)
        para(c, body, x + 0.22 * inch, 4.55 * inch, 3.45 * inch, size=11, leading=16, color=MUTED)
        x += 4.1 * inch
    footer(c, 2)
    c.showPage()


def page_product(c: canvas.Canvas) -> None:
    draw_bg(c)
    heading(c, "02  —  The product", "One stack. Software only. Existing hardware.")
    para(
        c,
        "A gated detector, gallery matcher, EKF tracker, and radar-side chain. RF is live on simulated I/Q today — swap in real recordings without a board spin.",
        0.55 * inch,
        6.18 * inch,
        12.2 * inch,
        size=12.5,
        leading=18,
    )
    rows = [
        ("Motion gate", "Two-pass change detection. Quiet frames skip YOLO. GPU budget goes to movers."),
        ("Detector", "YOLO11 / YOLOv8-military / YOLO-World. Classes plus open-vocab prompts (UAV, tank, helicopter)."),
        ("Identity", "Hyperdimensional item memory on chips. Operator gallery, cosine match — not a 99.8% claim."),
        ("Tracker", "EKF + Hungarian association. Optional radar range/bearing update when RF is present."),
        ("RF path", "FISTA compressed sensing, Gaussian scale-space, SVD clutter, lightweight peak extraction."),
        ("Runtime", "Windows desktop now. ONNX export. TensorRT / INT8 and Jetson are the next SKU, not a redesign."),
    ]
    col_w = 6.05 * inch
    positions = [
        (0.55 * inch, 4.35 * inch),
        (6.75 * inch, 4.35 * inch),
        (0.55 * inch, 2.70 * inch),
        (6.75 * inch, 2.70 * inch),
        (0.55 * inch, 1.05 * inch),
        (6.75 * inch, 1.05 * inch),
    ]
    for (title, body), (x, y) in zip(rows, positions):
        card(c, x, y, col_w, 1.48 * inch)
        c.setFillColor(GOLD)
        c.circle(x + 0.28 * inch, y + 1.12 * inch, 0.07 * inch, fill=1, stroke=0)
        c.setFillColor(NAVY)
        c.setFont(FONT_B, 12.5)
        c.drawString(x + 0.48 * inch, y + 1.04 * inch, title)
        para(c, body, x + 0.22 * inch, y + 0.68 * inch, col_w - 0.44 * inch, size=10.5, color=MUTED, leading=15)
    footer(c, 3)
    c.showPage()


def page_classes(c: canvas.Canvas) -> None:
    draw_bg(c)
    heading(c, "03  —  What it is built to see", "Eleven tactical classes. Not an 80-class toy list.")
    classes = [
        ("Person", "Aerial + street sets"),
        ("Bicycle", "VisDrone / COCO"),
        ("Car", "Van mapped to car"),
        ("Motorcycle", "Ground movers"),
        ("Bus", "Large vehicle"),
        ("Truck", "Logistics / mil truck"),
        ("Airplane", "Civil + military"),
        ("Drone", "Rotary and fixed-wing"),
        ("Tank", "MBT / tracked armor"),
        ("Armored car", "APC / AFV family"),
        ("Helicopter", "Rotary wing"),
    ]
    x, y = 0.55 * inch, 5.12 * inch
    for i, (name, note) in enumerate(classes):
        card(c, x, y, 2.35 * inch, 0.78 * inch)
        c.setFillColor(GOLD)
        c.setFont(FONT_B, 8)
        c.drawString(x + 0.16 * inch, y + 0.54 * inch, f"{i:02d}")
        c.setFillColor(NAVY)
        c.setFont(FONT_B, 12.5)
        c.drawString(x + 0.16 * inch, y + 0.30 * inch, name)
        c.setFillColor(MUTED)
        c.setFont(FONT, 8)
        c.drawString(x + 0.16 * inch, y + 0.12 * inch, note)
        x += 2.5 * inch
        if (i + 1) % 5 == 0:
            x = 0.55 * inch
            y -= 0.90 * inch
    card(c, 0.55 * inch, 0.95 * inch, 12.25 * inch, 2.05 * inch)
    c.setFillColor(TEAL)
    c.setFont(FONT_B, 11)
    c.drawString(0.78 * inch, 2.55 * inch, "IN SCOPE")
    para(
        c,
        "Detect, classify, track, and cue an operator. RGB cameras now. Thermal/IR when labeled frames are supplied. File, webcam, or tower stream. Synthetic coupled radar for software integration today.",
        0.78 * inch,
        2.25 * inch,
        5.5 * inch,
        size=10,
        color=MUTED,
        leading=14,
    )
    c.setFillColor(CLAY)
    c.setFont(FONT_B, 11)
    c.drawString(7.1 * inch, 2.55 * inch, "OUT OF SCOPE")
    para(
        c,
        "No fire control, no weapons release, no autonomous engagement, no claimed 99% combat ID. This is a perception layer a company can take into a certified product — not a finished weapon.",
        7.1 * inch,
        2.25 * inch,
        5.3 * inch,
        size=10,
        color=MUTED,
        leading=14,
    )
    footer(c, 4)
    c.showPage()


def _pipe_box(c: canvas.Canvas, x: float, y: float, w: float, h: float, title: str, sub: str, fill: Color = NAVY) -> None:
    c.setFillColor(fill)
    c.roundRect(x, y, w, h, 6, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont(FONT_B, 9)
    c.drawCentredString(x + w / 2, y + h / 2 + 4, title)
    c.setFont(FONT, 7.5)
    c.setFillColor(HexColor("#D5E2EC"))
    c.drawCentredString(x + w / 2, y + h / 2 - 10, sub)


def _arrow(c: canvas.Canvas, x0: float, y0: float, x1: float) -> None:
    c.setStrokeColor(GOLD)
    c.setFillColor(GOLD)
    c.setLineWidth(1.4)
    c.line(x0, y0, x1 - 6, y0)
    path = c.beginPath()
    path.moveTo(x1, y0)
    path.lineTo(x1 - 7, y0 + 4)
    path.lineTo(x1 - 7, y0 - 4)
    path.close()
    c.drawPath(path, fill=1, stroke=0)


def page_architecture(c: canvas.Canvas) -> None:
    draw_bg(c)
    heading(c, "04  —  Architecture", "Two sensors. One track picture.")
    # optical row
    c.setFillColor(NAVY)
    c.setFont(FONT_B, 10)
    c.drawString(0.55 * inch, 6.12 * inch, "OPTICAL PATH")
    boxes = [
        (0.55, "Camera / file", "RGB · IR later"),
        (3.05, "Motion gate", "Skip idle GPU"),
        (5.55, "YOLO detector", "11 classes + World"),
        (8.05, "HDC gallery", "Chip identity"),
        (10.55, "EKF tracks", "IDs over time"),
    ]
    y = 5.15 * inch
    for i, (x, t, s) in enumerate(boxes):
        _pipe_box(c, x * inch, y, 2.15 * inch, 0.72 * inch, t, s)
        if i < len(boxes) - 1:
            _arrow(c, (x + 2.15) * inch, y + 0.36 * inch, boxes[i + 1][0] * inch)

    c.setFillColor(NAVY)
    c.setFont(FONT_B, 10)
    c.drawString(0.55 * inch, 4.55 * inch, "RF PATH  —  simulated I/Q now, real recordings later")
    rf = [
        (0.55, "Range-Doppler", "Sim or .npy"),
        (3.05, "FISTA CS", "Sparse recover"),
        (5.55, "Scale-space", "Multi-sigma"),
        (8.05, "SVD clutter", "Rank strip"),
        (10.55, "TDA-lite peaks", "Not full VR"),
    ]
    y = 3.58 * inch
    for i, (x, t, s) in enumerate(rf):
        _pipe_box(c, x * inch, y, 2.15 * inch, 0.72 * inch, t, s, fill=NAVY2)
        if i < len(rf) - 1:
            _arrow(c, (x + 2.15) * inch, y + 0.36 * inch, rf[i + 1][0] * inch)

    c.setStrokeColor(GOLD)
    c.setDash(3, 3)
    c.setLineWidth(1.2)
    c.line(11.62 * inch, 3.58 * inch + 0.72 * inch, 11.62 * inch, 5.15 * inch)
    c.setDash()
    c.setFillColor(GOLD)
    c.setFont(FONT_B, 8)
    c.drawRightString(11.45 * inch, 4.88 * inch, "range / bearing update")

    card(c, 0.55 * inch, 1.05 * inch, 12.25 * inch, 2.05 * inch)
    notes = [
        ("Software product, not a board spin", "CLI already looks like YOLO: predict, track, export, fetch, train. Integrators can wrap it without learning a new religion."),
        ("Coupled scene simulator", "Optical movers and synthetic radar returns share the same scene so fusion can be tested before a range day."),
        ("Honest RF", "Compressed sensing + clutter + cheap topology on peaks. Ready for real I/Q — not a fielded radar STAP stack."),
    ]
    x = 0.78 * inch
    for title, body in notes:
        c.setFillColor(NAVY)
        c.setFont(FONT_B, 11)
        c.drawString(x, 2.68 * inch, title)
        para(c, body, x, 2.42 * inch, 3.7 * inch, size=9.5, color=MUTED, leading=13.5)
        x += 4.05 * inch
    footer(c, 5)
    c.showPage()


def page_traction(c: canvas.Canvas) -> None:
    draw_bg(c)
    heading(c, "05  —  What exists today", "A prototype you can run this week. Not a slide-ware stack.")
    stats = [
        ("8,755", "Labeled train images already merged (VisDrone + COCO sample)"),
        ("11", "Tactical classes in the training taxonomy"),
        ("15+", "Automated tests (gate, CS, HDC, EKF, JSONL, video+RF sim)"),
        ("3060 Ti", "CUDA prototype GPU — Jetson is the edge SKU, not a rewrite"),
    ]
    x = 0.55 * inch
    for num, label in stats:
        card(c, x, 4.55 * inch, 3.0 * inch, 1.55 * inch)
        c.setFillColor(GOLD)
        c.setFont(FONT_B, 22)
        c.drawString(x + 0.2 * inch, 5.55 * inch, num)
        para(c, label, x + 0.2 * inch, 5.18 * inch, 2.6 * inch, size=9.5, color=MUTED, leading=13)
        x += 3.15 * inch

    left = [
        "Package: tacyolo  ·  Python 3.10+  ·  Ultralytics backend",
        "Weights path: YOLO11, military YOLOv8s, YOLO-World v2",
        "Training curriculum staged: core aerial → drones → military air/ground → Google Open Images",
        "Outputs: overlay video, JSONL tracks, ONNX export hook",
        "Config-driven: YAML for gate, RF, HDC, EKF — no recompile",
    ]
    right = [
        "Windows desktop first (this lab). Linux / Jetson next.",
        "No custom PCB. No RF frontend claimed. Software on your camera.",
        "Public data only so far: VisDrone, COCO, Hugging Face military/drone sets, Open Images planned.",
        "Private operational video is the fastest accuracy unlock a partner can bring.",
        "Code is structured for a company team to own, not a one-off Colab.",
    ]
    card(c, 0.55 * inch, 0.95 * inch, 6.05 * inch, 3.35 * inch)
    card(c, 6.75 * inch, 0.95 * inch, 6.05 * inch, 3.35 * inch)
    c.setFillColor(NAVY)
    c.setFont(FONT_B, 12)
    c.drawString(0.78 * inch, 3.92 * inch, "In the repository")
    c.drawString(6.98 * inch, 3.92 * inch, "What that means for a buyer")
    y = 3.55 * inch
    for line in left:
        c.setFillColor(GOLD)
        c.circle(0.92 * inch, y + 3, 2.2, fill=1, stroke=0)
        para(c, line, 1.15 * inch, y, 5.1 * inch, size=10, color=INK, leading=13)
        y -= 0.48 * inch
    y = 3.55 * inch
    for line in right:
        c.setFillColor(GOLD)
        c.circle(7.12 * inch, y + 3, 2.2, fill=1, stroke=0)
        para(c, line, 7.35 * inch, y, 5.1 * inch, size=10, color=INK, leading=13)
        y -= 0.48 * inch
    footer(c, 6)
    c.showPage()


def page_honesty(c: canvas.Canvas) -> None:
    draw_bg(c)
    heading(c, "06  —  Trust", "We would rather lose a meeting than fake a metric.")
    para(
        c,
        "Defense and security companies have been burned by inflated demos. This page is the product. If a claim is not below, do not write it on a capture slide.",
        0.55 * inch,
        6.15 * inch,
        12.2 * inch,
        size=12.5,
        leading=18,
    )
    rows = [
        ("Identification", "HDC gallery matching is shipping as cosine-on-chips. There is no 99.8% combat-ID number. Accuracy is gallery- and data-dependent."),
        ("Radar", "The RF chain runs on synthetic range-Doppler until the partner provides I/Q or recordings. Adapters are the work, not a science project from zero."),
        ("Topology", "Peak extraction is TDA-lite. It is not a full Vietoris–Rips pipeline and we will not sell it as one."),
        ("Edge performance", "INT8 / TensorRT is designed in (export hook) and not yet the production runtime. ONNX is the current portable artifact."),
        ("Thermal / all cameras", "RGB is trained. Thermal, SWIR, and compressed air-to-ground need labeled frames. Architecture does not block them; data does."),
        ("Training data", "Public sets get you a starting detector. Operational footage from the partner’s cameras is what makes a SKU."),
    ]
    y = 5.55 * inch
    for title, body in rows:
        c.setFillColor(CLAY)
        c.roundRect(0.55 * inch, y - 0.12 * inch, 1.85 * inch, 0.42 * inch, 4, fill=1, stroke=0)
        c.setFillColor(white)
        c.setFont(FONT_B, 8.5)
        c.drawCentredString(0.55 * inch + 0.925 * inch, y, title.upper())
        para(c, body, 2.55 * inch, y + 0.02 * inch, 10.2 * inch, size=10.5, color=INK, leading=14)
        y -= 0.72 * inch
    footer(c, 7)
    c.showPage()


def page_market(c: canvas.Canvas) -> None:
    draw_bg(c)
    heading(c, "07  —  Who this is for", "A perception layer for companies that already ship the rest.")
    buyers = [
        ("Counter-UAS & airfields", "Detect small UAS and distinguish drones from birds/aircraft in a camera network. Feed an existing alert console."),
        ("Perimeter & energy", "Towers on pipelines, plants, bases. Motion gate keeps GPU cost sane on mostly-empty scenes."),
        ("Defense ISR software", "Add a tracker + optional RF cue to an existing Dismount/vehicle workflow without a hardware spin."),
        ("Public safety / SAR", "People and vehicles from drones or mast cameras. Same model family, different concept of employment."),
    ]
    y = 5.55 * inch
    for title, body in buyers:
        card(c, 0.55 * inch, y - 0.15 * inch, 12.25 * inch, 0.95 * inch)
        c.setFillColor(GOLD)
        c.rect(0.55 * inch, y - 0.15 * inch, 0.09 * inch, 0.95 * inch, fill=1, stroke=0)
        c.setFillColor(NAVY)
        c.setFont(FONT_B, 13)
        c.drawString(0.9 * inch, y + 0.48 * inch, title)
        para(c, body, 0.9 * inch, y + 0.22 * inch, 11.5 * inch, size=11, color=MUTED, leading=14)
        y -= 1.08 * inch
    footer(c, 8)
    c.showPage()


def page_roadmap(c: canvas.Canvas) -> None:
    draw_bg(c)
    heading(c, "08  —  Twelve months", "From lab prototype to a SKU a company can sell.")
    phases = [
        (
            "Days 0–90",
            "Pilot",
            [
                "Freeze a v0.1 API the partner can call",
                "Run on their cameras, not only lab files",
                "Label a private set from that feed",
                "TensorRT path on one Jetson",
                "Overlay + JSONL into their existing bus",
            ],
        ),
        (
            "Days 90–180",
            "Harden",
            [
                "Real I/Q adapter if radar exists",
                "RGB-T if they already have dual sensors",
                "Night / weather trial with a false-alarm budget",
                "Retrain loop a non-hero engineer can run",
                "Written limitation list for capture teams",
            ],
        ),
        (
            "Days 180–365",
            "Product",
            [
                "Two SKUs: tower and vehicle or UAV payload software",
                "Qualification and accreditation support",
                "Licensing: evaluation → assignment / exclusive vertical",
                "Edge performance envelope in writing",
                "Operator training pack, not a GitHub README",
            ],
        ),
    ]
    x = 0.55 * inch
    for t, tag, bullets in phases:
        card(c, x, 1.05 * inch, 4.0 * inch, 4.65 * inch)
        c.setFillColor(NAVY)
        c.rect(x, 5.35 * inch, 4.0 * inch, 0.35 * inch, fill=1, stroke=0)
        c.setFillColor(GOLD)
        c.setFont(FONT_B, 9)
        c.drawString(x + 0.22 * inch, 5.46 * inch, t.upper())
        c.setFillColor(NAVY)
        c.setFont(FONT_B, 18)
        c.drawString(x + 0.22 * inch, 4.85 * inch, tag)
        by = 4.4 * inch
        for b in bullets:
            c.setFillColor(GOLD)
            c.circle(x + 0.32 * inch, by + 3, 2.1, fill=1, stroke=0)
            para(c, b, x + 0.48 * inch, by, 3.25 * inch, size=10.5, color=MUTED, leading=14)
            by -= 0.58 * inch
        x += 4.2 * inch
    footer(c, 9)
    c.showPage()


def page_ask(c: canvas.Canvas) -> None:
    draw_bg(c)
    heading(c, "09  —  The ask", "Take this model forward with us.")
    para(
        c,
        "We are offering the working stack, the training pipeline, and a 90-day joint sprint. You bring cameras, a product owner, and the first customer shape.",
        0.55 * inch,
        6.15 * inch,
        12.2 * inch,
        size=12.5,
        leading=18,
    )

    c.setFillColor(NAVY)
    c.roundRect(0.55 * inch, 0.95 * inch, 7.35 * inch, 4.75 * inch, 8, fill=1, stroke=0)
    c.setFillColor(GOLD)
    c.setFont(FONT_B, 9)
    c.drawString(0.85 * inch, 5.28 * inch, "WHAT THE COMPANY GETS")
    gets = [
        "Source software: gated YOLO, HDC matcher, EKF tracker, RF sim chain",
        "A Windows prototype that already runs on NVIDIA GPU",
        "A staged training recipe (aerial people/vehicles → drones → armor/air → Open Images)",
        "Evaluation license for the pilot period, with a path to assignment or exclusive fielding in one vertical",
        "A written limitation list so capture teams do not over-promise",
    ]
    y = 4.85 * inch
    for i, line in enumerate(gets, 1):
        c.setFillColor(GOLD)
        c.setFont(FONT_B, 11)
        c.drawString(0.85 * inch, y, f"{i:02d}")
        para(c, line, 1.25 * inch, y, 6.3 * inch, size=11, color=white, leading=14)
        y -= 0.7 * inch

    card(c, 8.1 * inch, 2.85 * inch, 4.7 * inch, 2.85 * inch)
    c.setFillColor(NAVY)
    c.setFont(FONT_B, 9)
    c.drawString(8.35 * inch, 5.3 * inch, "WHAT WE NEED FROM YOU")
    need = [
        "90-day paid pilot + named product owner",
        "Operational video (and RF if you have it) under NDA",
        "One target SKU: tower, vehicle, or payload software",
        "1–2 engineers to own integration",
    ]
    y = 4.95 * inch
    for line in need:
        c.setFillColor(GOLD)
        c.circle(8.5 * inch, y + 3, 2.2, fill=1, stroke=0)
        para(c, line, 8.7 * inch, y, 3.85 * inch, size=10, color=INK, leading=13)
        y -= 0.48 * inch

    card(c, 8.1 * inch, 0.95 * inch, 4.7 * inch, 1.7 * inch)
    c.setFillColor(NAVY)
    c.setFont(FONT_B, 9)
    c.drawString(8.35 * inch, 2.28 * inch, "NEXT MEETING  ·  90 MIN")
    para(
        c,
        "Live track on your sample video, then a SKU decision. Contact: name / email / phone before send.",
        8.35 * inch,
        1.98 * inch,
        4.2 * inch,
        size=10,
        color=MUTED,
        leading=14,
    )
    footer(c, 10)
    c.showPage()


def build() -> Path:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUT), pagesize=(W, H))
    c.setTitle("TACYOLO — Partner briefing")
    c.setAuthor("TACYOLO")
    c.setSubject("Software-defined detection and tracking — confidential partner pitch")
    page_cover(c)
    page_problem(c)
    page_product(c)
    page_classes(c)
    page_architecture(c)
    page_traction(c)
    page_honesty(c)
    page_market(c)
    page_roadmap(c)
    page_ask(c)
    c.save()
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"wrote {path}")
