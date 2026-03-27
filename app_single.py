from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from openai import OpenAI
from page_analysis import analyze_page
from classify_gpt5_nano import classify_page
from batch_utils import cached_tokens_from_usage, safe_int
from bundle_utils import update_bundle, write_bundle

APP_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = APP_DIR / "artifacts_single"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MAX_IMAGE_BYTES = 32 * 1024 * 1024
POSTPROCESS_MODEL = os.getenv("OPENAI_POSTPROCESS_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-5-nano"
POSTPROCESS_ENABLED = os.getenv("OPENAI_POSTPROCESS_PRINTED", "on").strip().lower() in {"1", "true", "yes", "on"}
POSTPROCESS_MARGIN = int(os.getenv("OPENAI_POSTPROCESS_MARGIN", "2"))
POSTPROCESS_MAX_ROWS = int(os.getenv("OPENAI_POSTPROCESS_MAX_ROWS", "400"))


def _render_gray_image_pdf(pdf_path: Path, page_number: int) -> bytes | None:
    try:
        import fitz
    except Exception:
        return None
    dpi_candidates = [300, 250, 200, 150, 120, 96]
    with fitz.open(pdf_path) as doc:
        if page_number < 1 or page_number > doc.page_count:
            return None
        page = doc.load_page(page_number - 1)
        data = None
        for dpi in dpi_candidates:
            scale = dpi / 72.0
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csGRAY)
            data = pix.tobytes("png")
            if len(data) <= MAX_IMAGE_BYTES:
                return data
        return data


def _encode_png_with_limit(img, max_bytes: int) -> bytes:
    from PIL import Image

    buf = BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    if len(data) <= max_bytes:
        return data
    current = img
    scale = 0.85
    while len(data) > max_bytes and min(current.size) > 256:
        new_size = (max(1, int(current.width * scale)), max(1, int(current.height * scale)))
        current = current.resize(new_size, Image.LANCZOS)
        buf = BytesIO()
        current.save(buf, format="PNG")
        data = buf.getvalue()
    return data


def _render_gray_image_tiff(tiff_path: Path) -> bytes | None:
    try:
        from PIL import Image
    except Exception:
        return None
    with Image.open(tiff_path) as im:
        frame = im.convert("L")
        return _encode_png_with_limit(frame, MAX_IMAGE_BYTES)


def _render_gray_image_tiff_bytes(tiff_bytes: bytes) -> bytes | None:
    try:
        from PIL import Image
    except Exception:
        return None
    with Image.open(BytesIO(tiff_bytes)) as im:
        frame = im.convert("L")
        return _encode_png_with_limit(frame, MAX_IMAGE_BYTES)


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "N/A"
    try:
        value = float(seconds)
    except Exception:
        return "N/A"
    if value < 60:
        return f"{value:,.2f}s" if value >= 1 else f"{value:,.3f}s"
    if value < 3600:
        minutes, secs = divmod(value, 60)
        return f"{int(minutes)}m {secs:04.1f}s"
    hours, rem = divmod(value, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{int(hours)}h {int(minutes)}m {secs:04.1f}s"


def _sort_key_for_name(name: str) -> tuple:
    stem = Path(name).stem
    match = re.search(r"(\d+)", stem)
    if match:
        return (0, int(match.group(1)), stem.lower())
    return (1, stem.lower())


def _is_valid_tiff_entry(name: str) -> bool:
    path = Path(name)
    if any(part == "__MACOSX" for part in path.parts):
        return False
    if path.name.startswith("._"):
        return False
    return path.suffix.lower() in {".tif", ".tiff"}


def _parse_classification(value: str) -> tuple[str, str]:
    if not value:
        return "", ""
    value = value.strip()
    match = re.match(r"^\s*\d+\s*::\s*(.*?)\s*-\s*(.*)\s*$", value)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    match = re.match(r"^\s*\d+\s*-\s*(.*)\s*$", value)
    if match:
        return "", match.group(1).strip()
    return "", value


def _format_classification(page_number: int, printed_page_number: str, label: str) -> str:
    printed_page_number = (printed_page_number or "").strip()
    label = (label or "").strip()
    if printed_page_number:
        return f"{page_number}::{printed_page_number} - {label}"
    return f"{page_number} - {label}"


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    in_string = False
    escape = False
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "\"":
                in_string = False
            continue
        if ch == "\"":
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _postprocess_printed_page_numbers(csv_path: Path) -> dict:
    if not POSTPROCESS_ENABLED:
        return {"skipped": "disabled"}
    if not csv_path.exists():
        return {"skipped": "missing_csv"}

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
        fieldnames = reader.fieldnames or []

    if not rows or "page_number" not in fieldnames or "classification" not in fieldnames:
        return {"skipped": "empty_or_invalid"}

    parsed = []
    for row in rows:
        page_number = safe_int(row.get("page_number"), 0)
        printed_page_number, label = _parse_classification(row.get("classification", ""))
        parsed.append(
            {
                "page_number": page_number,
                "printed_page_number": printed_page_number,
                "label": label,
            }
        )

    parsed.sort(key=lambda r: r["page_number"])
    if len(parsed) <= (POSTPROCESS_MARGIN * 2 + 1):
        return {"skipped": "too_few_pages"}

    min_page = parsed[0]["page_number"]
    max_page = parsed[-1]["page_number"]
    candidates = []
    for idx, row in enumerate(parsed):
        if row["page_number"] <= min_page + POSTPROCESS_MARGIN:
            continue
        if row["page_number"] >= max_page - POSTPROCESS_MARGIN:
            continue
        if row["label"] == "No Page Number":
            continue
        prev_row = parsed[idx - 1] if idx > 0 else None
        next_row = parsed[idx + 1] if idx + 1 < len(parsed) else None
        if not row["printed_page_number"]:
            continue
        candidates.append(
            {
                "page_number": row["page_number"],
                "label": row["label"],
                "printed_page_number": row["printed_page_number"],
                "prev_printed_page_number": prev_row["printed_page_number"] if prev_row else "",
                "next_printed_page_number": next_row["printed_page_number"] if next_row else "",
            }
        )

    if not candidates:
        return {"skipped": "no_candidates"}

    if len(candidates) > POSTPROCESS_MAX_ROWS:
        candidates = candidates[:POSTPROCESS_MAX_ROWS]

    prompt = (
        "You clean up printed page numbers in a CSV of page classifications.\n"
        "Input is JSON with rows from the middle of a book. Each row has:\n"
        "page_number, label, printed_page_number, prev_printed_page_number, next_printed_page_number.\n"
        "Task: Correct printed_page_number only when it is clearly wrong given neighbors "
        "(e.g., obvious OCR error or impossible jump). Do NOT invent numbers or fill in missing "
        "printed_page_number values. Leave as-is when unsure. Preserve section ranges like "
        "1172-1180 and non-numeric values unless the error is obvious. Do not change labels. "
        "Never add a printed page number for rows labeled \"No Page Number\".\n"
        "Return strict JSON only:\n"
        "{ \"corrections\": [ {\"page_number\": 12, \"printed_page_number\": \"37\"} ] }\n"
        "If no changes, return {\"corrections\": []}.\n\n"
        f"rows:\n{json.dumps(candidates, ensure_ascii=False, indent=2)}"
    )

    client = OpenAI()
    response = client.responses.create(
        model=POSTPROCESS_MODEL,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": "You return strict JSON only."}]},
            {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
        ],
    )

    raw_text = getattr(response, "output_text", "") or ""
    raw_text = raw_text.strip()
    if not raw_text:
        return {"skipped": "empty_response"}

    candidate_json = _extract_first_json_object(raw_text)
    if not candidate_json:
        return {"skipped": "no_json"}

    try:
        data = json.loads(candidate_json)
    except Exception:
        return {"skipped": "invalid_json"}

    corrections = data.get("corrections")
    if not isinstance(corrections, list):
        return {"skipped": "no_corrections_list"}

    corrections_map: dict[int, str] = {}
    for item in corrections:
        if not isinstance(item, dict):
            continue
        page_number = safe_int(item.get("page_number"), 0)
        if page_number <= 0:
            continue
        printed_page_number = str(item.get("printed_page_number", "")).strip()
        corrections_map[page_number] = printed_page_number

    if not corrections_map:
        return {"corrections": 0}

    updated = 0
    for row in rows:
        page_number = safe_int(row.get("page_number"), 0)
        if page_number in corrections_map:
            printed_page_number, label = _parse_classification(row.get("classification", ""))
            row["classification"] = _format_classification(
                page_number, corrections_map[page_number], label
            )
            updated += 1

    if updated:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return {"corrections": updated}


def _list_tiffs_in_zip(zip_bytes: bytes) -> list[str]:
    with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
        names = [
            info.filename
            for info in zf.infolist()
            if not info.is_dir() and _is_valid_tiff_entry(info.filename)
        ]
    return sorted(names, key=_sort_key_for_name)


def _extract_tiffs(zip_bytes: bytes, dest_dir: Path) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
        infos = [
            info
            for info in zf.infolist()
            if not info.is_dir() and _is_valid_tiff_entry(info.filename)
        ]
        infos = sorted(infos, key=lambda info: _sort_key_for_name(info.filename))
        for info in infos:
            out_path = dest_dir / Path(info.filename).name
            with zf.open(info) as src, out_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(out_path)
    return extracted


def _pdf_page_count(pdf_bytes: bytes) -> int:
    try:
        import fitz
    except Exception:
        return 0
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return doc.page_count


def _crop_pdf(full_pdf: Path, cropped_pdf: Path, start_page: int, end_page: int) -> None:
    import fitz

    with fitz.open(full_pdf) as doc:
        cropped = fitz.open()
        for page_num in range(start_page, end_page + 1):
            cropped.insert_pdf(doc, from_page=page_num - 1, to_page=page_num - 1)
        cropped.save(cropped_pdf)


def _crop_pdf_bytes(pdf_bytes: bytes, start_page: int, end_page: int) -> bytes:
    import fitz

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        cropped = fitz.open()
        for page_num in range(start_page, end_page + 1):
            cropped.insert_pdf(doc, from_page=page_num - 1, to_page=page_num - 1)
        data = cropped.write()
        cropped.close()
    return data


def _ensure_preview_cached(
    *,
    file_ext: str,
    file_bytes: bytes,
    active_start: int,
    active_end: int,
    preview_key: str,
) -> None:
    if st.session_state.get("preview_key") == preview_key:
        return
    st.session_state["preview_pdf_bytes"] = None
    st.session_state["preview_tiff_bytes"] = []

    if file_ext == ".pdf":
        try:
            cropped_bytes = _crop_pdf_bytes(file_bytes, active_start, active_end)
        except Exception:
            cropped_bytes = None
        st.session_state["preview_pdf_bytes"] = cropped_bytes
    elif file_ext == ".zip":
        names = _list_tiffs_in_zip(file_bytes)
        selected = names[active_start - 1 : active_end]
        previews: list[dict] = []
        with zipfile.ZipFile(BytesIO(file_bytes)) as zf:
            for name in selected:
                try:
                    data = zf.read(name)
                except Exception:
                    continue
                img_bytes = _render_gray_image_tiff_bytes(data)
                if img_bytes:
                    previews.append({"name": Path(name).name, "bytes": img_bytes})
        st.session_state["preview_tiff_bytes"] = previews

    st.session_state["preview_key"] = preview_key


st.set_page_config(page_title="Batch Page Analyzer", layout="wide")


def _safe_read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_single_session(run_dir: Path) -> None:
    logs_path = run_dir / "logs.json"
    logs_bundle = _safe_read_json(logs_path) if logs_path.exists() else {}

    logs_data: list[dict] = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "cached_tokens": 0}
    total_cost = 0.0
    total_elapsed_seconds = 0.0

    if isinstance(logs_bundle, dict):
        for page_key in sorted(logs_bundle.keys(), key=lambda x: safe_int(x, 0)):
            entry = logs_bundle.get(page_key)
            if not isinstance(entry, dict):
                continue
            usage = entry.get("usage") or {}
            pricing = entry.get("pricing") or {}
            token_breakdown = entry.get("token_breakdown") or {}

            prompt_tokens = safe_int(token_breakdown.get("prompt_tokens", usage.get("input_tokens", 0)), 0)
            completion_tokens = safe_int(token_breakdown.get("completion_tokens", usage.get("output_tokens", 0)), 0)
            reasoning_tokens = safe_int(token_breakdown.get("reasoning_tokens", 0), 0)
            cached_tokens = safe_int(token_breakdown.get("cached_tokens", cached_tokens_from_usage(usage)), 0)

            total_usage["prompt_tokens"] += prompt_tokens
            total_usage["completion_tokens"] += completion_tokens
            total_usage["reasoning_tokens"] += reasoning_tokens
            total_usage["cached_tokens"] += cached_tokens
            total_cost += float(pricing.get("estimated_cost_usd", 0.0) or 0.0)
            total_elapsed_seconds += float(entry.get("elapsed_seconds", 0.0) or 0.0)

            logs_data.append(
                {
                    "source_file": entry.get("source_file", ""),
                    "page_number": entry.get("page_number", page_key),
                    "elapsed_seconds": entry.get("elapsed_seconds"),
                    "usage": usage,
                    "pricing": pricing,
                    "token_breakdown": token_breakdown,
                }
            )

    csv_matches = list(run_dir.glob("*_tagged.csv"))
    csv_path = csv_matches[0] if csv_matches else None

    inputs_dir = run_dir / "inputs"
    input_kind = run_dir.parent.name if run_dir.parent else None
    preview_pdf_path = None
    preview_tiffs: list[Path] = []
    preview_pdf_bytes = None
    preview_tiff_previews: list[dict] = []

    if input_kind == "pdf":
        pdf_matches = list(inputs_dir.glob("*_cropped_*.pdf"))
        if not pdf_matches:
            pdf_matches = list(inputs_dir.glob("*.pdf"))
        preview_pdf_path = pdf_matches[0] if pdf_matches else None
        if preview_pdf_path and preview_pdf_path.exists():
            try:
                preview_pdf_bytes = preview_pdf_path.read_bytes()
            except Exception:
                preview_pdf_bytes = None
    else:
        preview_tiffs = sorted(inputs_dir.glob("*.tif*")) if inputs_dir.exists() else []
        for path in preview_tiffs:
            img_bytes = _render_gray_image_tiff(path)
            if img_bytes:
                preview_tiff_previews.append({"name": path.name, "bytes": img_bytes})

    st.session_state["run_dir"] = str(run_dir)
    st.session_state["csv_path"] = str(csv_path) if csv_path else None
    st.session_state["log_totals"] = {
        "usage": total_usage,
        "estimated_cost_usd": total_cost,
        "total_elapsed_seconds": total_elapsed_seconds,
        "batch_elapsed_seconds": total_elapsed_seconds,
        "total_pages": len(logs_data),
    }
    st.session_state["logs_data"] = logs_data
    st.session_state["input_kind"] = input_kind
    st.session_state["preview_pdf_path"] = str(preview_pdf_path) if preview_pdf_path else None
    st.session_state["preview_tiffs"] = [str(p) for p in preview_tiffs]
    st.session_state["preview_pdf_bytes"] = preview_pdf_bytes
    st.session_state["preview_tiff_bytes"] = preview_tiff_previews
    st.session_state["preview_key"] = f"loaded:{run_dir.name}"
    st.session_state["preview_locked"] = False
    st.session_state["selection_locked"] = False
    st.session_state["last_run_key"] = hashlib.sha256(str(run_dir).encode("utf-8")).hexdigest()
    st.session_state["loaded_session_dir"] = str(run_dir)

st.session_state.setdefault("run_dir", None)
st.session_state.setdefault("csv_path", None)
st.session_state.setdefault("log_totals", None)
st.session_state.setdefault("logs_data", [])
st.session_state.setdefault("last_run_key", None)
st.session_state.setdefault("input_kind", None)
st.session_state.setdefault("preview_pdf_path", None)
st.session_state.setdefault("preview_tiffs", [])
st.session_state.setdefault("preview_locked", False)
st.session_state.setdefault("locked_start", None)
st.session_state.setdefault("locked_end", None)
st.session_state.setdefault("locked_hash", None)
st.session_state.setdefault("preview_pdf_bytes", None)
st.session_state.setdefault("preview_tiff_bytes", [])
st.session_state.setdefault("preview_key", None)
st.session_state.setdefault("selection_locked", False)
st.session_state.setdefault("loaded_session_dir", None)

st.sidebar.header("Inputs")
uploaded = st.sidebar.file_uploader(
    "Upload PDF or ZIP of TIFFs",
    type=["pdf", "zip"],
    disabled=st.session_state.get("preview_locked", False),
)

file_bytes = None
file_name = None
file_ext = None
file_stem = None
file_hash = None
page_count = 0
tiff_names: list[str] = []

if uploaded is not None:
    file_bytes = uploaded.getbuffer().tobytes()
    file_name = Path(uploaded.name).name
    file_ext = Path(file_name).suffix.lower()
    file_stem = Path(file_name).stem
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    if file_ext == ".pdf":
        page_count = _pdf_page_count(file_bytes)
    elif file_ext == ".zip":
        tiff_names = _list_tiffs_in_zip(file_bytes)
        page_count = len(tiff_names)

loaded_session_dir = st.session_state.get("loaded_session_dir")
using_loaded_session = bool(loaded_session_dir) and uploaded is None
should_stop = False

if uploaded is None and not using_loaded_session:
    st.sidebar.info("Upload a PDF or ZIP file to begin.")
    should_stop = True

if uploaded is not None and page_count <= 0:
    st.sidebar.error("No pages found in the uploaded file.")
    should_stop = True

if using_loaded_session:
    input_kind = st.session_state.get("input_kind")
    file_ext = ".pdf" if input_kind == "pdf" else ".zip"
    file_hash = st.session_state.get("last_run_key") or str(loaded_session_dir)
    loaded_totals = st.session_state.get("log_totals") or {}
    loaded_logs_data = st.session_state.get("logs_data") or []
    page_count = safe_int(loaded_totals.get("total_pages", 0), 0) or len(loaded_logs_data) or 1
    st.sidebar.caption(f"Loaded pages: {page_count}")
    start_page = 1
    end_page = page_count
    preview_ready = bool(st.session_state.get("preview_pdf_bytes") or st.session_state.get("preview_tiff_bytes"))
    tag_clicked = False
else:
    if not should_stop:
        st.sidebar.caption(f"Detected pages: {page_count}")
        inputs_disabled = st.session_state.get("preview_locked", False) or st.session_state.get(
            "selection_locked", False
        )
        start_page = st.sidebar.number_input(
            "Start page",
            min_value=1,
            max_value=page_count,
            value=1,
            step=1,
            disabled=inputs_disabled,
        )
        end_page = st.sidebar.number_input(
            "End page",
            min_value=1,
            max_value=page_count,
            value=page_count,
            step=1,
            disabled=inputs_disabled,
        )

        active_start_for_preview = (
            st.session_state.get("locked_start") if st.session_state.get("preview_locked") else start_page
        )
        active_end_for_preview = (
            st.session_state.get("locked_end") if st.session_state.get("preview_locked") else end_page
        )
        active_start_for_preview = safe_int(active_start_for_preview, start_page)
        active_end_for_preview = safe_int(active_end_for_preview, end_page)
        preview_key = f"{file_hash}:{active_start_for_preview}:{active_end_for_preview}:{file_ext}"
        _ensure_preview_cached(
            file_ext=file_ext,
            file_bytes=file_bytes,
            active_start=active_start_for_preview,
            active_end=active_end_for_preview,
            preview_key=preview_key,
        )
        preview_ready = False
        if st.session_state.get("preview_key") == preview_key:
            if file_ext == ".pdf":
                preview_ready = st.session_state.get("preview_pdf_bytes") is not None
            elif file_ext == ".zip":
                preview_ready = bool(st.session_state.get("preview_tiff_bytes"))

        tag_clicked = st.sidebar.button(
            "Tag",
            type="primary",
            disabled=st.session_state.get("preview_locked", False) or not preview_ready,
        )
    else:
        tag_clicked = False
tag_spinner_slot = st.sidebar.empty()

if file_hash and st.session_state.get("locked_hash") and st.session_state.get("locked_hash") != file_hash:
    st.session_state["preview_locked"] = False
    st.session_state["selection_locked"] = False
    st.session_state["locked_start"] = None
    st.session_state["locked_end"] = None
    st.session_state["locked_hash"] = None

if tag_clicked:
    if start_page > end_page:
        st.sidebar.error("Start page must be <= end page.")
        st.stop()

    st.session_state["preview_locked"] = True
    st.session_state["selection_locked"] = True
    st.session_state["locked_start"] = start_page
    st.session_state["locked_end"] = end_page
    st.session_state["locked_hash"] = file_hash

    timestamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S")
    base_dir = ARTIFACTS_DIR / ("pdf" if file_ext == ".pdf" else "tiff")
    base_dir.mkdir(parents=True, exist_ok=True)
    run_dir = base_dir / f"{file_stem}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    inputs_dir = run_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    csv_rows: list[dict] = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "cached_tokens": 0}
    total_cost = 0.0
    total_elapsed_seconds = 0.0
    logs_data: list[dict] = []
    batch_started_at = time.perf_counter()
    analysis_bundle: dict[str, dict] = {}
    logs_bundle: dict[str, dict] = {}

    if file_ext == ".pdf":
        full_pdf_path = inputs_dir / file_name
        full_pdf_path.write_bytes(file_bytes)
        cropped_pdf_path = inputs_dir / f"{file_stem}_cropped_{start_page}_{end_page}.pdf"
        _crop_pdf(full_pdf_path, cropped_pdf_path, start_page, end_page)

        page_entries = [(page_num, full_pdf_path) for page_num in range(start_page, end_page + 1)]
        preview_pdf_path = cropped_pdf_path
        preview_tiffs: list[Path] = []
    else:
        extracted = _extract_tiffs(file_bytes, inputs_dir)
        extracted = sorted(extracted, key=lambda p: _sort_key_for_name(p.name))
        selected = extracted[start_page - 1 : end_page]

        # Delete non-selected TIFFs after "cropping".
        for path in extracted:
            if path not in selected:
                try:
                    path.unlink()
                except Exception:
                    pass

        page_entries = [(start_page + idx, path) for idx, path in enumerate(selected)]
        preview_pdf_path = None
        preview_tiffs = selected

    total_pages = len(page_entries)
    for idx, (page_num, source_path) in enumerate(page_entries, start=1):
        with tag_spinner_slot:
            with st.spinner(f"Tagging page {idx} out of {total_pages}"):
                page_started_at = time.perf_counter()
                if file_ext == ".pdf":
                    page_id = f"{file_stem}_p{page_num:06d}"
                    display_name = f"{file_name} — page {page_num}"
                    analysis = analyze_page(source_path, page_number=page_num)
                    img_bytes = _render_gray_image_pdf(source_path, page_num)
                else:
                    page_id = Path(source_path).stem
                    display_name = Path(source_path).name
                    analysis = analyze_page(source_path)
                    img_bytes = _render_gray_image_tiff(source_path)

                analysis["source_file"] = display_name
                analysis["source_page_number"] = page_num

                update_bundle(analysis_bundle, page_num, analysis)

                if not img_bytes:
                    raise RuntimeError(f"Failed to render page image: {display_name}")

                image_path: Path | None = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                        tmp.write(img_bytes)
                        image_path = Path(tmp.name)

                    result, log_text, _raw = classify_page(analysis, image_path)
                    elapsed_seconds = round(time.perf_counter() - page_started_at, 3)
                finally:
                    if image_path:
                        try:
                            image_path.unlink()
                        except Exception:
                            pass

                try:
                    log_obj = json.loads(log_text)
                except Exception:
                    log_obj = {}
                usage = log_obj.get("usage") or {}
                pricing = log_obj.get("pricing") or {}
                token_breakdown = log_obj.get("token_breakdown") or {}

                prompt_tokens = safe_int(token_breakdown.get("prompt_tokens", usage.get("input_tokens", 0)), 0)
                completion_tokens = safe_int(
                    token_breakdown.get("completion_tokens", usage.get("output_tokens", 0)), 0
                )
                reasoning_tokens = safe_int(token_breakdown.get("reasoning_tokens", 0), 0)
                cached_tokens = safe_int(token_breakdown.get("cached_tokens", cached_tokens_from_usage(usage)), 0)

                total_usage["prompt_tokens"] += prompt_tokens
                total_usage["completion_tokens"] += completion_tokens
                total_usage["reasoning_tokens"] += reasoning_tokens
                total_usage["cached_tokens"] += cached_tokens
                total_cost += float(pricing.get("estimated_cost_usd", 0.0) or 0.0)
                total_elapsed_seconds += elapsed_seconds

                logs_data.append(
                    {
                        "source_file": display_name,
                        "page_number": page_num,
                        "elapsed_seconds": elapsed_seconds,
                        "usage": usage,
                        "pricing": pricing,
                        "token_breakdown": token_breakdown,
                    }
                )
                log_obj["elapsed_seconds"] = elapsed_seconds
                log_obj["source_file"] = display_name
                log_obj["page_number"] = page_num
                update_bundle(logs_bundle, page_num, log_obj)

                csv_rows.append(
                    {
                        "page_number": page_num,
                        "classification": (
                            f"{page_num}::{result.get('printed_page_number', '')} - {result.get('label', '')}"
                            if result.get("printed_page_number")
                            else f"{page_num} - {result.get('label', '')}"
                        ),
                        "error": "",
                    }
                )
    tag_spinner_slot.empty()

    csv_path = run_dir / f"{file_stem}_{timestamp}_tagged.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["page_number", "classification", "error"],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    postprocess_summary = {}
    with st.spinner("Post-processing printed page numbers..."):
        try:
            postprocess_summary = _postprocess_printed_page_numbers(csv_path)
        except Exception as exc:
            postprocess_summary = {"error": str(exc)}

    batch_elapsed_seconds = round(time.perf_counter() - batch_started_at, 3)

    st.session_state["run_dir"] = str(run_dir)
    st.session_state["csv_path"] = str(csv_path)
    st.session_state["log_totals"] = {
        "usage": total_usage,
        "estimated_cost_usd": total_cost,
        "total_elapsed_seconds": total_elapsed_seconds,
        "batch_elapsed_seconds": batch_elapsed_seconds,
    }
    st.session_state["postprocess_summary"] = postprocess_summary
    st.session_state["logs_data"] = logs_data
    st.session_state["input_kind"] = "pdf" if file_ext == ".pdf" else "tiff"
    st.session_state["preview_pdf_path"] = str(preview_pdf_path) if preview_pdf_path else None
    st.session_state["preview_tiffs"] = [str(p) for p in preview_tiffs]
    st.session_state["last_run_key"] = hashlib.sha256(
        (file_name + str(start_page) + str(end_page) + timestamp).encode("utf-8")
    ).hexdigest()
    st.session_state["preview_locked"] = False

    analysis_path = run_dir / "analysis.json"
    write_bundle(analysis_path, analysis_bundle)
    logs_path = run_dir / "logs.json"
    write_bundle(logs_path, logs_bundle)

st.sidebar.header("Load Session")
session_dirs: list[Path] = []
for kind_dir in ARTIFACTS_DIR.iterdir():
    if not kind_dir.is_dir():
        continue
    for run_dir in kind_dir.iterdir():
        if run_dir.is_dir():
            session_dirs.append(run_dir)
session_dirs = sorted(session_dirs, key=lambda p: p.name, reverse=True)
session_labels = ["(none)"] + [f"{p.parent.name}/{p.name}" for p in session_dirs]
selected_session = st.sidebar.selectbox("Previous runs", session_labels, index=0)
load_clicked = st.sidebar.button("Load Session")
clear_clicked = st.sidebar.button("Clear Loaded Session")

if load_clicked and selected_session != "(none)":
    selected_path = ARTIFACTS_DIR / selected_session
    if selected_path.exists():
        _load_single_session(selected_path)
        st.rerun()

if clear_clicked:
    st.session_state["loaded_session_dir"] = None
    st.session_state["run_dir"] = None
    st.session_state["csv_path"] = None
    st.session_state["log_totals"] = None
    st.session_state["logs_data"] = []
    st.session_state["input_kind"] = None
    st.session_state["preview_pdf_path"] = None
    st.session_state["preview_tiffs"] = []
    st.session_state["preview_pdf_bytes"] = None
    st.session_state["preview_tiff_bytes"] = []
    st.session_state["preview_key"] = None
    st.session_state["preview_locked"] = False
    st.session_state["selection_locked"] = False
    st.session_state["last_run_key"] = None
    st.rerun()

if should_stop:
    st.stop()

run_dir = Path(st.session_state["run_dir"]) if st.session_state.get("run_dir") else None
csv_path = Path(st.session_state["csv_path"]) if st.session_state.get("csv_path") else None
log_totals = st.session_state.get("log_totals") or {}
logs_data = st.session_state.get("logs_data") or []
input_kind = st.session_state.get("input_kind")
preview_pdf_path = Path(st.session_state["preview_pdf_path"]) if st.session_state.get("preview_pdf_path") else None
preview_tiffs = [Path(p) for p in st.session_state.get("preview_tiffs", [])]

active_start = st.session_state.get("locked_start") if st.session_state.get("preview_locked") else start_page
active_end = st.session_state.get("locked_end") if st.session_state.get("preview_locked") else end_page
active_start = safe_int(active_start, start_page)
active_end = safe_int(active_end, end_page)
preview_key = f"{file_hash}:{active_start}:{active_end}:{file_ext}"

if not run_dir or not csv_path:
    st.info("Click Tag to process the selected pages.")

tabs = st.tabs(["Preview", "Classification", "Logs"])
preview_tab, classification_tab, logs_tab = tabs

with preview_tab:
    if file_ext == ".pdf":
        cropped_bytes = st.session_state.get("preview_pdf_bytes")
        if cropped_bytes:
            b64 = __import__("base64").b64encode(cropped_bytes).decode("ascii")
            iframe = (
                f'<iframe src="data:application/pdf;base64,{b64}" '
                f'width="100%" height="900" type="application/pdf"></iframe>'
            )
            st.components.v1.html(iframe, height=900, scrolling=True)
        else:
            st.write("No PDF preview available for the selected range.")
    elif file_ext == ".zip":
        previews = st.session_state.get("preview_tiff_bytes") or []
        if not previews:
            st.write("No TIFF previews available for the selected range.")
        else:
            for item in previews:
                title = item.get("name") or "TIFF"
                with st.expander(title, expanded=False):
                    st.image(item.get("bytes"), width="stretch")
    else:
        st.write("No preview available.")

with classification_tab:
    if csv_path and csv_path.exists():
        df = pd.read_csv(csv_path)
        styled = df.style.set_properties(
            **{
                "white-space": "pre-wrap",
                "text-align": "center",
                "vertical-align": "middle",
            }
        ).set_table_styles(
            [
                {"selector": "th", "props": [("text-align", "center"), ("vertical-align", "middle")]}
            ]
        )
        st.dataframe(styled, width="stretch")
    else:
        st.write("No classification results.")

with logs_tab:
    usage = (log_totals or {}).get("usage") or {}
    total_cost = (log_totals or {}).get("estimated_cost_usd", 0.0)
    total_elapsed_seconds = float((log_totals or {}).get("total_elapsed_seconds", 0.0) or 0.0)
    batch_elapsed_seconds = float((log_totals or {}).get("batch_elapsed_seconds", 0.0) or 0.0)

    prompt_tokens = safe_int(usage.get("prompt_tokens", 0), 0)
    completion_tokens = safe_int(usage.get("completion_tokens", 0), 0)
    reasoning_tokens = safe_int(usage.get("reasoning_tokens", 0), 0)
    cached_tokens = safe_int(usage.get("cached_tokens", 0), 0)
    total_tokens = prompt_tokens + completion_tokens + reasoning_tokens

    st.metric("Total tokens", f"{total_tokens:,}")
    st.metric("Estimated cost (USD)", f"${total_cost:,.4f}")
    st.metric("Total classify time", _format_duration(total_elapsed_seconds))
    st.metric("Wall time", _format_duration(batch_elapsed_seconds))

    st.write(
        {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cached_tokens": cached_tokens,
        }
    )

    if logs_data:
        rows = []
        for entry in logs_data:
            usage = entry.get("usage") or {}
            pricing = entry.get("pricing") or {}
            token_breakdown = entry.get("token_breakdown") or {}
            rows.append(
                {
                    "source_file": entry.get("source_file", ""),
                    "page_number": entry.get("page_number", ""),
                    "elapsed_time": _format_duration(entry.get("elapsed_seconds")),
                    "prompt_tokens": token_breakdown.get("prompt_tokens", usage.get("input_tokens", "")),
                    "completion_tokens": token_breakdown.get("completion_tokens", usage.get("output_tokens", "")),
                    "reasoning_tokens": token_breakdown.get("reasoning_tokens", ""),
                    "cached_tokens": token_breakdown.get("cached_tokens", ""),
                    "estimated_cost_usd": pricing.get("estimated_cost_usd", ""),
                }
            )
        df_logs = pd.DataFrame(rows)
        st.dataframe(df_logs, width="stretch")
