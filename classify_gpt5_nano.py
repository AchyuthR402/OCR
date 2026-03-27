from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import BadRequestError, OpenAI

APP_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = APP_DIR / "reference"
KB_PATH = REFERENCE_DIR / "RamTaggingAI.docx"
EXAMPLE_CLASSIFICATION_DIR = REFERENCE_DIR / "Example Classification"
CATEGORIES_PATH = REFERENCE_DIR / "categories.txt"
REFERENCE_CACHE_PATH = REFERENCE_DIR / ".openai_file_cache.json"

DEFAULT_MODEL_NAME = "gpt-5-nano"
PROMPT_CACHE_KEY = os.getenv("OPENAI_PROMPT_CACHE_KEY", "pagetagging-static-prompt-v1")
PROMPT_CACHE_RETENTION = os.getenv("OPENAI_PROMPT_CACHE_RETENTION", "24h")

DEFAULT_PRICING_PER_1M = {
    "gpt-5-nano": {"input": 0.05, "cached_input": 0.005, "output": 0.40},
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
}


def _get_model_name() -> str:
    return os.getenv("OPENAI_MODEL") or os.getenv("OPEN_AI_MODEL") or DEFAULT_MODEL_NAME


def _load_env() -> None:
    for env_path in (APP_DIR / ".env", APP_DIR.parent / ".env"):
        if not env_path.exists():
            continue
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key:
                    # Treat .env as the source of truth so changing the file updates the model/pricing.
                    os.environ[key] = value
        except Exception:
            pass
        break


def _load_labels(categories_path: Path) -> List[str]:
    labels: List[str] = []
    if categories_path.exists():
        for line in categories_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("image\t"):
                continue
            parts = [p.strip() for p in line.split("\t") if p.strip()]
            if not parts:
                continue
            label = parts[-1]
            if label and label not in labels:
                labels.append(label)
    if not labels:
        labels = [
            "Acknowledgement",
            "Title Page",
            "Table of Contents",
            "Preface",
            "Foreword",
            "Blank page",
            "No Page Number",
            "Section",
            "Illustrations/Pictures",
            "Maps",
            "Tables",
            "Appendix",
            "Index",
            "Supplement",
            "Errata",
            "Annex",
            "Bibliography",
        ]
    return labels


LABELS = _load_labels(CATEGORIES_PATH)

SYSTEM_PROMPT_TEMPLATE = """
Role:
You are a page classifier. You will receive:
1) An analysis JSON string.
2) One page image attached to the request.

Objective:
Classify the page into exactly one label (or empty string if none applies) and return strict JSON only.
The page image is the source of truth. The analysis JSON is a hint only.

Labels (must choose exactly one unless none applies):
- Acknowledgement
- Title Page
- Table of Contents
- Preface
- Foreword
- Blank page
- No Page Number
- Section
- Illustrations/Pictures
- Maps
- Tables
- Appendix
- Index
- Supplement
- Errata
- Annex
- Bibliography

Label guidance:
- Do NOT choose a label just because the word appears in running text. The page must primarily be that type (clear heading and matching structure/content).
- Use Acknowledgement when the page is the first image of a volume and contains donor or digitization credit information with no intellectual content.
- Use Title Page for the first informational page listing title/author and often publisher/place/year; usually not part of pagination.
- Use Table of Contents for chronological listings of a volume’s contents; ordering is by sequence (top-to-bottom or beginning-to-end). If a page is labeled “Index” but is clearly chronological, tag it as Table of Contents.
- Use Preface for an introduction describing subject/scope/aim, usually titled “Preface,” before the main content.
- Use Foreword for a short introduction written by someone other than the author, usually titled “Foreword,” before the main content.
- Use Blank page when the page is empty but is included in pagination (even if the number is not printed). You may need to infer this from surrounding pages.
- Use No Page Number for blank/minimal pages not included in pagination, typically before pagination starts or after it ends. Sometimes a mid-book page with information is not included in pagination; tag it as No Page Number. Completely blank No Page Number pages should not appear within pagination.
- Use Section ONLY when the page explicitly uses section notation (e.g., “Section”, “Sect.”, § or §§) and the content is organized by those section numbers. Do not use Section for general body text without explicit section numbering.
- A plain prose/history page is NOT Section just because it is part of the main body. For example, narrative text such as "CONNECTICUT. OUTLINE HISTORY." without explicit Section/Sect./§ notation must not be labeled Section.
- Use Illustrations/Pictures when the page contains drawings, graphics, photos, or illustrations.
- Use Maps when any map or map-like depiction is present, even as a sketch or illustration.
- Use Tables for tabular information that is not a table of contents.
- Use Appendix for supplemental material labeled Appendix/Appendices, typically near the end.
- Use Index for alphabetical listings of names/subjects with page references (even if called “table” in some countries).
- Use Supplement for a separate additional section that accompanies the publication and is designed to extend/complete/reinforce the original work. It is usually near the end of the volume. Only tag Supplement when it appears as a section within the volume; if the entire volume is a supplement, do not tag as Supplement.
- Use Errata for lists of corrected errors at the front or back.
- Use Annex for supplementary material labeled Annex, usually at the end.
- Use Bibliography for lists of sources/citations used in the work, typically near the end.
- Blank vs No Page Number distinction:
  - If any printed page number appears, do NOT use No Page Number.
  - Blank page is within pagination (surrounded by numbered pages or clearly inside the numbered section), even if the number is not printed.
  - No Page Number is outside pagination (front/back matter not counted) or minimal content not part of pagination.

Decision rules:
- If analysis_json indicates has_image=true, the page likely contains an illustration or map. Choose Illustrations/Pictures or Maps if appropriate.
- If analysis_json indicates has_table=true, choose Tables if appropriate.
- If a page would otherwise be Section or would otherwise remain unlabeled, but contains any illustration, map, or table (even small), use Illustrations/Pictures, Maps, or Tables instead.
- For front-matter labels (Title Page, Preface, Foreword, Table of Contents, Index, Appendix, Supplement, Errata, Annex, Bibliography, Acknowledgement, No Page Number, Blank page), do NOT choose Tables/Illustrations/Maps for incidental content. Only override to Tables/Illustrations/Maps if the page would otherwise be Section or unlabeled.
- For all other categories (Title Page, Table of Contents, etc.), do not override them just because an image/table appears.

Printed page number:
- If a printed page number is visible, capture it exactly (any language or script). Check page borders/edges.
- For Section pages, if section numbers or a section range are present in the body or header/footer, use those section numbers as the printed_page_number instead of the page number (e.g., 1172-1180). Omit the § symbol.
- If multiple page numbers appear (top and bottom, or two pagination systems), choose the one that makes logical sense given analysis_json.page_number (the PDF page order).
- If no printed page number appears on the page, return an empty string.
- Never infer or extrapolate a page number from neighboring pages.
- If the page number uses non-Latin digits, convert them to standard 0-9 (do not convert Roman numerals).

Output (strict JSON only):
{
  "label": "...",
  "printed_page_number": ""
}

Output constraints:
- label must be one of labels, or an empty string if no label applies.
- If there is no printed page number, return empty string.
- If the page does not match any label, return an empty string for label and only the printed_page_number (if present).
- Do not include extra keys or commentary.
""".strip()

USER_PROMPT_TEMPLATE = """
analysis_json:
{analysis_json}

The page image is attached to this request.
""".strip()


def _load_file_cache() -> Dict[str, str]:
    if not REFERENCE_CACHE_PATH.exists():
        return {}
    try:
        raw = REFERENCE_CACHE_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        return {}
    return {}


def _save_file_cache(cache: Dict[str, str]) -> None:
    try:
        REFERENCE_CACHE_PATH.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def _file_fingerprint(path: Path) -> str:
    stat = path.stat()
    return f"{path.resolve()}|{stat.st_size}|{stat.st_mtime}"


def _upload_file(
    client: OpenAI, path: Path, cache: Dict[str, str], purpose: str = "user_data", force: bool = False
) -> str:
    key = _file_fingerprint(path)
    if not force and key in cache:
        return cache[key]
    with path.open("rb") as f:
        created = client.files.create(file=f, purpose=purpose)
    file_id = created.id
    cache[key] = file_id
    _save_file_cache(cache)
    return file_id


def build_system_prompt() -> str:
    return SYSTEM_PROMPT_TEMPLATE


def _normalize_analysis_for_prompt(analysis: Any) -> str:
    data: Any = analysis
    if isinstance(analysis, (str, Path)):
        raw_text = Path(analysis).read_text(encoding="utf-8").strip()
        try:
            data = json.loads(raw_text)
        except Exception:
            return raw_text

    if isinstance(data, dict):
        cleaned = dict(data)
        for key in (
            "source_file",
            "source_page_number",
            "rendered_image",
            "source_path",
            "file_name",
            "filename",
            "input_path",
        ):
            cleaned.pop(key, None)
        return json.dumps(cleaned, ensure_ascii=False, indent=2)

    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        return str(data)


def build_user_prompt(analysis: Any) -> str:
    raw = _normalize_analysis_for_prompt(analysis)
    return USER_PROMPT_TEMPLATE.format(analysis_json=raw)


# --- Codex CLI (kept for reference; unused) ---

def resolve_codex_command() -> str:
    ensure_codex_on_path()
    override = os.getenv("CODEX_CMD") or os.getenv("CODEX_CLI")
    if override:
        return override
    if os.name == "nt":
        candidates = ["codex.cmd", "codex"]
    else:
        candidates = ["codex", "codex.cmd"]
    for cmd in candidates:
        if shutil.which(cmd):
            return cmd
    return candidates[0]


def ensure_codex_on_path() -> None:
    if os.name != "nt":
        return
    if shutil.which("codex") or shutil.which("codex.cmd"):
        return
    candidates: List[str] = []
    codex_home = os.getenv("CODEX_HOME")
    if codex_home:
        candidates.extend(
            [
                str(Path(codex_home)),
                str(Path(codex_home) / "bin"),
                str(Path(codex_home) / "Scripts"),
            ]
        )
    local_appdata = os.getenv("LOCALAPPDATA")
    if local_appdata:
        candidates.extend(
            [
                str(Path(local_appdata) / "Programs" / "codex"),
                str(Path(local_appdata) / "Programs" / "codex" / "bin"),
            ]
        )
    appdata = os.getenv("APPDATA")
    if appdata:
        candidates.append(str(Path(appdata) / "npm"))
    userprofile = os.getenv("USERPROFILE")
    if userprofile:
        candidates.append(str(Path(userprofile) / ".local" / "bin"))
    path = os.environ.get("PATH", "")
    for folder in candidates:
        if folder and Path(folder).exists() and folder not in path:
            path = f"{folder}{os.pathsep}{path}"
    os.environ["PATH"] = path


def run_codex(prompt: str) -> Tuple[str, str]:
    _load_env()
    cmd = resolve_codex_command()
    result = subprocess.run(
        [cmd, "exec", "--skip-git-repo-check", "-"],
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        cwd=str(REFERENCE_DIR),
    )
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    if result.returncode != 0:
        raise RuntimeError(f"codex exec failed ({result.returncode}): {stderr}")
    return stdout, stderr

# --- End Codex CLI ---


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


def _sanitize_json_candidate(text: str) -> str:
    # Remove trailing commas before } or ].
    return re.sub(r",\s*([}\]])", r"\1", text)


SECTION_SYMBOL_RE = re.compile(r"§{1,2}")
SECTION_NUMBERED_RE = re.compile(
    r"(?ix)\b(?:section|sect\.?|sections|sects\.?)\s+"
    r"(?:"
    r"\d+[a-z]?(?:\s*[-.]\s*\d+[a-z]?)*"
    r"|[ivxlcdm]+"
    r"|[a-z]"
    r")\b"
)


def _coerce_analysis_to_text(analysis: Any) -> str:
    if isinstance(analysis, dict):
        parts = [
            str(analysis.get("header_text", "") or ""),
            str(analysis.get("text_excerpt", "") or ""),
            str(analysis.get("footer_text", "") or ""),
        ]
        return "\n".join(part for part in parts if part).strip()
    if isinstance(analysis, (str, Path)):
        try:
            raw_text = Path(analysis).read_text(encoding="utf-8").strip()
        except Exception:
            raw_text = str(analysis)
        try:
            data = json.loads(raw_text)
        except Exception:
            return raw_text
        return _coerce_analysis_to_text(data)
    return str(analysis or "")


def _has_explicit_section_marker(analysis: Any) -> bool:
    text = _coerce_analysis_to_text(analysis)
    return bool(SECTION_SYMBOL_RE.search(text) or SECTION_NUMBERED_RE.search(text))


def parse_ai_json(text: str) -> Dict:
    cleaned = (text or "").strip()
    if not cleaned:
        raise json.JSONDecodeError("Empty response", "", 0)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    code_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    candidate = code_match.group(1) if code_match else _extract_first_json_object(cleaned)
    if not candidate:
        raise json.JSONDecodeError("No JSON object found", cleaned[:200], 0)

    candidate = _sanitize_json_candidate(candidate.strip())
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Last-resort: tolerate Python dict-style output (single quotes, etc.).
    try:
        data = ast.literal_eval(candidate)
    except Exception as exc:
        raise json.JSONDecodeError("Failed to parse JSON candidate", candidate[:200], 0) from exc
    if not isinstance(data, dict):
        raise json.JSONDecodeError("Parsed data is not an object", candidate[:200], 0)
    return data


def _normalize_result(data: Dict) -> Dict:
    label = str(data.get("label", "")).strip()
    if label not in LABELS:
        label = ""
    raw_page = str(data.get("printed_page_number", "")).strip()
    decimal_match = re.fullmatch(r"(\d+)\.0+", raw_page)
    if decimal_match:
        cleaned_page = decimal_match.group(1)
    else:
        normalized = (
            raw_page.replace("–", "-")
            .replace("—", "-")
            .replace("−", "-")
        )
        cleaned_page = "".join(ch for ch in normalized if ch.isalnum() or ch == "-")
        cleaned_page = re.sub(r"-{2,}", "-", cleaned_page).strip("-")
    return {
        "label": label,
        "printed_page_number": cleaned_page,
    }


def _enforce_label_rules(result: Dict, analysis: Any) -> Dict:
    normalized = dict(result)
    if normalized.get("label") == "Section" and not _has_explicit_section_marker(analysis):
        normalized["label"] = ""
    return normalized


def _response_text(response) -> str:
    if hasattr(response, "output_text"):
        text = response.output_text or ""
        return text.strip()
    text_parts: List[str] = []
    output = getattr(response, "output", None) or []
    for item in output:
        if getattr(item, "type", None) == "message":
            for content in getattr(item, "content", None) or []:
                if getattr(content, "type", None) in {"output_text", "text"}:
                    text_parts.append(getattr(content, "text", ""))
    return "\n".join(text_parts).strip()


def _usage_to_dict(usage) -> Dict:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    try:
        return dict(usage)
    except Exception:
        return {}


def _get_cached_tokens(usage_dict: Dict) -> int:
    for key in ("input_tokens_details", "prompt_tokens_details"):
        details = usage_dict.get(key)
        if isinstance(details, dict) and "cached_tokens" in details:
            try:
                return int(details["cached_tokens"] or 0)
            except Exception:
                return 0
    try:
        return int(usage_dict.get("cached_tokens") or 0)
    except Exception:
        return 0


def _pricing_for_model(model_name: str) -> Dict[str, float]:
    base = DEFAULT_PRICING_PER_1M.get(model_name, {})
    return {
        "input": base.get("input"),
        "cached_input": base.get("cached_input"),
        "output": base.get("output"),
    }


def _estimate_cost_usd(usage_dict: Dict, model_name: str) -> Dict[str, float | int]:
    input_tokens = int(usage_dict.get("input_tokens") or 0)
    output_tokens = int(usage_dict.get("output_tokens") or 0)
    cached_tokens = _get_cached_tokens(usage_dict)
    if cached_tokens < 0:
        cached_tokens = 0
    if cached_tokens > input_tokens:
        cached_tokens = input_tokens
    uncached_input = input_tokens - cached_tokens

    rates = _pricing_for_model(model_name)
    input_rate = rates.get("input")
    cached_rate = rates.get("cached_input")
    output_rate = rates.get("output")
    if input_rate is None or cached_rate is None or output_rate is None:
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "estimated_cost_usd": 0.0,
            "pricing_available": 0,
        }
    estimated_cost = (
        (uncached_input * input_rate)
        + (cached_tokens * cached_rate)
        + (output_tokens * output_rate)
    ) / 1_000_000.0
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
        "estimated_cost_usd": estimated_cost,
        "pricing_available": 1,
        "rate_input_per_1m": input_rate,
        "rate_cached_input_per_1m": cached_rate,
        "rate_output_per_1m": output_rate,
    }


def _token_breakdown(usage_dict: Dict, pricing: Dict[str, float | int]) -> Dict[str, float | int]:
    prompt_tokens = int(usage_dict.get("input_tokens") or 0)
    completion_tokens = int(usage_dict.get("output_tokens") or 0)
    cached_tokens = _get_cached_tokens(usage_dict)
    reasoning_tokens = 0
    output_details = usage_dict.get("output_tokens_details")
    if isinstance(output_details, dict):
        try:
            reasoning_tokens = int(output_details.get("reasoning_tokens") or 0)
        except Exception:
            reasoning_tokens = 0
    visible_output_tokens = completion_tokens - reasoning_tokens
    if visible_output_tokens < 0:
        visible_output_tokens = 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "visible_output_tokens": visible_output_tokens,
        "cached_tokens": cached_tokens,
        "estimated_cost_usd": pricing.get("estimated_cost_usd", 0.0),
    }


def classify_page(analysis: Any, image_path: Path) -> Tuple[Dict, str, str]:
    _load_env()
    model_name = _get_model_name()
    client = OpenAI()

    cache = _load_file_cache()

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(analysis)
    system_content = [{"type": "input_text", "text": system_prompt}]

    page_image_id = _upload_file(client, image_path, cache, purpose="user_data")

    user_content = [
        {"type": "input_text", "text": user_prompt},
        {"type": "input_image", "file_id": page_image_id, "detail": "high"},
    ]

    input_items = [{"role": "system", "content": system_content}]
    input_items.append({"role": "user", "content": user_content})

    started_at = time.perf_counter()
    try:
        response = client.responses.create(
            model=model_name,
            input=input_items,
            prompt_cache_key=PROMPT_CACHE_KEY,
            prompt_cache_retention=PROMPT_CACHE_RETENTION,
        )
    except BadRequestError as exc:
        msg = str(exc)
        if "Timeout while downloading" in msg and "param': 'url'" in msg:
            page_image_id = _upload_file(client, image_path, cache, purpose="user_data", force=True)
            user_content = [
                {"type": "input_text", "text": user_prompt},
                {"type": "input_image", "file_id": page_image_id, "detail": "high"},
            ]
            input_items = [{"role": "system", "content": system_content}]
            input_items.append({"role": "user", "content": user_content})
            response = client.responses.create(
                model=model_name,
                input=input_items,
                prompt_cache_key=PROMPT_CACHE_KEY,
                prompt_cache_retention=PROMPT_CACHE_RETENTION,
            )
        else:
            raise

    raw_text = _response_text(response)
    data = parse_ai_json(raw_text)

    try:
        raw_response = json.dumps(response.model_dump(), ensure_ascii=False, indent=2)
    except Exception:
        raw_response = json.dumps(response, default=str, ensure_ascii=False, indent=2)

    usage = getattr(response, "usage", None)
    usage_dict = _usage_to_dict(usage)
    pricing = _estimate_cost_usd(usage_dict, model_name)
    elapsed_seconds = round(time.perf_counter() - started_at, 3)
    log_text = json.dumps(
        {
            "model": model_name,
            "elapsed_seconds": elapsed_seconds,
            "usage": usage_dict,
            "pricing": pricing,
            "token_breakdown": _token_breakdown(usage_dict, pricing),
        },
        ensure_ascii=False,
        indent=2,
    )

    result = _enforce_label_rules(_normalize_result(data), analysis)
    return result, log_text, raw_response
