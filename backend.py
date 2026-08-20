import datetime
import json
import os
import logging
import re
import time
import threading
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
IST_TZ = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

import asyncio
import requests
import io
import pypdf

from bs4 import BeautifulSoup
from fastapi import FastAPI, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv isn't installed - that's fine in production if the host
    # (Render, etc.) injects real environment variables directly instead of
    # a .env file. Add `python-dotenv` to requirements.txt for local dev.
    pass

fetched_at = datetime.datetime.now(IST_TZ).isoformat()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("notices-backend")

# ---------------------------------------------------------------------------
# AI summary + auto-refresh configuration.
#
# Everything here is overridable via environment variables so you can tune
# behaviour (or swap models) with no code change - e.g. once you move off
# the free tier. Google changes Gemini free-tier model names and rate
# limits without much notice, so GEMINI_MODEL in particular is worth
# checking against https://ai.google.dev/gemini-api/docs/models and
# https://ai.google.dev/gemini-api/docs/rate-limits if this ever starts
# 404-ing or getting rate-limited harder than expected.
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# How many un-summarized notices to send to Gemini in a single request, and
# how long to sleep between chunks. Defaults are deliberately conservative
# (~10 requests/minute) so a cold start (hundreds of notices with no
# summary yet) survives comfortably inside free-tier rate limits instead of
# tripping 429s. Raise these once you're on a paid tier.
SUMMARY_CHUNK_SIZE = int(os.environ.get("SUMMARY_CHUNK_SIZE", "12"))
SUMMARY_CHUNK_DELAY_SECONDS = float(os.environ.get("SUMMARY_CHUNK_DELAY_SECONDS", "6"))

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "").strip()
supabase_client = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")

# How often the backend re-scrapes every source and looks for notices that
# still need a summary. Summaries already on disk are never regenerated.
REFRESH_INTERVAL_SECONDS = int(os.environ.get("REFRESH_INTERVAL_SECONDS", str(60 * 60)))
ENABLE_SCHEDULER = os.environ.get("ENABLE_SCHEDULER", "true").strip().lower() != "false"

if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY is not set - AI summaries will be skipped until it is.")

try:
    from curl_cffi import requests as _curl_requests

    def new_session():
        return _curl_requests.Session(impersonate="chrome")

    logger.info("Using curl_cffi (Chrome TLS impersonation) for HTTP sessions.")
except ImportError:
    try:
        import cloudscraper

        def new_session():
            return cloudscraper.create_scraper(browser={"browser": "chrome", "platform": "windows", "mobile": False})

        logger.info("curl_cffi not installed - using cloudscraper for HTTP sessions.")
    except ImportError:
        import requests

        def new_session():
            return requests.Session()

        logger.warning("Neither curl_cffi nor cloudscraper installed - falling back to plain requests. "
                        "`pip install curl_cffi` for the best chance against MCX/BSE bot protection.")

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "application/json, text/plain, */*",
}

REQUEST_TIMEOUT = 45
INTER_REQUEST_DELAY = 0.4

app = FastAPI(title="Regulatory Notices Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# RAM cache — used as a fast in-process layer so page loads don't always
# need to round-trip Supabase. The persistent source of truth is now the
# `notices` Supabase table (append-only, deduplicated by id).

import threading

CACHE_LOCK = threading.Lock()
CACHE_REFRESHING = False
LAST_REFRESHED_AT: Optional[str] = None


def _normalize_datetime(value: Optional[str]) -> Optional[datetime.datetime]:
    """Parse ISO-like values and normalize them to IST."""
    if value is None or value == "":
        return None
    try:
        dt = datetime.datetime.fromisoformat(value)
    except ValueError:
        try:
            dt = datetime.datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST_TZ)
    return dt.astimezone(IST_TZ)


def _latest_refresh_timestamp(last_refresh_at: Optional[str], max_scraped: Optional[str]) -> str:
    """Use the newest valid refresh/scrape timestamp. Avoid stale in-memory values."""
    candidates: List[datetime.datetime] = []
    for value in (last_refresh_at, max_scraped):
        dt = _normalize_datetime(value)
        if dt is not None:
            candidates.append(dt)

    if candidates:
        return max(candidates).isoformat()
    return datetime.datetime.now(IST_TZ).isoformat()

# ---------------------------------------------------------------------------
# Supabase `notices` table helpers  (JSONB hybrid design)
#
# Table has 4 real SQL columns (fast filter/index) + 1 jsonb column
# that stores all remaining fields — only non-null values, so sparse
# rows (e.g. notices with no dividend fields) take minimal space.
#
# Schema (run once in your Supabase SQL editor):
#
#   create table if not exists notices (
#     id         text primary key,          -- e.g. "nse-NSE/CMTR/12345"
#     date       date         not null,     -- YYYY-MM-DD (or ex_date for dividends)
#     body       text,                      -- NSE | BSE | SEBI | MCX | MCXCCL | IFSCA
#     type       text default 'notice',     -- 'notice' | 'dividend'
#     data       jsonb        not null,     -- all other non-null fields
#     scraped_at timestamptz  default now()
#   );
#   create index if not exists notices_date_idx on notices (date desc);
#   create index if not exists notices_body_idx on notices (body);
# ---------------------------------------------------------------------------

# Top-level SQL columns (everything else goes in `data` jsonb)
_NOTICE_TOP_COLS = {"id", "date", "body", "type", "_source_key"}

def _notice_to_row(n: Dict) -> Dict:
    """Convert a notice dict into the 4-column + JSONB hybrid Supabase row.

    Row shape: { id, date, body, type, data: {...all other non-null fields...} }
    Only non-null / non-empty values go into `data`, so sparse notice rows
    (e.g. regular notices with no dividend fields) are as compact as possible.
    """
    ntype = n.get("type") or "notice"
    data: Dict = {}
    for k, v in n.items():
        if k in _NOTICE_TOP_COLS:  # already a real SQL column or internal tag
            continue
        if v is None or v == "":
            continue
        data[k] = v

    return {
        "id":   n.get("id", ""),
        "date": n.get("date") or n.get("ex_date") or "",
        "body": n.get("body") or "",
        "type": ntype,
        "data": data,
    }


def _row_to_notice(row: Dict) -> Dict:
    """Reconstruct a flat notice dict from a JSONB hybrid Supabase row.
    Merges the 4 top-level SQL columns back with the `data` jsonb payload.
    """
    notice = dict(row.get("data") or {})
    # Top-level SQL columns always win over any duplicate keys in data
    notice["id"]   = row.get("id",   notice.get("id",   ""))
    notice["date"] = row.get("date", notice.get("date", ""))
    notice["body"] = row.get("body", notice.get("body", ""))
    notice["type"] = row.get("type", notice.get("type", "notice"))
    notice["scraped_at"] = row.get("scraped_at")
    return notice

def _notices_upsert(notices: List[Dict]) -> int:
    """Upsert notices into the hybrid Supabase table.

    Preserve any existing stored fields (especially AI summaries) when a fresh
    scrape reuses the same notice id but doesn't include previously-generated
    metadata in its payload. This avoids wiping summaries on every refresh.
    """
    if not supabase_client or not notices:
        return len(notices)

    rows = [_notice_to_row(n) for n in notices if n.get("id")]
    if not rows: return 0

    upserted = 0
    for i in range(0, len(rows), 100):
        chunk = rows[i:i+100]
        try:
            ids = [r["id"] for r in chunk if r.get("id")]
            existing = {}
            if ids:
                resp = supabase_client.table("notices").select("id,data").in_("id", ids).execute()
                for row in (resp.data or []):
                    existing[row.get("id")] = row.get("data") or {}

            merged_chunk = []
            for row in chunk:
                rid = row.get("id")
                if rid in existing:
                    merged_data = dict(existing[rid])
                    merged_data.update(row.get("data") or {})
                    row["data"] = merged_data
                merged_chunk.append(row)

            supabase_client.table("notices").upsert(merged_chunk).execute()
            upserted += len(merged_chunk)
        except Exception as e:
            logger.error(f"Failed to upsert notices: {e}")

    return upserted

def _notices_fetch(from_date: datetime.date, to_date: datetime.date) -> Optional[Dict]:
    """Fetch notices from the hybrid Supabase table."""
    from_iso, to_iso = from_date.isoformat(), to_date.isoformat()

    # Always fetch from Supabase to ensure accurate date range data
    if not supabase_client: return None

    all_rows = []
    offset = 0
    try:
        while True:
            resp = supabase_client.table("notices").select("*").gte("date", from_iso).lte("date", to_iso).range(offset, offset + 999).execute()
            batch = [_row_to_notice(r) for r in (resp.data or [])]
            all_rows.extend(batch)
            if len(batch) < 1000: break
            offset += 1000
    except Exception as e:
        logger.error(f"Failed to fetch: {e}")
        return None

    return _wrap_notices_result(all_rows, from_date, to_date)


def _wrap_notices_result(notices: List[Dict], from_date: datetime.date,
                         to_date: datetime.date) -> Dict:
    """Package a flat list of notices into the standard result dict shape
    expected by get_all_notices() and _refresh_and_summarize()."""
    # Build source_status from what's in the list
    status: Dict[str, Dict] = {}
    for n in notices:
        src = n.get("body", "Unknown")
        if src not in status:
            status[src] = {"count": 0, "error": None, "note": None}
        status[src]["count"] += 1

    # Calculate the most recent time any of these notices were scraped/updated in the DB
    max_scraped = None
    for n in notices:
        sa = n.get("scraped_at")
        if sa:
            if not max_scraped or sa > max_scraped:
                max_scraped = sa
                
    # Prefer the newest valid refresh/scrape timestamp to avoid stale in-memory
    # values from a previous scheduler cycle appearing on the frontend.
    global LAST_REFRESHED_AT
    fetched_at = _latest_refresh_timestamp(LAST_REFRESHED_AT, max_scraped)

    return {
        "data": notices,
        "total": len(notices),
        "source_status": status,
        "ltp_failed_symbols": [],
        "from_date": from_date.isoformat(),
        "to_date": to_date.isoformat(),
        "version": "2026-07-28-v4-ai-summaries",
        "fetched_at": fetched_at,
    }

# Guards against two summarization passes running at once (one kicked off
# by the hourly scheduler, one by a page load) - same idea as
# CACHE_REFRESHING/CACHE_LOCK below for dataset refreshes.
_SUMMARY_RUNNING = False
_SUMMARY_RUN_LOCK = threading.Lock()

def _extract_text_from_url(url: str) -> str:
    """Download a document (PDF or HTML) and extract its text."""
    if not url:
        return ""
    try:
        s = new_session()
        s.headers.update(BROWSER_HEADERS)
        resp = s.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "").lower()
        if "pdf" in content_type or url.lower().endswith(".pdf"):
            reader = pypdf.PdfReader(io.BytesIO(resp.content))
            text = ""
            for i, page in enumerate(reader.pages):
                if i >= 5: # Limit to first 5 pages to save tokens
                    break
                text += page.extract_text() + "\n"
            return text.strip()
        elif "html" in content_type:
            soup = BeautifulSoup(resp.content, "lxml")
            return soup.get_text(separator=" ", strip=True)
        else:
            return ""
    except Exception as e:
        logger.warning(f"Failed to extract text from {url}: {e}")
        return ""


class GeminiRateLimited(Exception):
    """Raised when Gemini returns HTTP 429 - signals the caller to stop
    this summarization pass early rather than keep hammering the API."""
    pass


def _gemini_summarize_chunk(items: List[Dict]) -> Dict[str, str]:
    """Send one batch of notices/dividends to Gemini in a single request
    and return {id: summary}. Batching multiple items per call is what
    keeps this inside free-tier request-per-minute/day limits - see
    SUMMARY_CHUNK_SIZE."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")

    payload_items = []
    for it in items:
        if it.get("type") == "dividend":
            payload_items.append({
                "id": it.get("id"),
                "type": "dividend",
                "company": it.get("company", ""),
                "symbol": it.get("symbol", ""),
                "subject": it.get("subject", ""),
                "ex_date": it.get("ex_date", ""),
            })
        else:
            notice_payload = {
                "id": it.get("id"),
                "type": "notice",
                "body": it.get("body", ""),
                "category": it.get("category", ""),
                "title": _strip_html(it.get("title", "")),
                "date": it.get("date", ""),
            }
            if "document_text" in it:
                notice_payload["document_text"] = it["document_text"]
            payload_items.append(notice_payload)

    system_instruction = (
        "You are a concise financial regulatory analyst. You will receive a JSON array of "
        "Indian stock-market notices and dividend announcements, each with an 'id'. For every "
        "item, write a single 1-2 sentence summary of its core impact or directive; for "
        "dividends, mention the company, amount, and ex-date. Never use markdown formatting "
        "such as bold or bullet points. Respond with ONLY a JSON array, no other text and no "
        "markdown code fences, with exactly one object per input item in this exact shape: "
        '[{"id": "<same id as input>", "summary": "<your 1-2 sentence summary>"}, ...]. '
        "Reuse each input id unchanged so the caller can match summaries back to notices."
    )

    body = {
        "contents": [{"parts": [{"text": json.dumps(payload_items, ensure_ascii=False)}]}],
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "maxOutputTokens": 4096,
            # Low temperature: these are short factual summaries, not
            # creative writing - we want consistent, literal output rather
            # than varied phrasing across runs.
            "temperature": 0.2,
        },
    }

    max_retries = 6
    for attempt in range(max_retries):
        resp = requests.post(
            GEMINI_API_URL,
            headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
            json=body,
            timeout=60,
        )

        if resp.status_code == 429:
            if attempt < max_retries - 1:
                sleep_time = 2 ** attempt * 2  # 2s, 4s, 8s, 16s, 32s
                logger.warning(f"Gemini rate limited (429). Retrying in {sleep_time}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(sleep_time)
                continue
            else:
                raise GeminiRateLimited(resp.text[:300])

        if resp.status_code >= 400:
            raise RuntimeError(f"Gemini {resp.status_code}: {resp.text[:500]}")
        
        break

    data = resp.json()
    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {json.dumps(data)[:500]}")

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts).strip()

    # Defensive: strip stray markdown fences in case the model adds them
    # despite the JSON-mode instruction.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```$", "", text).strip()

    parsed = json.loads(text)

    out: Dict[str, str] = {}
    for row in parsed:
        rid = row.get("id")
        summ = _clean(row.get("summary"))
        if rid and summ:
            out[rid] = summ
    return out


def _run_summary_pass(notices: List[Dict]) -> Dict[str, int]:
    """Find notices with no stored summary yet and summarize them in small
    chunks, saving progress after every chunk so a rate-limit or crash
    partway through never loses already-generated summaries and never
    causes a previously-summarized notice to be re-sent to Gemini."""
    if not GEMINI_API_KEY:
        return {"generated": 0, "pending": len(notices)}

    pending = [n for n in notices if n.get("id") and not n.get("summary")]

    if not pending:
        logger.info("Summary pass: nothing new (%d notices already covered).", len(notices))
        return {"generated": 0, "pending": 0}

    logger.info("Summary pass: %d new notice(s) need AI summaries (model=%s).", len(pending), GEMINI_MODEL)

    generated = 0

    for i in range(0, len(pending), SUMMARY_CHUNK_SIZE):
        chunk = pending[i:i + SUMMARY_CHUNK_SIZE]
        
        # Deep read extraction
        deep_chunk = []
        for n in chunk:
            item = dict(n)
            link = item.get("link")
            if link:
                try:
                    doc_text = _extract_text_from_url(link)
                    if doc_text:
                        item["document_text"] = doc_text
                except Exception as e:
                    logger.warning("Failed to deep read %s: %s", link, e)
            deep_chunk.append(item)

        try:
            result = _gemini_summarize_chunk(deep_chunk)
        except GeminiRateLimited as e:
            logger.warning("Gemini rate-limited - stopping this pass early, will resume next cycle. %s", e)
            break
        except Exception:
            logger.exception("Gemini summarization failed for a chunk - skipping it this cycle.")
            time.sleep(SUMMARY_CHUNK_DELAY_SECONDS)
            continue

        if result:
            now_iso = datetime.datetime.now(IST_TZ).isoformat()
            updated_notices = []
            for item in deep_chunk:
                rid = item["id"]
                if rid in result:
                    item["summary"] = result[rid]
                    item["summary_generated_at"] = now_iso
                    item["summary_model"] = GEMINI_MODEL
                    updated_notices.append(item)
            
            generated += len(result)
            if updated_notices:
                _notices_upsert(updated_notices)

        time.sleep(SUMMARY_CHUNK_DELAY_SECONDS)

    still_pending = sum(1 for n in notices if n.get("id") and not n.get("summary"))
    logger.info("Summary pass complete: %d generated, %d still pending.", generated, still_pending)

    return {"generated": generated, "pending": still_pending}


def _run_summary_pass_guarded(notices: List[Dict]):
    """Thread entrypoint: runs _run_summary_pass then always clears the
    running flag, even on error."""
    global _SUMMARY_RUNNING
    try:
        _run_summary_pass(notices)
    except Exception:
        logger.exception("Summary pass crashed")
    finally:
        with _SUMMARY_RUN_LOCK:
            _SUMMARY_RUNNING = False


def _try_start_summary_pass() -> bool:
    global _SUMMARY_RUNNING
    with _SUMMARY_RUN_LOCK:
        if _SUMMARY_RUNNING:
            return False
        _SUMMARY_RUNNING = True
        return True


def _maybe_kickoff_summary_pass(notices: List[Dict]):
    """Fire-and-forget: start a background summarization pass if one isn't
    already running and a Gemini key is configured. Safe to call on every
    request - it's a cheap no-op once everything is caught up."""
    if not GEMINI_API_KEY:
        return
    if not _try_start_summary_pass():
        return
    threading.Thread(target=_run_summary_pass_guarded, args=(notices,), daemon=True).start()


def _clean(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()

def _strip_html(text: Optional[str]) -> str:
    """A few NSE press-release titles embed literal <br/> tags between
    multiple clarifications bundled into one row - turn those into a plain
    separator instead of feeding raw markup to Gemini (or the DOM)."""
    if not text:
        return ""
    text = re.sub(r"<br\s*/?>", "; ", str(text), flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return _clean(text)

def _parse_date_loose(text: Any) -> Optional[str]:
    """Try common exchange date formats -> 'YYYY-MM-DD'."""
    if text is None:
        return None
    text = _clean(text)
    if not text:
        return None
    fmts = ["%d-%b-%Y", "%d %b %Y", "%d-%m-%Y", "%d/%m/%Y", "%b %d, %Y",
            "%B %d, %Y", "%d %B %Y", "%Y%m%d", "%Y-%m-%dT%H:%M:%S"]
    for fmt in fmts:
        try:
            return datetime.datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    m = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    if m:
        return m.group(1)
    m = re.search(r"(\d{2})[-/](\d{2})[-/](\d{4})", text)
    if m:
        d, mo, y = m.groups()
        try:
            return datetime.date(int(y), int(mo), int(d)).isoformat()
        except ValueError:
            pass
    return None


def _get_first(row: Dict, *keys) -> Optional[Any]:
    """Case-insensitive first-match lookup across a list of candidate keys -
    handles the fact BSE/SEBI/NSE each use different casing conventions."""
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    lower_map = {str(k).lower(): v for k, v in row.items()}
    for k in keys:
        v = lower_map.get(k.lower())
        if v not in (None, ""):
            return v
    return None


def _unwrap_json_rows(payload: Any) -> List[Dict]:
    """Handle the handful of common JSON shapes these APIs return: a bare
    list, {"data": [...]}, {"Table": [...]} (classic ASP.NET convention),
    or a legacy {"d": "<json-encoded string>"} ASMX wrapper."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if "d" in payload:
            wrapped = payload["d"]
            if isinstance(wrapped, str):
                # classic ASMX convention: "d" is a JSON-encoded string
                try:
                    return _unwrap_json_rows(json.loads(wrapped))
                except Exception:
                    pass
            elif isinstance(wrapped, (list, dict)):
                # newer webmethod convention (e.g. MCX/MCXCCL): "d" is
                # already a nested object/array, no double-decoding needed
                return _unwrap_json_rows(wrapped)
        for key in ("Table", "table", "data", "Data", "results", "Result", "rows", "Rows", "notices"):
            if key in payload and isinstance(payload[key], list):
                return payload[key]
    return []


def _log_unknown_shape(source: str, rows: List[Dict]) -> str:
    if not rows:
        return ""
    sample_keys = list(rows[0].keys())
    logger.warning(f"{source}: could not find title/date in row - available keys: {sample_keys}")
    return f"Got {len(rows)} raw rows but couldn't map them to title/date fields. Raw keys: {sample_keys}"


# ===========================================================================
# NSE - Circulars & Press Releases (confirmed real JSON APIs)
# ===========================================================================
NSE_BASE = "https://www.nseindia.com"


_NSE_SESSION = None
_NSE_WARMUP_LOCK = threading.Lock()

def _nse_warmup_session():
    """NSE blocks direct API hits without first establishing browser-like
    cookies - hit the homepage before calling the API."""
    global _NSE_SESSION
    with _NSE_WARMUP_LOCK:
        if _NSE_SESSION is not None:
            return _NSE_SESSION
            
        s = new_session()
        s.headers.update(BROWSER_HEADERS)
        try:
            s.get(NSE_BASE, timeout=REQUEST_TIMEOUT)
            time.sleep(0.3)
        except Exception as e:
            logger.warning(f"NSE session warm-up failed (continuing anyway): {e}")
            
        _NSE_SESSION = s
        return s

def _clear_nse_session():
    global _NSE_SESSION
    with _NSE_WARMUP_LOCK:
        _NSE_SESSION = None


def _fetch_nse_json(path: str, from_date: datetime.date, to_date: datetime.date, referer: str) -> Dict:
    notices: List[Dict] = []
    error = None

    try:
        session = _nse_warmup_session()

        params = {
            "fromDate": from_date.strftime("%d-%m-%Y"),
            "toDate": to_date.strftime("%d-%m-%Y"),
        }

        resp = session.get(
            f"{NSE_BASE}{path}",
            params=params,
            headers={"Referer": referer},
            timeout=REQUEST_TIMEOUT,
        )
        
        if resp.status_code in (401, 403):
            _clear_nse_session()

        resp.raise_for_status()

        payload = resp.json()
        rows = _unwrap_json_rows(payload)

        seen = set()

        for row in rows:

            company = (row.get("circCompany") or "").strip().upper()
            circ_no = (row.get("circDisplayNo") or "").strip()

            # Keep ONLY NSE circulars
            if company != "NSE" and not circ_no.startswith("NSE/"):
                continue

            # Remove duplicates
            if circ_no in seen:
                continue
            seen.add(circ_no)

            title = _clean(row.get("sub"))
            if not title:
                continue

            parsed_date = (
                _parse_date_loose(row.get("cirDisplayDate"))
                or _parse_date_loose(row.get("cirDate"))
                or to_date.isoformat()
            )

            link = row.get("circFilelink", "")

            if link and not link.startswith("http"):
                link = f"https://nsearchives.nseindia.com{link}"

            notices.append({
                "id": f"nse-{circ_no}",
                "date": parsed_date,
                "title": title,
                "link": link,
                "department": row.get("circDepartment", ""),
                "category": row.get("circCategory", ""),
                "body": "NSE",
            })

    except Exception as e:
        logger.exception(f"NSE fetch failed for {path}")
        error = str(e)

    return {
        "data": notices,
        "error": error,
    }

def fetch_nse_circulars(from_date: datetime.date, to_date: datetime.date) -> Dict:
    return _fetch_nse_json(
        "/api/circulars",
        from_date,
        to_date,
        f"{NSE_BASE}/resources/exchange-communication-circulars",
    )


def fetch_nse_press_releases(from_date: datetime.date, to_date: datetime.date) -> Dict:
    """NSE press releases (cms20): each row's 'content' field is itself a
    JSON object (sometimes double-encoded as a string) shaped like:
        {"title": "NSE Indices", "body": "Inclusion in Nifty IPO w.e.f. ...",
         "field_a": "", "field_date": "16-Jul-2026",
         "field_file_attachement": {"url": "https://nsearchives..."}}
    Note 'title' here is actually a department/category label (e.g. "NSE
    Indices"), not the headline - the real headline is 'body'."""
    notices: List[Dict] = []
    error = None
    referer = f"{NSE_BASE}/resources/exchange-communication-press-releases"
    try:
        session = _nse_warmup_session()
        params = {"fromDate": from_date.strftime("%d-%m-%Y"), "toDate": to_date.strftime("%d-%m-%Y")}
        resp = session.get(f"{NSE_BASE}/api/press-release-cms20", params=params,
                            headers={"Referer": referer}, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            rows = _unwrap_json_rows(resp.json())
            for row in rows:
                content = row.get("content")
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except Exception:
                        content = None
                if not isinstance(content, dict):
                    continue

                title = _clean(_get_first(content, "body", "title"))
                if not title:
                    continue

                raw_date = _get_first(content, "field_date", "date")
                parsed_date = _parse_date_loose(raw_date)
                if not parsed_date:
                    changed_val = row.get("changed")
                    if isinstance(changed_val, (int, float)):
                        try:
                            parsed_date = datetime.datetime.utcfromtimestamp(changed_val).strftime("%Y-%m-%d")
                        except Exception:
                            pass
                    elif isinstance(changed_val, str) and changed_val.isdigit():
                        try:
                            parsed_date = datetime.datetime.utcfromtimestamp(int(changed_val)).strftime("%Y-%m-%d")
                        except Exception:
                            pass
                    elif isinstance(changed_val, str):
                        parsed_date = _parse_date_loose(changed_val)

                attachment = content.get("field_file_attachement") or content.get("field_file_attachment")
                link = None
                if isinstance(attachment, dict):
                    link = attachment.get("url")
                elif isinstance(attachment, str):
                    link = attachment
                if not link:
                    link = referer
                elif link.startswith("/"):
                    link = NSE_BASE + link

                notices.append({
                    "id": f"nse-pr-{row.get('id') or len(notices)}",
                    "body": "NSE", "category": "Press Release",
                    "date": parsed_date or to_date.isoformat(),
                    "title": title,
                    "link": link,
                })
            if rows and not notices:
                error = _log_unknown_shape("NSE Press Releases", rows)
        else:
            if resp.status_code in (401, 403):
                _clear_nse_session()
            error = f"NSE press releases API returned status {resp.status_code}"
    except Exception as e:
        logger.exception("NSE press release fetch failed")
        error = f"NSE press release fetch failed: {e}"
    if not notices and not error:
        error = "NSE press releases API returned 0 rows for this date range."
    return {"data": notices, "error": error}


# ===========================================================================
# BSE - Notices & Media Releases (confirmed real JSON APIs, api.bseindia.com)
# ===========================================================================
BSE_WWW = "https://www.bseindia.com"
BSE_API = "https://api.bseindia.com"


def _bse_session():
    s = new_session()
    s.headers.update(BROWSER_HEADERS)
    s.headers["Origin"] = BSE_WWW
    try:
        s.get(BSE_WWW, timeout=REQUEST_TIMEOUT)
    except Exception as e:
        logger.warning(f"BSE session warm-up failed (continuing anyway): {e}")
    return s


def fetch_bse_notices(from_date: datetime.date, to_date: datetime.date) -> Dict:
    notices: List[Dict] = []
    error = None
    url = f"{BSE_API}/BseIndiaAPI/api/getDataAdvance_New/w"
    params = {
        "strTxtNoticeNo": "", "strTxtDate": from_date.isoformat(), "strTxtTodate": to_date.isoformat(),
        "strScripcode": "", "strDep": "", "strSegment": "", "subject": "", "category": "", "containgtext": "",
    }
    try:
        session = _bse_session()
        resp = session.get(url, params=params,
                            headers={"Referer": f"{BSE_WWW}/markets/marketinfo/noticescirculars?id=0"},
                            timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            rows = _unwrap_json_rows(resp.json())
            for row in rows:
                title = _clean(_get_first(row, "Subject", "SUBJECT", "NOTICESUBJECT", "subject", "NOTICE_SUB", "Title"))
                if not title:
                    continue
                raw_date = _get_first(row, "Notice_Date", "dt_tm", "NOTICEDT", "NoticeDate", "NOTICE_DT", "date", "Date", "NEWS_DT")
                raw_link = _get_first(row, "FileName", "ATTACHMENTNAME", "AttachmentName", "PDFFLAG", "Attachment", "attachment")
                if raw_link and str(raw_link).startswith("http"):
                    link = raw_link
                elif raw_link:
                    link = f"{BSE_WWW}/xml-data/corpfiling/AttachLive/{raw_link}"
                else:
                    link = f"{BSE_WWW}/markets/marketinfo/noticescirculars?id=0"
                notice_no = _get_first(row, "Notice_id", "Notice_no", "NOTICENO", "NoticeNo", "NOTICE_NO", "Id")
                notices.append({
                    "id": f"bse-cir-{notice_no or len(notices)}",
                    "body": "BSE", "category": "Circular",
                    "date": _parse_date_loose(raw_date) or to_date.isoformat(),
                    "title": title, "link": link,
                })
            if rows and not notices:
                error = _log_unknown_shape("BSE Notices", rows)
        else:
            error = f"BSE notices API returned status {resp.status_code}"
    except Exception as e:
        logger.exception("BSE notices fetch failed")
        error = f"BSE notices fetch failed: {e}"
    if not notices and not error:
        error = "BSE notices API returned 0 rows for this date range."
    return {"data": notices, "error": error}


def fetch_bse_press_releases(from_date: datetime.date, to_date: datetime.date) -> Dict:
    """BSE's press-release API is year-granular (no day-level range), so we
    fetch every year the requested window touches and filter down client-side."""
    notices: List[Dict] = []
    error = None
    url = f"{BSE_API}/BseIndiaAPI/api/GetMediareleaseData/w"
    try:
        session = _bse_session()
        years = sorted({from_date.year, to_date.year})
        got_any_rows = False
        any_title_found = False  # true once we successfully map at least one
                                  # row - distinguishes "parsing worked, just
                                  # nothing in this date range" from a real
                                  # field-mapping failure
        for year in years:
            resp = session.get(url, params={"strCategory": "", "strYear": year},
                                headers={"Referer": f"{BSE_WWW}/markets/mediainfo/mediarelease"},
                                timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                logger.warning(f"BSE press release year {year} returned status {resp.status_code}")
                continue
            rows = _unwrap_json_rows(resp.json())
            if rows:
                got_any_rows = True
            for row in rows:
                if str(row.get("deleted", "")).strip().lower() in ("1", "true", "y", "yes"):
                    continue  # BSE marks retracted press releases with a deleted flag
                title = _clean(_get_first(row, "mr_heading", "HEADLINE", "Headline", "SUBJECT", "Title", "title"))
                if not title:
                    continue
                any_title_found = True
                raw_date = _get_first(row, "mr_date", "publish_date", "userapproved_date",
                                       "NEWS_DT", "NewsDate", "DATE", "Date", "PublishDate")
                parsed_date = _parse_date_loose(raw_date)
                if parsed_date and not (from_date.isoformat() <= parsed_date <= to_date.isoformat()):
                    continue  # outside requested window
                raw_link = _get_first(row, "mr_url", "PDFNAME", "PdfName", "ATTACHMENTNAME", "Attachment")
                if raw_link and str(raw_link).startswith("http"):
                    link = raw_link
                elif raw_link:
                    link = f"{BSE_WWW}/xml-data/corpfiling/AttachLive/{raw_link}"
                else:
                    link = f"{BSE_WWW}/markets/mediainfo/mediarelease"
                notices.append({
                    "id": f"bse-pr-{len(notices)}",
                    "body": "BSE", "category": "Press Release",
                    "date": parsed_date or to_date.isoformat(),
                    "title": title, "link": link,
                })
            time.sleep(INTER_REQUEST_DELAY)
        if years and not notices and got_any_rows and not any_title_found:
            error = _log_unknown_shape("BSE Press Releases", rows)
    except Exception as e:
        logger.exception("BSE press release fetch failed")
        error = f"BSE press release fetch failed: {e}"
    if not notices and not error:
     return {
        "data": [],
        "error": None,
        "note": f"No press releases published during the selected date range. (Debug: got_rows={got_any_rows}, title_found={any_title_found})"
    }

    return {
    "data": notices,
    "error": error,
    "note": None,
}

# ===========================================================================
# SEBI - "What's New" feed (HTML fragment, grouped by date then category)
# ===========================================================================
SEBI_BASE = "https://www.sebi.gov.in"


def fetch_sebi_whats_new(from_date: datetime.date, to_date: datetime.date) -> Dict:
    notices: List[Dict] = []
    error = None
    note = None

    ajax_url = f"{SEBI_BASE}/sebiweb/ajax/home/getnewslistallinfo.jsp"
    referer = f"{SEBI_BASE}/sebiweb/home/HomeAction.do?doListingAll=yes"

    from_iso = from_date.isoformat()
    to_iso = to_date.isoformat()

    MAX_SEBI_PAGES = 40

    def _parse_table(html_text: str) -> List[Dict]:
        soup = BeautifulSoup(html_text, "html.parser")

        table = soup.find("table", id="sample_1") or soup.find("table")
        if not table:
            return []

        tbody = table.find("tbody") or table

        rows = []

        for tr in tbody.find_all("tr"):

            tds = tr.find_all("td")

            if len(tds) < 3:
                continue

            date_text = _clean(tds[0].get_text(" ", strip=True))
            category = _clean(tds[1].get_text(" ", strip=True))

            a = tds[2].find("a", href=True)
            if not a:
                continue

            title = _clean(a.get_text(" ", strip=True))
            if not title:
                continue

            href = a["href"]

            if href.startswith("/"):
                href = SEBI_BASE + href

            rows.append({
                "date_text": date_text,
                "category": category or "Update",
                "title": title,
                "link": href,
            })

        return rows

    try:

        session = new_session()
        session.headers.update(BROWSER_HEADERS)

        session.get(referer, timeout=REQUEST_TIMEOUT)

        time.sleep(0.3)

        payload_base = {
            "search": "",
            "fromDate": from_date.strftime("%d-%m-%Y"),
            "toDate": to_date.strftime("%d-%m-%Y"),
            "deptId": "-1",
            "sid": "-1",
            "ssid": "-1",
            "smid": "-1",
            "cid": "-1",
            "sText": "-- All Section --",
            "ssText": "-- All Sub Section --",
            "smText": "-- All Sub Section List --",
            "cText": "-- All Info for --",
        }

        seen = set()

        next_value = "-1"

        total_pages = MAX_SEBI_PAGES

        pages_fetched = 0

        for page in range(MAX_SEBI_PAGES):

            payload = dict(payload_base)

            if page == 0:
                payload["next"] = "s"
                payload["nextValue"] = "-1"
                payload["doDirect"] = "1"
            else:
                payload["next"] = "n"
                payload["nextValue"] = next_value
                payload["doDirect"] = str(page)

            resp = session.post(
                ajax_url,
                data=payload,
                headers={
                    "Referer": referer,
                    "Origin": SEBI_BASE,
                    "X-Requested-With": "XMLHttpRequest",
                },
                timeout=REQUEST_TIMEOUT,
            )

            if resp.status_code != 200:
                error = f"SEBI returned HTTP {resp.status_code}"
                break

            html = resp.text

            m_total = re.search(
                r"name=['\"]totalpage['\"]\s+value=['\"]?(\d+)",
                html,
                re.I,
            )

            if m_total:
                total_pages = int(m_total.group(1))

            rows = _parse_table(html)

            if not rows:
                break

            pages_fetched += 1

            m_next = re.search(
                r"name=['\"]nextValue['\"]\s+value=['\"]?([^'\"> ]+)",
                html,
                re.I,
            )

            if m_next:
                next_value = m_next.group(1)

            stop = False

            for row in rows:

                parsed_date = _parse_date_loose(row["date_text"])

                if parsed_date:

                    if parsed_date < from_iso:
                        stop = True
                        continue

                    if parsed_date > to_iso:
                        continue

                key = (
                    row["title"],
                    parsed_date,
                    row["link"],
                )

                if key in seen:
                    continue

                seen.add(key)

                notices.append({
                    "id": f"sebi-{len(notices)}",
                    "body": "SEBI",
                    "category": row["category"],
                    "date": parsed_date or to_iso,
                    "title": row["title"],
                    "link": row["link"],
                })

            if stop:
                break

            # Last page reached
            if page + 1 >= total_pages:
                break

            time.sleep(INTER_REQUEST_DELAY)

        if not notices and not error:
            error = "No SEBI notices found."

        logger.info(
            "SEBI: %d notices fetched from %d page(s)",
            len(notices),
            pages_fetched,
        )

    except Exception as e:
        logger.exception("SEBI fetch failed")
        error = str(e)

    return {
        "data": notices,
        "error": error,
        "note": note,
    }


# ===========================================================================
# NSE Clearing (NCL) - own API paths (mirrors the NSE circulars shape), with
# a fallback that filters the already-fetched NSE circulars feed for
# clearing-department rows (NCL circulars are commonly cross-posted there
# under NCL/-prefixed circular numbers - this is why fetch_nse_circulars
# above now also captures each row's department).
# ===========================================================================
NCL_ARCHIVE = "https://www.archive.nseclearing.in/Others"


def fetch_nse_clearing(from_date: datetime.date, to_date: datetime.date) -> Dict:
    notices: List[Dict] = []
    error = None

    try:
        session = new_session()
        session.headers.update(BROWSER_HEADERS)

        params = {
            "from_date": from_date.strftime("%d-%m-%Y"),
            "to_date": to_date.strftime("%d-%m-%Y"),
        }

        resp = session.get(
            NCL_ARCHIVE,
            params=params,
            headers={
                **BROWSER_HEADERS,
                "Referer": "https://www.archive.nseclearing.in/",
                "Accept": "application/json, text/plain, */*",
            },
            timeout=REQUEST_TIMEOUT,
        )

        resp.raise_for_status()

        rows = resp.json()
        
        if isinstance(rows, dict):
            rows = (
                rows.get("data")
                or rows.get("result")
                or rows.get("rows")
                or []
            )

        if not isinstance(rows, list):
            rows = []

        seen = set()

        for row in rows:

            title = _clean(row.get("sub"))
            if not title:
                continue

            circ_no = row.get("circDisplayNo") or row.get("circNumber")

            parsed_date = (
                _parse_date_loose(row.get("cirDisplayDate"))
                or _parse_date_loose(row.get("cirDate"))
                or to_date.isoformat()
            )

            key = (circ_no, title, parsed_date)
            if key in seen:
                continue
            seen.add(key)

            notices.append({
                "id": f"ncl-{row.get('circNumber')}",
                "body": row.get("circCompany", "NCL"),
                "category": row.get("circCategory", "Circular"),
                "department": row.get("circDepartment", ""),
                "date": parsed_date,
                "title": title,
                "link": row.get("circFilelink"),
                "circular_no": row.get("circDisplayNo"),
                "file_size": row.get("file_Size"),
                "file_ext": row.get("fileExt"),
            })

        notices.sort(key=lambda x: x["date"], reverse=True)

    except Exception as e:
        logger.exception("NSE Clearing fetch failed")
        error = str(e)

    return {
        "data": notices,
        "error": error,
        "note": None,
    }

# ===========================================================================
# NSE CORPORATE ACTIONS (DIVIDEND)
# ===========================================================================

from urllib.parse import quote


def _extract_dividend_amount(subject: str) -> Optional[float]:
    """Parse the per-share dividend amount (in ₹) from the subject text.
    Common formats: 'Rs 125/-', 'Rs.5.50', 'Re 1/-', 'Rs 0.50 Per Share'."""
    if not subject:
        return None
    m = re.search(r'(?:Rs\.?|Re\.?)\s*([\d.]+)', subject, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _fetch_ltp_for_symbols(symbols: List[str]) -> Dict[str, Optional[float]]:
    """Fetch the Last Traded Price (LTP) from NSE for a list of symbols.
    Uses the NSE NextApi/GetQuoteApi endpoint which returns data nested
    under equityResponse[0]. We read buyPrice1 as the effective LTP.
    Returns {symbol: ltp_float_or_None}."""
    ltp_map: Dict[str, Optional[float]] = {}
    if not symbols:
        return ltp_map

    session = _nse_warmup_session()
    unique_symbols = list(set(symbols))
    logger.info("Fetching LTP for %d unique dividend symbols...", len(unique_symbols))

    for sym in unique_symbols:
        try:
            resp = session.get(
                f"{NSE_BASE}/api/NextApi/apiClient/GetQuoteApi",
                params={
                    "functionName": "getSymbolData",
                    "marketType": "N",
                    "series": "EQ",
                    "symbol": sym,
                },
                headers={"Referer": f"{NSE_BASE}/get-quotes/equity?symbol={sym}"},
                timeout=REQUEST_TIMEOUT,
            )

            if resp.status_code in (401, 403):
                _clear_nse_session()
                session = _nse_warmup_session()
                resp = session.get(
                    f"{NSE_BASE}/api/NextApi/apiClient/GetQuoteApi",
                    params={
                        "functionName": "getSymbolData",
                        "marketType": "N",
                        "series": "EQ",
                        "symbol": sym,
                    },
                    headers={"Referer": f"{NSE_BASE}/get-quotes/equity?symbol={sym}"},
                    timeout=REQUEST_TIMEOUT,
                )

            if resp.status_code == 200:
                data = resp.json()
                ltp_val = None

                # Response shape:
                # {"equityResponse": [{
                #   "orderBook": {"buyPrice1": 13977, "lastPrice": 13978, ...},
                #   "metaData": {"previousClose": 13928.12, "averagePrice": ...},
                #   "tradeInfo": {"lastPrice": 13978, ...},
                #   ...
                # }]}
                eq_resp = data.get("equityResponse")
                if isinstance(eq_resp, list) and len(eq_resp) > 0:
                    item = eq_resp[0]
                    order_book = item.get("orderBook") or {}
                    meta_data = item.get("metaData") or {}
                    trade_info = item.get("tradeInfo") or {}

                    # Prefer the freshest available quote, then fall back to the
                    # most stable reference price if the live quote is blank.
                    # This covers cases where NSE returns buyPrice1 or previousClose
                    # but not a true lastPrice field for some REIT/InvIT symbols.
                    ltp_val = (
                        order_book.get("lastPrice")
                        or order_book.get("buyPrice1")
                        or trade_info.get("lastPrice")
                        or meta_data.get("previousClose")
                        or meta_data.get("averagePrice")
                        or meta_data.get("closePrice")
                    )

                if ltp_val is not None:
                    ltp_map[sym] = float(ltp_val)
                else:
                    logger.warning("Could not extract LTP for %s from response keys: %s",
                                   sym, list(data.keys()) if isinstance(data, dict) else type(data))
            else:
                logger.warning("LTP fetch for %s returned status %s", sym, resp.status_code)

        except Exception as e:
            logger.warning("LTP fetch failed for %s: %s", sym, e)

        # Rate-limit courtesy
        time.sleep(0.15)

    logger.info("LTP fetched: %d/%d symbols resolved.", len(ltp_map), len(unique_symbols))
    return ltp_map


# ---------------------------------------------------------------------------
# LTP schedule-based fetching
#
# Instead of calling the NSE quote API on every refresh (slow, rate-limited),
# LTP is fetched only at specific times of day.  Between those times, the
# cached LTP values are served instantly without blocking.
# ---------------------------------------------------------------------------
_LTP_CACHE: Dict[str, float] = {}

# (hour, minute) in IST — fetch LTP around these times each day
LTP_FETCH_SCHEDULE = [
    (7, 50),   # Before 8 AM  — pre-market
    (8, 30),   # 8:30 AM      — pre-open session
    (9, 17),   # 9:17 AM      — just after market open (9:15)
    (11, 40),  # 11:40 AM     — mid-morning    
    (15, 40),  # 3:40 PM      — just after market close (3:30)
    (18, 0),   # 6:00 PM      — end of day
]

# Tracks which schedule slot was last served, as (date_str, slot_index)
_LTP_LAST_SERVED_SLOT: Optional[tuple] = None
_LTP_FETCH_LOCK = threading.Lock()


def _current_ltp_slot() -> Optional[tuple]:
    """Return (today_str, slot_index) for the most recent schedule slot
    that has already passed, or None if no slot has passed yet today."""
    now = datetime.datetime.now(IST_TZ)
    today_str = now.strftime("%Y-%m-%d")
    current_minutes = now.hour * 60 + now.minute

    best_slot = None
    for i, (h, m) in enumerate(LTP_FETCH_SCHEDULE):
        slot_minutes = h * 60 + m
        if current_minutes >= slot_minutes:
            best_slot = (today_str, i)

    return best_slot


def _should_fetch_ltp() -> bool:
    """Check if we've entered a new schedule slot that hasn't been served."""
    global _LTP_LAST_SERVED_SLOT
    current = _current_ltp_slot()
    if current is None:
        return False
    if _LTP_LAST_SERVED_SLOT is None:
        return True
    return current != _LTP_LAST_SERVED_SLOT


def _maybe_fetch_ltp_sync(symbols: List[str]):
    """Fetch LTPs synchronously if a new schedule slot is due.
    Otherwise, does nothing (relying on cache)."""
    global _LTP_LAST_SERVED_SLOT

    if not _should_fetch_ltp():
        return

    with _LTP_FETCH_LOCK:
        # Double check inside lock
        if not _should_fetch_ltp():
            return
            
        unique_syms = list(set(symbols))
        if not unique_syms:
            return

        current_slot = _current_ltp_slot()
        slot_label = f"slot {current_slot}" if current_slot else "unknown"
        logger.info("Scheduled LTP fetch starting (%s) for %d symbols...", slot_label, len(unique_syms))

        ltp_map = _fetch_ltp_for_symbols(unique_syms)

        # Update cache with fresh values
        for sym, price in ltp_map.items():
            _LTP_CACHE[sym] = price

        # Mark this slot as served
        if current_slot:
            _LTP_LAST_SERVED_SLOT = current_slot

        logger.info("Scheduled LTP fetch done: %d/%d fresh.", len(ltp_map), len(unique_syms))


def fetch_nse_corporate_actions(from_date: datetime.date,
                                to_date: datetime.date) -> Dict:

    notices = []
    error = None
    ltp_failed_symbols: List[str] = []

    try:
        session = _nse_warmup_session()

        response = session.get(
            "https://www.nseindia.com/api/corporates-corporateActions",
            params={
                "index": "equities",
                "from_date": from_date.strftime("%d-%m-%Y"),
                "to_date": to_date.strftime("%d-%m-%Y")
            },
            headers={
                "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-actions"
            },
            timeout=REQUEST_TIMEOUT,
        )
        
        if response.status_code in (401, 403):
            _clear_nse_session()

        response.raise_for_status()

        rows = response.json()

        logger.info("Dividend API response type: %s", type(rows))

        if isinstance(rows, dict):
            logger.info("Dividend API keys: %s", rows.keys())
            rows = rows.get("data", [])

        logger.info("Dividend rows received: %d", len(rows))

        for row in rows:

            ex_date = _parse_date_loose(row.get("exDate"))

            if not ex_date:
                continue

            d = datetime.date.fromisoformat(ex_date)

            if d < from_date or d > to_date:
                continue

            symbol = (row.get("symbol") or "").strip()
            company = (row.get("comp") or "").strip()

            # Create NSE quote URL
            company_slug = (
                company
                .replace("(", "")
                .replace(")", "")
                .replace(",", "")
                .replace(".", "")
                .replace("/", "-")
                .replace(" ", "-")
            )

            link = (
                f"https://www.nseindia.com/get-quote/equity/"
                f"{quote(symbol)}/{quote(company_slug, safe='-')}"
            )

            notices.append({
                "id": f"dividend-{symbol}-{ex_date}",

                "type": "dividend",

                "body": "NSE",

                "category": "Dividend",

                "date": ex_date,

                "title": company,

                "symbol": symbol,
                "company": company,
                "subject": row.get("subject", ""),
                "ex_date": ex_date,
                "record_date": _parse_date_loose(row.get("recDate")),
                "face_value": row.get("faceVal", ""),

                # Dynamic NSE Quote Link
                "link": link,
            })

        # --- Enrich dividends with LTP (non-blocking outside schedule, sync during schedule) ---
        if notices:
            symbols = [n["symbol"] for n in notices if n.get("symbol")]

            # Fetch synchronously if a new schedule slot is due, otherwise skip
            _maybe_fetch_ltp_sync(symbols)

            # Always enrich from cache instantly (never blocks)
            for n in notices:
                sym = n.get("symbol", "")
                ltp = _LTP_CACHE.get(sym)
                div_amount = _extract_dividend_amount(n.get("subject", ""))

                n["ltp"] = ltp
                n["ltp_cached"] = True  # always from cache in this path
                n["dividend_amount"] = div_amount

                if ltp and div_amount and ltp > 0:
                    n["dividend_pct"] = round((div_amount / ltp) * 100, 2)
                else:
                    n["dividend_pct"] = None

                if ltp is None:
                    ltp_failed_symbols.append(sym)

    except Exception as e:
        logger.exception("Dividend fetch failed")
        error = str(e)

    return {
        "data": notices,
        "error": error,
        "ltp_failed_symbols": ltp_failed_symbols,
    }
# ===========================================================================
# MCX / MCXCCL - ASP.NET webmethod (backpage.aspx/GetCircularSearch), with a
# fallback that scrapes circular PDF links straight off the listing page.
# Both exchanges share the exact same webmethod shape, so one helper drives
# both fetch_mcx() and fetch_mcxccl(). See the module docstring for why MCX
# is back after being removed once before.
# ===========================================================================
def _fetch_new_mcx_api(url, source, from_date, to_date):

    notices = []

    try:
        session = new_session()
        session.headers.update(BROWSER_HEADERS)

        page = 1

        while True:

            params = {
                "CircularTitle": "",
                "CircularsCategory": "",
                "CircularNo": "",
                "fromdate": from_date.strftime("%d/%m/%Y"),
                "todate": to_date.strftime("%d/%m/%Y"),
                "page": page
            }

            r = session.get(url, params=params, timeout=REQUEST_TIMEOUT)

            if r.status_code != 200:
                return {
                    "data": [],
                    "error": f"{source} returned {r.status_code}"
                }

            data = r.json()

            rows = data.get("Announcements", [])

            if not rows:
                break

            for row in rows:

                notices.append({
                    "id": f"{source}-{row['CircularNo']}",
                    "body": source,
                    "category": row.get("CircularsCategory", "Circular"),
                    "date": datetime.datetime.strptime(
                        row["DisplayDate"],
                        "%d %b %Y"
                    ).strftime("%Y-%m-%d"),
                    "title": row["Title"],
                    "link": row["CircularFile"]
                })

            if page >= data.get("TotalPages", 1):
                break

            page += 1
            time.sleep(0.2)

        return {
            "data": notices,
            "error": None
        }

    except Exception as e:
        logger.exception(e)

        return {
            "data": [],
            "error": str(e)
        }

def fetch_mcx(from_date: datetime.date, to_date: datetime.date) -> Dict:
    return _fetch_new_mcx_api(
        "https://www.mcxindia.com/circulars/all-circulars/GetFilteredAnnouncements",
        "MCX",
        from_date,
        to_date
    )


def fetch_mcxccl(from_date: datetime.date, to_date: datetime.date) -> Dict:
    return _fetch_new_mcx_api(
        "https://www.mcxccl.com/all-circulars/GetFilteredAnnouncements",
        "MCXCCL",
        from_date,
        to_date
    )

# ===========================================================================
# IFSCA - Legal/circulars index page (HTML, no machine-readable per-item
# dates, so results aren't date-filtered - see the note surfaced below)
# ===========================================================================
from urllib.parse import urljoin

IFSCA_URL = "https://www.ifsca.gov.in/Home/NewSection"

def fetch_ifsca(from_date: datetime.date, to_date: datetime.date) -> Dict:
    notices = []
    error = None

    try:
        session = new_session()
        session.headers.update(BROWSER_HEADERS)

        r = session.get(IFSCA_URL, timeout=90)
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "lxml")

        table = soup.find("table", id="tblNewSec")
        if not table:
            return {
                "data": [],
                "error": "IFSCA table not found."
            }

        rows = table.find("tbody").find_all("tr")

        for tr in rows:
            cols = tr.find_all("td")
            if len(cols) < 3:
                continue

            raw_date = _clean(cols[0].get_text(" ", strip=True))
            category = _clean(cols[1].get_text(" ", strip=True))

            a = cols[2].find("a")
            if not a:
                continue

            title = _clean(a.get_text(" ", strip=True))
            link = urljoin(IFSCA_URL, a.get("href", ""))

            date = _parse_date_loose(raw_date)
            if not date:
                continue

            d = datetime.date.fromisoformat(date)
            if d < from_date or d > to_date:
                continue

            notices.append({
                "id": f"ifsca-{len(notices)+1}",
                "body": "IFSCA",
                "category": category,
                "date": date,
                "title": title,
                "link": link,
            })

    except Exception as e:
        logger.exception("IFSCA fetch failed")
        error = str(e)

    return {
        "data": notices,
        "error": error,
    }

from bs4 import BeautifulSoup


# ===========================================================================
# Aggregation endpoint
# ===========================================================================
def _scrape_fresh(from_date: datetime.date, to_date: datetime.date) -> Dict:
    """Run all scrapers in parallel for the given date window and return a
    raw result dict. Does NOT touch cache or Supabase — pure scrape."""
    jobs = {
        "NSE Circulars": fetch_nse_circulars,
        "NSE Press Releases": fetch_nse_press_releases,
        "NSE Dividend": fetch_nse_corporate_actions,
        "NSE Clearing": fetch_nse_clearing,
        "BSE Notices": fetch_bse_notices,
        "BSE Press Releases": fetch_bse_press_releases,
        "SEBI Updates": fetch_sebi_whats_new,
        "MCX Circulars": fetch_mcx,
        "MCXCCL Circulars": fetch_mcxccl,
        "IFSCA": fetch_ifsca,
    }

    sources: Dict[str, Dict] = {}
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        future_map = {
            executor.submit(func, from_date, to_date): name
            for name, func in jobs.items()
        }
        for future in as_completed(future_map):
            name = future_map[future]
            try:
                sources[name] = future.result()
                logger.info(f"✓ {name} completed")
            except Exception as e:
                logger.exception(f"{name} failed")
                sources[name] = {"data": [], "error": str(e), "note": "Fetch failed"}

    all_notices: List[Dict] = []
    status: Dict[str, Dict] = {}
    from_iso = from_date.isoformat()
    to_iso = to_date.isoformat()

    for name, src in sources.items():
        filtered = [r for r in src["data"] if from_iso <= r.get("date", to_iso) <= to_iso]
        for row in filtered:
            row["_source_key"] = name
        all_notices.extend(filtered)
        status[name] = {
            "count": len(filtered),
            "error": src.get("error"),
            "note": src.get("note"),
        }

    all_notices.sort(key=lambda x: x.get("date", ""), reverse=True)
    ltp_failed = sources.get("NSE Dividend", {}).get("ltp_failed_symbols", [])

    result = _wrap_notices_result(all_notices, from_date, to_date)
    result.update({
        "total": len(all_notices),
        "source_status": status,
        "ltp_failed_symbols": ltp_failed,
    })
    return result


def _build_dataset(from_date: datetime.date, to_date: datetime.date,
                   force_refresh: bool = False) -> Dict:
    """Return notices for the requested date range.

    - force_refresh=False (normal page load): read from Supabase `notices`
      table. Never re-scrapes live sources.
    - force_refresh=True (called only by the scheduler): scrape live, upsert
      to Supabase, then return the freshly-scraped data.

    The Supabase `notices` table is the persistent source of truth and grows
    day-by-day.
    """
    global CACHE_REFRESHING, LAST_REFRESHED_AT

    # -------------------------------------------------------
    # Normal page load: serve from Supabase / RAM cache
    # -------------------------------------------------------
    if not force_refresh:
        result = _notices_fetch(from_date, to_date)
        if result is not None:
            return result
        # Nothing in Supabase/cache yet → fall through to a live scrape
        logger.info("No cached notices found; falling back to live scrape.")

    # -------------------------------------------------------
    # Guard against concurrent refreshes
    # -------------------------------------------------------
    if CACHE_REFRESHING:
        logger.info("Refresh already running. Returning cached notices.")
        result = _notices_fetch(from_date, to_date)
        if result is not None:
            return result

    with CACHE_LOCK:
        CACHE_REFRESHING = True
        try:
            logger.info("Building live dataset for %s → %s …",
                        from_date.isoformat(), to_date.isoformat())
            result = _scrape_fresh(from_date, to_date)
            
            LAST_REFRESHED_AT = datetime.datetime.now(IST_TZ).isoformat()

            # Upsert scraped rows into Supabase (append + dedup by id)
            _notices_upsert(result["data"])

            logger.info("Live scrape complete: %d notices.", len(result["data"]))
            return result

        except Exception:
            logger.exception("Dataset refresh failed.")
            # Last resort: return whatever is in Supabase/RAM
            fallback = _notices_fetch(from_date, to_date)
            if fallback is not None:
                logger.info("Returning cached notices as fallback.")
                return fallback
            raise
        finally:
            CACHE_REFRESHING = False
@app.get("/api/notices")
def get_all_notices(
    from_date: Optional[str] = Query(None, description="YYYY-MM-DD, defaults to 7 days ago"),
    to_date: Optional[str] = Query(None, description="YYYY-MM-DD, defaults to today"),
):
    """Returns notices/circulars/press releases from NSE, NSE Clearing, BSE,
    SEBI, MCX, MCXCCL, IFSCA for the given date window.

    Data is read from the persistent Supabase `notices` table which is
    appended to on each scheduled refresh (8am, 9am, 10am, 3:30pm, 5pm,
    11pm IST). Each notice is stored exactly once (deduplicated by id).

    Every row also carries a `summary` field pre-generated by Gemini.
    Rows with no summary yet show `summary: null`."""
    today = datetime.datetime.now(IST_TZ).date()
    to_d = datetime.datetime.fromisoformat(to_date).date() if to_date else today
    from_d = datetime.date.fromisoformat(from_date) if from_date else (to_d - datetime.timedelta(days=7))

    # Read from Supabase `notices` table (or RAM cache) — never force-scrapes
    result = _build_dataset(from_d, to_d, force_refresh=False)
    all_notices = result.get("data", [])

    # Date-filter (Supabase already filters, but RAM cache may have extras)
    from_iso = from_d.isoformat()
    to_iso = to_d.isoformat()
    raw_notices = [
        n for n in all_notices
        if from_iso <= (n.get("date") or "") <= to_iso
        and not (
            n.get("type") == "dividend"
            and from_d != to_d
        )
    ]

    # Rebuild source_status counts for the filtered window
    new_source_status: Dict[str, Dict] = {}
    for n in raw_notices:
        src = n.get("body") or n.get("_source_key") or "Unknown"
        if src not in new_source_status:
            new_source_status[src] = {"count": 0, "error": None, "note": None}
        new_source_status[src]["count"] += 1

    decorated = raw_notices
    have = sum(1 for d in decorated if d.get("summary"))

    # Fire-and-forget: summarize any notices that don't have a summary yet
    _maybe_kickoff_summary_pass(raw_notices)

    # Remove internal tags before sending to frontend
    for d in decorated:
        d.pop("_source_key", None)

    return {
        "data": decorated,
        "total": len(decorated),
        "source_status": new_source_status,
        "ltp_failed_symbols": result.get("ltp_failed_symbols", []),
        "from_date": from_iso,
        "to_date": to_iso,
        "fetched_at": result.get("fetched_at") or datetime.datetime.now(IST_TZ).isoformat(),
        "version": result.get("version", ""),
        "summary_status": {
            "generated": have,
            "pending": len(decorated) - have,
            "model": GEMINI_MODEL if GEMINI_API_KEY else None,
        },
    }

@app.post("/api/admin/force-refresh")
async def admin_force_refresh():
    """Admin endpoint to manually trigger today's scrape, upsert to Supabase,
    and run the summary pass. Returns any source errors."""
    try:
        status = await _refresh_and_summarize()
        errors = {name: s["error"] for name, s in status.items() if s.get("error")}
        notes = {name: s["note"] for name, s in status.items() if s.get("note") and not s.get("error")}

        return {
            "status": "ok",
            "message": "Manual refresh (today only) completed.",
            "errors": errors if errors else None,
            "notes": notes if notes else None,
            "full_status": status
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/admin/backfill")
async def admin_backfill(
    days: int = Query(30, description="Number of past days to scrape and store (max 90)"),
):
    """One-shot backfill: scrape the last N days (default 30, max 90) and
    upsert everything into the Supabase `notices` table. Safe to run
    multiple times - existing rows are never duplicated (upsert by id).
    
    Use this once after migrating to the new append-only architecture to
    seed historical data into the `notices` table."""
    days = min(max(days, 1), 90)  # clamp to 1-90
    loop = asyncio.get_event_loop()
    today = datetime.datetime.now(IST_TZ).date()
    from_d = today - datetime.timedelta(days=days)

    logger.info("Backfill requested: %d days (%s → %s)", days, from_d.isoformat(), today.isoformat())

    try:
        result = await loop.run_in_executor(None, _build_dataset, from_d, today, True)
        count = len(result.get("data", []))
        errors = {name: s["error"] for name, s in result.get("source_status", {}).items() if s.get("error")}

        return {
            "status": "ok",
            "message": f"Backfill complete: {count} notices upserted into Supabase for the last {days} days.",
            "notices_upserted": count,
            "from_date": from_d.isoformat(),
            "to_date": today.isoformat(),
            "errors": errors if errors else None,
        }
    except Exception as e:
        logger.exception("Backfill failed")
        return {"status": "error", "message": str(e)}


@app.get("/api/summaries/status")
def summaries_status():
    """Lightweight progress check - how many notices have an AI summary."""
    # Since we are stateless, we can do a quick count from the DB, or just 
    # return a static placeholder as the UI doesn't strictly depend on this for core functionality.
    return {
        "status": "stateless-mode",
        "model": GEMINI_MODEL,
        "gemini_key_configured": bool(GEMINI_API_KEY),
        "pass_running": _SUMMARY_RUNNING,
    }





@app.post("/api/summaries/notice/{notice_id:path}")
def summarize_one(notice_id: str, force: bool = False):
    """On-demand summary for a single notice - used by the \"Generate now\"
    fallback in the UI so a visitor doesn't have to wait for the hourly
    pass. Idempotent: if this id is already summarized, returns the
    existing summary instead of spending another Gemini call on it."""
    # Fetch from Supabase directly
    notice = None
    if supabase_client:
        try:
            resp = supabase_client.table("notices").select("*").eq("id", notice_id).limit(1).execute()
            if resp.data:
                notice = _row_to_notice(resp.data[0])
        except Exception as e:
            logger.warning("Could not fetch notice %s from Supabase: %s", notice_id, e)

    if not notice:
        return {"status": "not-found"}

    if not force:
        if notice.get("summary"):
            return {"status": "ok", "summary": notice["summary"], "cached": True}

    if not GEMINI_API_KEY:
        return {"status": "no-key", "message": "GEMINI_API_KEY is not set on the backend."}

    notice_to_summarize = dict(notice)
    link = notice_to_summarize.get("link")
    if link:
        doc_text = _extract_text_from_url(link)
        if doc_text:
            notice_to_summarize["document_text"] = doc_text

    try:
        result = _gemini_summarize_chunk([notice_to_summarize])
    except GeminiRateLimited as e:
        return {"status": "rate-limited", "message": str(e)}
    except Exception as e:
        logger.exception("On-demand summarize failed for %s", notice_id)
        return {"status": "error", "message": str(e)}

    summary = result.get(notice_id)
    if not summary:
        return {"status": "error", "message": "Model did not return a summary for this id."}

    notice_to_summarize["summary"] = summary
    notice_to_summarize["summary_model"] = GEMINI_MODEL
    notice_to_summarize["summary_generated_at"] = datetime.datetime.now(IST_TZ).isoformat()
    
    _notices_upsert([notice_to_summarize])

    return {"status": "ok", "summary": summary, "cached": False}


# ===========================================================================
# Hourly background job: re-fetch every source, then summarize whatever's
# new. Runs as an asyncio task on the same event loop FastAPI already uses;
# the actual scraping/Gemini work happens in a thread executor so it never
# blocks request handling. Single-process assumption: if this ever runs
# behind multiple Uvicorn workers, move this loop to its own process/cron
# job instead, or every worker will refresh + summarize independently.
# ===========================================================================
async def _refresh_and_summarize():
    """Scheduled cycle: scrape TODAY's data only, upsert into the Supabase
    `notices` table (append-only, deduplicated by id), then summarize any
    notices that don't have a summary yet.

    We no longer re-scrape 30 days every run. Each day's data is stored once
    and never overwritten (unless the notice itself is updated upstream).
    """
    loop = asyncio.get_event_loop()
    today = datetime.datetime.now(IST_TZ).date()
    # Scrape today only — yesterday is already in Supabase from prior runs.
    # We include yesterday as a safety buffer (some sources post late).
    from_d = today - datetime.timedelta(days=1)

    logger.info("Scheduled cycle: scraping today's data (%s → %s)…",
                from_d.isoformat(), today.isoformat())

    # force_refresh=True → always scrapes live, then upserts to Supabase
    result = await loop.run_in_executor(None, _build_dataset, from_d, today, True)

    status = result.get("source_status", {})
    scraped_count = len(result.get("data", []))
    logger.info("Scheduled cycle: %d notices scraped and upserted.", scraped_count)

    # NOTE: We do NOT purge old summaries on a today-only scrape.
    # Summaries live persistently inside the `data` JSON column of the `notices` table.

    if not _try_start_summary_pass():
        logger.info("Scheduled cycle: a summary pass is already running - skipping.")
        return status

    logger.info("Scheduled cycle: summarizing any new notices...")
    await loop.run_in_executor(None, _run_summary_pass_guarded, result.get("data", []))

    return status


async def _scheduled_job_loop():
    await asyncio.sleep(5)  # let the app finish starting first
    
    # Target hours and minutes in IST (8am, 9am, 10am, 3:30pm, 5pm, 11pm)
    target_times = [(8, 0), (9, 0), (10, 0), (15, 30), (17, 0), (23, 0)]
    last_run_time = None
    
    while True:
        try:
            now = datetime.datetime.now(IST_TZ)
            current_time = (now.hour, now.minute)
            
            # If current time matches one of the target times, and we haven't already run in this minute
            if current_time in target_times and current_time != last_run_time:
                logger.info(f"Scheduled time reached: {now.strftime('%H:%M')} IST. Starting refresh.")
                last_run_time = current_time
                await _refresh_and_summarize()
                
        except Exception:
            logger.exception("Scheduled refresh/summarize cycle failed")
            
        # Check every 30 seconds so we don't miss the minute mark
        await asyncio.sleep(30)


@app.on_event("startup")
async def _on_startup():
    if ENABLE_SCHEDULER:
        logger.info(
            "Starting scheduled refresh loop (8am, 9am, 10am, 3:30pm, 5pm, 11pm IST)."
        )
        asyncio.create_task(_scheduled_job_loop())
    else:
        logger.info("ENABLE_SCHEDULER=false - scheduled auto-refresh is disabled.")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "time": datetime.datetime.now(IST_TZ).isoformat(),
        "version": "2026-07-28-v4-ai-summaries",
        "scheduler_enabled": ENABLE_SCHEDULER,
        "refresh_interval_seconds": REFRESH_INTERVAL_SECONDS,
        "gemini_model": GEMINI_MODEL,
        "gemini_key_configured": bool(GEMINI_API_KEY),
    }
from fastapi.responses import FileResponse, HTMLResponse
import os

@app.get("/", response_class=FileResponse)
def home():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>✅ Regulatory Notices Dashboard API</h1><p>Backend is running, but index.html was not found.</p>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)