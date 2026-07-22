import datetime
import html
import json
import logging
import re
import time
from typing import List, Dict, Optional, Any

from bs4 import BeautifulSoup
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import datetime
from zoneinfo import ZoneInfo

fetched_at = datetime.datetime.now(ZoneInfo("Asia/Kolkata")).isoformat()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("notices-backend")

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

REQUEST_TIMEOUT = 15
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
# Cache - keyed by (from_date, to_date) so switching the date window doesn't
# require re-scraping if you flip back to a range you already fetched.
# ---------------------------------------------------------------------------
_CACHE: Dict[str, Dict] = {}
CACHE_TTL_SECONDS = 15 * 60


def _cache_get(key: str):
    entry = _CACHE.get(key)
    if entry and (time.time() - entry["ts"]) < CACHE_TTL_SECONDS:
        return entry["data"]
    return None


def _cache_set(key: str, data):
    _CACHE[key] = {"data": data, "ts": time.time()}


def _clean(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


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


def _nse_warmup_session():
    """NSE blocks direct API hits without first establishing browser-like
    cookies - hit the homepage before calling the API."""
    s = new_session()
    s.headers.update(BROWSER_HEADERS)
    try:
        s.get(NSE_BASE, timeout=REQUEST_TIMEOUT)
        time.sleep(0.3)
    except Exception as e:
        logger.warning(f"NSE session warm-up failed (continuing anyway): {e}")
    return s


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
        error = "BSE press release API returned 0 rows for this date range."
    return {"data": notices, "error": error}


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

        r = session.get(IFSCA_URL, timeout=REQUEST_TIMEOUT)
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
import datetime as dt



# ===========================================================================
# Aggregation endpoint
# ===========================================================================
def _build_dataset(from_date: datetime.date, to_date: datetime.date, force_refresh: bool = False) -> Dict:
    cache_key = f"dataset:{from_date.isoformat()}:{to_date.isoformat()}"
    if not force_refresh:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

    sources = {
        "NSE Circulars": fetch_nse_circulars(from_date, to_date),
        "NSE Press Releases": fetch_nse_press_releases(from_date, to_date),
        "NSE Clearing": fetch_nse_clearing(from_date, to_date),
        "BSE Notices": fetch_bse_notices(from_date, to_date),
        "BSE Press Releases": fetch_bse_press_releases(from_date, to_date),
        "SEBI Updates": fetch_sebi_whats_new(from_date, to_date),
        "MCX Circulars": fetch_mcx(from_date, to_date),
        "MCXCCL Circulars": fetch_mcxccl(from_date, to_date),
        "IFSCA": fetch_ifsca(from_date, to_date),
    }

    all_notices: List[Dict] = []
    status: Dict[str, Dict] = {}
    for name, result in sources.items():
        all_notices.extend(result["data"])
        status[name] = {"count": len(result["data"]), "error": result["error"], "note": result.get("note")}

    all_notices.sort(key=lambda x: x["date"], reverse=True)

    result = {
        "data": all_notices,
        "total": len(all_notices),
        "source_status": status,
        "from_date": from_date.isoformat(),
        "to_date": to_date.isoformat(),
        "version": "2026-07-22-v2",
        "fetched_at": datetime.datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
    }
    _cache_set(cache_key, result)
    return result


@app.get("/api/notices")
def get_all_notices(
    refresh: bool = Query(False, description="Force a fresh fetch, bypassing the cache"),
    from_date: Optional[str] = Query(None, description="YYYY-MM-DD, defaults to 30 days ago"),
    to_date: Optional[str] = Query(None, description="YYYY-MM-DD, defaults to today"),
):
    """Returns notices/circulars/press releases from NSE, NSE Clearing, BSE,
    SEBI, MCX, MCXCCL, IFSCA for the given date window (default:
    last 30 days). No mock data - each source's row count and any error is
    under `source_status`."""
    today = datetime.datetime.now(
    ZoneInfo("Asia/Kolkata")
).date()
    to_d = datetime.datetime.fromisoformat(to_date).date() if to_date else today
    from_d = datetime.date.fromisoformat(from_date) if from_date else (to_d - datetime.timedelta(days=30))
    return _build_dataset(from_d, to_d, force_refresh=refresh)


@app.get("/api/health")
def health():
    return {"status": "ok",  "time": datetime.datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),  "version": "2026-07-22-v2"}
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Regulatory Notices Dashboard API</title>
    </head>
    <body style="font-family: Arial; margin:40px;">
        <h1>✅ Regulatory Notices Dashboard API</h1>
        <p>Backend is running successfully.</p>

        <ul>
            <li><a href="/docs">Swagger Documentation</a></li>
            <li><a href="/api/health">Health Check</a></li>
            <li><a href="/api/notices">Latest Notices</a></li>
        </ul>
    </body>
    </html>
    """