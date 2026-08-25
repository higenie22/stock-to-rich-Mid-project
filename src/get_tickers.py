"""Retrieve and normalize the current S&P 500 universe."""

from __future__ import annotations

import io
import re
import time
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "cache"
REPO_ZIP = "https://codeload.github.com/fja05680/sp500/zip/refs/heads/master"
REPO_RAW = "https://raw.githubusercontent.com/fja05680/sp500/master/"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# Only verified ticker changes are applied.
TRUSTED_RENAMES = {
    "ABC": "COR", "ANTM": "ELV", "BLL": "BALL", "CDAY": "DAY",
    "FB": "META", "FBHS": "FBIN", "FLT": "CPAY", "FISV": "FI",
    "GPS": "GAP", "NLOK": "GEN", "PKI": "RVTY", "RE": "EG",
    "VIAC": "PARA", "WLTW": "WTW", "WRK": "SW",
}


def norm_ticker(value: str) -> str:
    """Convert a symbol to the Yahoo convention (for example BRK.B -> BRK-B)."""
    return str(value).strip().upper().replace(".", "-")


def _mirror_urls(url: str) -> list[str]:
    match = re.match(
        r"https://raw\.githubusercontent\.com/([^/]+)/([^/]+)/([^/]+)/(.+)", url
    )
    if not match:
        return [url]
    owner, repo, branch, path = match.groups()
    return [
        f"https://cdn.jsdelivr.net/gh/{owner}/{repo}@{branch}/{path}",
        f"https://github.com/{owner}/{repo}/raw/{branch}/{path}",
        url,
    ]


def _read_csv_url(url: str, retries: int = 2) -> Optional[pd.DataFrame]:
    for candidate in _mirror_urls(url):
        host = candidate.split("/")[2]
        for attempt in range(retries):
            try:
                response = requests.get(candidate, timeout=30, headers={"User-Agent": UA})
                if response.status_code == 429:
                    time.sleep(4 * (attempt + 1))
                    continue
                response.raise_for_status()
                frame = pd.read_csv(io.StringIO(response.text))
                print(f"[INFO] {host} 에서 확보 ({len(frame):,}행)")
                return frame
            except Exception as exc:
                print(f"[WARN] {host} 실패: {type(exc).__name__}")
                time.sleep(2)
    return None


def _read_from_repo_zip() -> Optional[pd.DataFrame]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = CACHE_DIR / "sp500_repo.zip"
    if not archive_path.exists():
        try:
            response = requests.get(REPO_ZIP, timeout=180, headers={"User-Agent": UA})
            response.raise_for_status()
            archive_path.write_bytes(response.content)
            print(f"[INFO] 저장소 zip 확보 ({len(response.content) / 1024:.0f} KB)")
        except Exception as exc:
            print(f"[WARN] zip 조회 실패: {type(exc).__name__}")
            return None
    try:
        with zipfile.ZipFile(archive_path) as archive:
            names = [name for name in archive.namelist() if name.endswith("/sp500.csv")]
            return pd.read_csv(archive.open(names[0])) if names else None
    except Exception as exc:
        print(f"[WARN] zip 읽기 실패: {type(exc).__name__}")
        return None


def _read_from_wikipedia() -> Optional[pd.DataFrame]:
    try:
        return pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    except Exception as exc:
        print(f"[WARN] 위키백과 폴백 실패: {type(exc).__name__}")
        return None


def get_sp500_universe(force_refresh: bool = False) -> pd.DataFrame:
    """Return the current constituents as ticker/company/sector data."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "sp500_universe.csv"
    if cache_path.exists() and not force_refresh:
        frame = pd.read_csv(cache_path)
        print(f"[캐시] 구성종목 {len(frame)}건 재사용 ({cache_path})")
        return frame

    raw = _read_from_repo_zip()
    if raw is None:
        raw = _read_csv_url(REPO_RAW + "sp500.csv")
    if raw is None:
        print("[WARN] 저장소 실패 -> 위키백과로 폴백")
        raw = _read_from_wikipedia()
    if raw is None:
        raise RuntimeError("현재 S&P 500 구성종목 목록을 어느 소스에서도 받지 못했습니다.")

    columns = {str(column).lower().strip(): column for column in raw.columns}

    def pick(*names: str):
        return next((columns[name] for name in names if name in columns), None)

    symbol_col = pick("symbol", "ticker")
    name_col = pick("security", "company", "name", "security name")
    sector_col = pick("gics sector", "sector")
    if symbol_col is None:
        raise RuntimeError(f"티커 컬럼을 찾지 못했습니다. 실제 컬럼: {list(raw.columns)}")

    frame = pd.DataFrame({"ticker": raw[symbol_col].map(norm_ticker)})
    frame["company"] = raw[name_col].astype(str).str.strip() if name_col else frame["ticker"]
    frame["sector"] = raw[sector_col].astype(str).str.strip() if sector_col else "Unknown"
    frame["ticker_original"] = frame["ticker"]
    frame["ticker"] = frame["ticker"].map(lambda ticker: TRUSTED_RENAMES.get(ticker, ticker))
    frame = (
        frame.dropna(subset=["ticker"])
        .query("ticker != '' and ticker != 'NAN'")
        .drop_duplicates("ticker", keep="first")
        .sort_values("ticker")
        .reset_index(drop=True)
    )
    frame.to_csv(cache_path, index=False)
    print(f"[INFO] 정규화/중복 제거 후 구성종목 {len(frame)}건")
    return frame
