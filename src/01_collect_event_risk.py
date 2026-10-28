#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01_collect_event_risk.py
================================

연구기간
--------
- 전체: 2024-10-01 ~ 2025-05-31
- Train/기준분포: 2024-10-01 ~ 2025-03-31
- Test: 2025-04-01 ~ 2025-05-31

역할
----
GDELT 2.0 이벤트와 기사 본문으로 event-level EventRisk를 만든 뒤,
날짜×시장별 원래 관측값과 기사 추출 coverage를 저장한다.

V3 핵심 수정
------------
1) 01/02/04의 Train 종료일을 2025-03-31로 통일한다.
2) 기사 coverage를 날짜 전체가 아니라 날짜×시장별로 계산한다.
3) 정상 source, partial source, source missing, genuine zero를 구분한다.
4) observed/raw/strict70 열 이름을 명확히 분리한다.
5) output을 methodology version 폴더에 저장해 다른 버전과 섞이지 않게 한다.
6) Pylance가 타입을 정상 인식하도록 static import를 사용한다.
7) GDELT/기사 cache는 기존 공용 cache를 그대로 재사용한다.

VS Code에서 이 파일을 직접 실행하면 된다. BigQuery/gcloud는 사용하지 않는다.
"""
from __future__ import annotations

import gc
import importlib
import json
import logging
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

# -----------------------------------------------------------------------------
# 0. VS Code에서 바로 실행하기 위한 의존성 자동 확인
# -----------------------------------------------------------------------------
def _ensure_package(import_name: str, pip_name: str) -> None:
    try:
        importlib.import_module(import_name)
    except ImportError:
        print(f"[SETUP] {pip_name} 설치 중... (최초 1회)")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])


for _import_name, _pip_name in [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("requests", "requests"),
    ("trafilatura", "trafilatura"),
    ("gdelt", "gdelt==0.1.14"),
]:
    _ensure_package(_import_name, _pip_name)

import numpy as np
import pandas as pd
import requests
import trafilatura
import gdelt

# -----------------------------------------------------------------------------
# 1. 사용자가 수정할 수 있는 설정: 여기만 보면 됨
# -----------------------------------------------------------------------------
START_DATE = date(2024, 10, 1)
END_DATE = date(2025, 5, 31)
TRAIN_END = date(2025, 3, 31)
TEST_START = date(2025, 4, 1)

# True: 영어 Event + GDELT 번역 Event 모두 사용 (연구용 권장)
# False: 영어 Event만 사용 (다운로드량 감소, 빠른 테스트용)
INCLUDE_TRANSLATED = True

ARTICLE_WORKERS = 10
MAX_URLS_PER_EVENT = 3
MIN_ARTICLE_COVERAGE = 0.70
USER_AGENT = "POSTECH-Commodity-EventRisk-Research/6.0-GDELT2-HTTP"

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR / "eventrisk_research_final_output"
METHODOLOGY_VERSION = "2024OCT_2025MAY_TRAIN6M_TEST2M_SOFT70_V3"

MARKET_TERMS = {
    "oil": [
        "oil", "crude oil", "brent crude", "wti crude", "petroleum",
        "refinery", "refineries", "oil field", "oilfield", "oil terminal",
        "oil tanker", "crude tanker", "petroleum terminal", "opec",
    ],
    "gas": [
        "natural gas", "lng", "liquefied natural gas", "gas pipeline",
        "gas field", "gas terminal", "lng terminal", "gas storage",
        "gas processing", "compressor station",
    ],
    "power": [
        "electricity", "power grid", "electric grid", "power plant",
        "nuclear plant", "substation", "transmission line",
        "transmission network", "distribution network", "transformer",
    ],
}

DISRUPTION_TERMS = [
    "attack", "attacked", "strike", "struck", "missile", "drone",
    "explosion", "exploded", "damage", "damaged", "destroyed",
    "shutdown", "shut down", "halted", "suspended", "stopped",
    "disrupted", "disruption", "interrupted", "interruption",
    "blocked", "blockade", "closed", "closure", "seized",
    "sanction", "sanctions", "embargo", "export ban", "import ban",
    "restriction", "restricted", "cut supply", "supply cut", "outage",
    "offline", "capacity reduction", "reduced capacity",
]


def compile_terms(terms: list[str]) -> list[re.Pattern]:
    return [re.compile(rf"(?<!\w){re.escape(t.lower())}(?!\w)") for t in terms]


MARKET_PATTERNS = {k: compile_terms(v) for k, v in MARKET_TERMS.items()}
DISRUPTION_PATTERNS = compile_terms(DISRUPTION_TERMS)


def ensure_dirs(root: Path):
    version_output = root / "outputs" / METHODOLOGY_VERSION
    version_logs = root / "logs" / METHODOLOGY_VERSION
    paths = {
        # 원자료/기사 cache는 버전 간 공용으로 재사용한다.
        "cache": root / "data" / "gdelt2_daily_cache",
        "articles": root / "data" / "raw" / "articles",
        # 계산 결과는 버전별 폴더로 분리한다.
        "outputs": version_output,
        "logs": version_logs,
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def setup_logging(path: Path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )


def daterange(start: date, end: date):
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def normalize_url(url):
    if not isinstance(url, str) or not url.strip():
        return None
    try:
        p = urlsplit(url.strip())
        return urlunsplit((
            p.scheme.lower() or "https",
            p.netloc.lower().removeprefix("www."),
            re.sub(r"/+$", "", p.path), "", "",
        ))
    except Exception:
        return url.strip()


def save_table(df: pd.DataFrame, stem: Path):
    # pyarrow가 없어도 반드시 저장되도록 CSV.GZ를 기본으로 사용한다.
    df.to_csv(stem.with_suffix(".csv.gz"), index=False, compression="gzip", encoding="utf-8-sig")


def read_saved_table(stem: Path) -> pd.DataFrame:
    gz = stem.with_suffix(".csv.gz")
    if not gz.exists():
        raise FileNotFoundError(gz)
    return pd.read_csv(gz, compression="gzip", low_memory=False)


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """gdeltPyR 버전에 따른 대/소문자 차이를 흡수한다."""
    wanted = [
        "GLOBALEVENTID", "SQLDATE", "Actor1Name", "Actor2Name",
        "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode",
        "QuadClass", "GoldsteinScale", "NumMentions", "NumSources",
        "NumArticles", "ActionGeo_FullName", "ActionGeo_FeatureID",
        "DATEADDED", "SOURCEURL",
    ]
    lookup = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in df.columns}
    ren = {}
    for w in wanted:
        key = re.sub(r"[^a-z0-9]", "", w.lower())
        if key in lookup:
            ren[lookup[key]] = w
    out = df.rename(columns=ren).copy()
    missing = [c for c in wanted if c not in out.columns]
    if missing:
        raise RuntimeError(
            "GDELT 2.0 결과의 예상 열을 찾지 못했습니다: " + ", ".join(missing) +
            "\n실제 열: " + ", ".join(map(str, df.columns[:80]))
        )
    return out[wanted]


def _metadata_candidate_mask(df: pd.DataFrame) -> pd.Series:
    text = (
        df["Actor1Name"].fillna("").astype(str) + " " +
        df["Actor2Name"].fillna("").astype(str) + " " +
        df["ActionGeo_FullName"].fillna("").astype(str) + " " +
        df["SOURCEURL"].fillna("").astype(str)
    ).str.lower()
    combined = r"(?<!\w)(?:" + "|".join(
        re.escape(t) for vals in MARKET_TERMS.values() for t in vals
    ) + r")(?!\w)"
    return text.str.contains(combined, regex=True, na=False)


def _filter_day(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    d = _canonicalize_columns(df)
    for c in ["IsRootEvent", "QuadClass", "GoldsteinScale", "NumMentions", "NumSources", "NumArticles", "DATEADDED", "SQLDATE", "GLOBALEVENTID"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    mask = (
        d["IsRootEvent"].eq(1)
        & d["QuadClass"].eq(4)
        & d["GoldsteinScale"].lt(0)
        & _metadata_candidate_mask(d)
    )
    return d.loc[mask].copy()


def _pull_one_dataset(gd, day: date, translation: bool):
    label = "translated" if translation else "english"
    try:
        # gdeltPyR 자체가 GDELT 2.0 공개 HTTP 파일을 병렬 GET한다.
        df = gd.Search(
            [day.strftime("%Y %m %d")],
            table="events",
            coverage=True,
            translation=translation,
        )
        if df is None:
            df = pd.DataFrame()
        logging.info("GDELT2 %s | %s | raw rows=%d", label, day, len(df))
        return df, True, None
    except Exception as exc:
        logging.warning("GDELT2 %s FAILED | %s | %r", label, day, exc)
        return pd.DataFrame(), False, repr(exc)


def fetch_gdelt2_daily(start: date, end: date, paths):
    """하루씩 받아 즉시 필터링하고 캐시한다. 전체 원자료를 디스크에 쌓지 않는다."""
    gd = gdelt.gdelt(version=2)
    candidate_parts = []
    availability_rows = []
    daily_log = []

    for i, day in enumerate(daterange(start, end), start=1):
        stem = paths["cache"] / f"candidates_{day:%Y%m%d}"
        meta_path = paths["cache"] / f"status_{day:%Y%m%d}.json"

        if stem.with_suffix(".csv.gz").exists() and meta_path.exists():
            cand = read_saved_table(stem)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            candidate_parts.append(cand)
            availability_rows.append(meta)
            logging.info("CACHE %s (%d/%d) | candidates=%d", day, i, (end-start).days+1, len(cand))
            continue

        eng, eng_ok, eng_err = _pull_one_dataset(gd, day, translation=False)
        trans = pd.DataFrame(); trans_ok = True; trans_err = None
        if INCLUDE_TRANSLATED:
            trans, trans_ok, trans_err = _pull_one_dataset(gd, day, translation=True)

        raw_rows = int(len(eng) + len(trans))
        frames = []
        if len(eng):
            eng = eng.copy(); eng["_gdelt_stream"] = "english"; frames.append(eng)
        if len(trans):
            trans = trans.copy(); trans["_gdelt_stream"] = "translated"; frames.append(trans)

        if frames:
            # stream 표시는 downstream에 필수는 아니므로 canonical filter 전에 합친다.
            candidates = pd.concat([_filter_day(x) for x in frames], ignore_index=True)
        else:
            candidates = pd.DataFrame(columns=[
                "GLOBALEVENTID", "SQLDATE", "Actor1Name", "Actor2Name", "IsRootEvent",
                "EventCode", "EventBaseCode", "EventRootCode", "QuadClass", "GoldsteinScale",
                "NumMentions", "NumSources", "NumArticles", "ActionGeo_FullName",
                "ActionGeo_FeatureID", "DATEADDED", "SOURCEURL"
            ])

        # BigQuery 버전과 같은 downstream 열명으로 맞춘다.
        candidates["EventTableNumSources"] = pd.to_numeric(candidates.get("NumSources"), errors="coerce")
        candidates["EventTableNumMentions"] = pd.to_numeric(candidates.get("NumMentions"), errors="coerce")
        candidates["EventTableNumArticles"] = pd.to_numeric(candidates.get("NumArticles"), errors="coerce")
        candidates["NumSourcesPIT"] = candidates["EventTableNumSources"]
        candidates["NumMentionsPIT"] = candidates["EventTableNumMentions"]
        candidates["NumArticlesPIT"] = candidates["EventTableNumArticles"]
        candidates["MentionStatsOK"] = False
        candidates["ArticleURL"] = candidates.get("SOURCEURL")
        candidates["url_rank"] = 1

        source_ok = bool(eng_ok and (trans_ok if INCLUDE_TRANSLATED else True) and raw_rows > 0)
        if source_ok:
            source_status = "GDELT2_HTTP_OK"
        elif raw_rows > 0:
            source_status = "GDELT2_HTTP_PARTIAL"
        else:
            source_status = "GDELT2_SOURCE_MISSING"

        meta = {
            "event_date": int(day.strftime("%Y%m%d")),
            "raw_event_count": raw_rows,
            "source_ok": source_ok,
            "source_status": source_status,
            "english_ok": bool(eng_ok),
            "translated_ok": bool(trans_ok) if INCLUDE_TRANSLATED else None,
            "english_rows": int(len(eng)),
            "translated_rows": int(len(trans)),
            "candidate_rows": int(len(candidates)),
            "english_error": eng_err,
            "translated_error": trans_err,
        }
        save_table(candidates, stem)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        candidate_parts.append(candidates)
        availability_rows.append(meta)
        daily_log.append(meta)
        logging.info(
            "DAY %s (%d/%d) | raw=%d candidates=%d status=%s",
            day, i, (end-start).days+1, raw_rows, len(candidates), source_status,
        )

        # 하루 원자료는 여기서 즉시 해제
        del eng, trans, frames, candidates
        gc.collect()

    availability = pd.DataFrame(availability_rows)
    availability.to_csv(paths["logs"] / "gdelt2_daily_download_log.csv", index=False, encoding="utf-8-sig")
    raw = pd.concat(candidate_parts, ignore_index=True) if candidate_parts else pd.DataFrame()
    return raw, availability


def match_terms(text: str, patterns: list[re.Pattern], original_terms: list[str]) -> list[str]:
    text = str(text).lower()
    return [term for term, pat in zip(original_terms, patterns) if pat.search(text)]


def metadata_market_matches(row):
    vals = [row.get("Actor1Name"), row.get("Actor2Name"), row.get("ActionGeo_FullName"),
            row.get("SOURCEURL"), row.get("ArticleURL")]
    text = " ".join(str(v).lower() for v in vals if pd.notna(v))
    return {m: match_terms(text, MARKET_PATTERNS[m], MARKET_TERMS[m]) for m in MARKET_TERMS}


def union_pipe(values) -> str:
    items = set()
    for v in values:
        if pd.isna(v):
            continue
        items.update(x for x in str(v).split("|") if x)
    return "|".join(sorted(items))


def prefilter_commodity_events(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in raw.iterrows():
        matches = metadata_market_matches(row)
        markets = [m for m, terms in matches.items() if terms]
        if not markets:
            continue
        out = row.to_dict()
        out["candidate_markets"] = "|".join(markets)
        out["metadata_match_terms"] = "|".join(sorted({t for terms in matches.values() for t in terms}))
        rows.append(out)
    return pd.DataFrame(rows)


def build_event_groups(filtered: pd.DataFrame, max_urls: int):
    df = filtered.copy()
    for c in ["GLOBALEVENTID", "SQLDATE", "DATEADDED"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["GLOBALEVENTID"].notna() & df["DATEADDED"].notna()].copy()
    df["event_id"] = df["GLOBALEVENTID"].astype("Int64").astype(str)
    df["first_seen_date"] = pd.to_numeric(df["DATEADDED"].astype("Int64").astype(str).str[:8], errors="coerce")
    df["occurrence_date"] = pd.to_numeric(df["SQLDATE"], errors="coerce")

    # Same GLOBALEVENTID can appear once per selected URL. Event attributes are summarized once.
    summary = (df.sort_values(["event_id", "DATEADDED"]).groupby("event_id", as_index=False).agg(
        first_seen_date=("first_seen_date", "min"),
        occurrence_date=("occurrence_date", "first"),
        actor1=("Actor1Name", "first"), actor2=("Actor2Name", "first"),
        event_code=("EventCode", "first"), event_base_code=("EventBaseCode", "first"),
        event_root_code=("EventRootCode", "first"), location=("ActionGeo_FullName", "first"),
        goldstein=("GoldsteinScale", "first"),
        num_sources=("NumSourcesPIT", "first"),
        num_mentions=("NumMentionsPIT", "first"),
        num_articles=("NumArticlesPIT", "first"),
        mention_stats_ok=("MentionStatsOK", "max"),
        candidate_markets=("candidate_markets", union_pipe),
        metadata_match_terms=("metadata_match_terms", union_pipe),
    ))

    # First-day EventMentions are preferred URLs; event-table SOURCEURL is a fallback.
    url_rows = []
    for row in df.itertuples(index=False):
        if isinstance(getattr(row, "ArticleURL", None), str) and getattr(row, "ArticleURL").strip():
            rank_value = getattr(row, "url_rank", np.nan)
            rank = int(rank_value) if pd.notna(rank_value) else 999
            url_rows.append((row.event_id, rank, getattr(row, "ArticleURL")))
        if isinstance(getattr(row, "SOURCEURL", None), str) and getattr(row, "SOURCEURL").strip():
            url_rows.append((row.event_id, 999, getattr(row, "SOURCEURL")))
    urls = pd.DataFrame(url_rows, columns=["event_id", "source_rank", "SOURCEURL"])
    if urls.empty:
        urls = pd.DataFrame(columns=["event_id", "url_rank", "SOURCEURL", "normalized_url"])
    else:
        urls["normalized_url"] = urls["SOURCEURL"].map(normalize_url)
        urls = urls.dropna(subset=["normalized_url"]).sort_values(["event_id", "source_rank"])
        urls = urls.drop_duplicates(["event_id", "normalized_url"])
        urls["url_rank"] = urls.groupby("event_id").cumcount() + 1
        urls = urls[urls["url_rank"] <= max_urls][["event_id", "url_rank", "SOURCEURL", "normalized_url"]]
    return summary, urls.reset_index(drop=True)


def ecdf_score_against_reference(values: pd.Series, reference: np.ndarray, positive_only=False) -> pd.Series:
    ref = np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    if positive_only:
        ref = ref[ref > 0]
    ref = np.sort(ref)
    out = pd.Series(np.nan, index=values.index, dtype=float)
    if len(ref) == 0:
        return out
    vals = pd.to_numeric(values, errors="coerce")
    for idx, v in vals.items():
        if not np.isfinite(v):
            continue
        if positive_only and v <= 0:
            out.loc[idx] = 0.0
            continue
        left = np.searchsorted(ref, v, side="left")
        right = np.searchsorted(ref, v, side="right")
        out.loc[idx] = (left + 0.5 * (right - left)) / len(ref)
    return out


def calculate_severity(summary: pd.DataFrame, calibration_end: date) -> pd.DataFrame:
    result = summary.copy()
    result["conflict"] = (-pd.to_numeric(result["goldstein"], errors="coerce")).clip(0, 10) / 10.0
    result["coverage_signal"] = np.log1p(pd.to_numeric(result["num_sources"], errors="coerce").clip(lower=0))
    cutoff = int(calibration_end.strftime("%Y%m%d"))
    ref = result.loc[result["first_seen_date"] <= cutoff, "coverage_signal"].to_numpy(float)
    if np.isfinite(ref).sum() < 2:
        raise RuntimeError("Too few calibration-period events to build the Coverage ECDF")
    result["coverage"] = ecdf_score_against_reference(result["coverage_signal"], ref)
    result["coverage_reference_n"] = int(np.isfinite(ref).sum())
    result["severity"] = 10.0 * np.sqrt(result["conflict"] * result["coverage"])
    return result


def fetch_article(task):
    event_id, url_rank, url, normalized_url = task
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=(5, 20), allow_redirects=True)
        r.raise_for_status()
        text = trafilatura.extract(
            r.text, include_comments=False, include_tables=False, favor_precision=True
        ) or ""
        return {
            "event_id": event_id, "url_rank": url_rank, "url": url,
            "normalized_url": normalized_url, "article_text": text.strip(),
            "text_extracted": int(bool(text.strip())), "error": None,
        }
    except Exception as exc:
        return {
            "event_id": event_id, "url_rank": url_rank, "url": url,
            "normalized_url": normalized_url, "article_text": "",
            "text_extracted": 0, "error": repr(exc),
        }


def select_best_articles(attempts: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    if attempts.empty:
        base = summary[["event_id"]].copy()
        base["url"] = np.nan; base["normalized_url"] = np.nan; base["article_text"] = ""
        base["text_extracted"] = 0; base["error"] = "NO_ARTICLE_ATTEMPT"; base["selected_url_rank"] = np.nan
        return base
    a = attempts.sort_values(["event_id", "text_extracted", "url_rank"], ascending=[True, False, True]).copy()
    best = a.drop_duplicates("event_id", keep="first").rename(columns={"url_rank": "selected_url_rank"})
    cols = ["event_id", "url", "normalized_url", "article_text", "text_extracted", "error", "selected_url_rank"]
    base = summary[["event_id"]].merge(best[cols], on="event_id", how="left")
    base["text_extracted"] = base["text_extracted"].fillna(0).astype(int)
    base["article_text"] = base["article_text"].fillna("")
    base["error"] = base["error"].fillna("NO_ARTICLE_ATTEMPT")
    return base


def split_sentences(text):
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    if not cleaned:
        return []
    return [s.strip().lower() for s in re.split(r"(?<=[.!?])\s+", cleaned) if len(s.strip()) >= 20]


def contains_pattern(sentence: str, patterns: list[re.Pattern]) -> bool:
    return any(p.search(sentence) for p in patterns)


def raw_exposure(text, market):
    sentences = split_sentences(text)
    if not sentences:
        return np.nan, 0, 0, ""
    matched = [
        s for s in sentences
        if contains_pattern(s, MARKET_PATTERNS[market]) and contains_pattern(s, DISRUPTION_PATTERNS)
    ]
    return len(matched) / len(sentences), len(matched), len(sentences), " || ".join(matched[:3])


def build_eventrisk(summary, articles, calibration_end: date):
    merged = summary.merge(articles, on="event_id", how="left")
    merged["article_ok"] = (
        merged["text_extracted"].fillna(0).astype(int).eq(1)
        & merged["article_text"].fillna("").str.strip().ne("")
    )
    cutoff = int(calibration_end.strftime("%Y%m%d"))

    for market in ["oil", "gas", "power"]:
        calc = merged.apply(
            lambda r: raw_exposure(r["article_text"], market) if r["article_ok"] else (np.nan, 0, 0, ""),
            axis=1,
        )
        merged[f"{market}_raw_exposure"] = calc.map(lambda x: x[0])
        merged[f"{market}_matched_sentence_count"] = calc.map(lambda x: x[1])
        merged[f"{market}_total_sentence_count"] = calc.map(lambda x: x[2])
        merged[f"{market}_evidence_sentences"] = calc.map(lambda x: x[3])

        ref_mask = (merged["first_seen_date"] <= cutoff) & merged["article_ok"]
        ref = merged.loc[ref_mask, f"{market}_raw_exposure"].to_numpy(float)
        merged[f"{market}_exposure"] = ecdf_score_against_reference(
            merged[f"{market}_raw_exposure"], ref, positive_only=True
        )
        merged[f"{market}_exposure_reference_n"] = int(np.sum(np.isfinite(ref) & (ref > 0)))
        merged[f"{market}_event_risk"] = merged["severity"] * merged[f"{market}_exposure"]
        merged.loc[~merged["article_ok"], f"{market}_event_risk"] = np.nan
    return merged


def _candidate_market_set(value) -> set[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    return {x.strip().lower() for x in str(value).split("|") if x.strip()}


def create_outputs(
    results: pd.DataFrame,
    start: date,
    end: date,
    paths,
    availability: pd.DataFrame,
    min_article_coverage: float,
):
    results = results.sort_values(
        ["first_seen_date", "severity"],
        ascending=[True, False],
    ).copy()
    results["methodology_version"] = METHODOLOGY_VERSION
    results.to_csv(
        paths["outputs"] / "event_risk_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    long_rows = []
    qc_rows = []

    for row in results.itertuples(index=False):
        candidate_markets = _candidate_market_set(
            getattr(row, "candidate_markets", "")
        )

        for market in ["oil", "gas", "power"]:
            raw_exp = getattr(row, f"{market}_raw_exposure")
            event_risk = getattr(row, f"{market}_event_risk")

            long_rows.append({
                "event_id": row.event_id,
                "event_date": int(row.first_seen_date),
                "occurrence_date": row.occurrence_date,
                "commodity": market.capitalize(),
                "article_ok": int(row.article_ok),
                "severity": row.severity,
                "raw_exposure": raw_exp,
                "exposure": getattr(row, f"{market}_exposure"),
                "event_risk": event_risk,
                "matched_sentence_count": getattr(
                    row, f"{market}_matched_sentence_count"
                ),
            })

            # 날짜×시장 coverage의 분모:
            # ① metadata상 해당 시장 후보이거나
            # ② 실제 기사에서 해당 시장 노출이 양수로 확인된 event.
            candidate_for_market = (
                market in candidate_markets
                or (pd.notna(raw_exp) and float(raw_exp) > 0)
            )
            if candidate_for_market:
                qc_rows.append({
                    "event_date": int(row.first_seen_date),
                    "commodity": market.capitalize(),
                    "event_id": str(row.event_id),
                    "article_ok": int(row.article_ok),
                })

    event_long = pd.DataFrame(long_rows)
    event_long.to_csv(
        paths["outputs"] / "event_risk_long.csv",
        index=False,
        encoding="utf-8-sig",
    )

    market_qc_events = pd.DataFrame(qc_rows)
    if market_qc_events.empty:
        market_qc = pd.DataFrame(columns=[
            "event_date", "commodity", "candidate_event_count",
            "article_success_count", "article_coverage_rate",
        ])
    else:
        market_qc = (
            market_qc_events
            .groupby(["event_date", "commodity"], as_index=False)
            .agg(
                candidate_event_count=("event_id", "nunique"),
                article_success_count=("article_ok", "sum"),
            )
        )
        market_qc["article_coverage_rate"] = np.where(
            market_qc["candidate_event_count"] > 0,
            market_qc["article_success_count"]
            / market_qc["candidate_event_count"],
            np.nan,
        )

    daily = (
        event_long
        .groupby(["event_date", "commodity"], as_index=False)
        .agg(
            daily_event_risk_observed=(
                "event_risk", lambda s: s.sum(min_count=1)
            ),
            event_count=("event_risk", lambda s: int((s > 0).sum())),
            max_event_risk=("event_risk", "max"),
            mean_event_risk=("event_risk", "mean"),
        )
        if not event_long.empty
        else pd.DataFrame(columns=[
            "event_date", "commodity", "daily_event_risk_observed",
            "event_count", "max_event_risk", "mean_event_risk",
        ])
    )

    all_dates = [int(d.strftime("%Y%m%d")) for d in daterange(start, end)]
    grid = pd.MultiIndex.from_product(
        [all_dates, ["Oil", "Gas", "Power"]],
        names=["event_date", "commodity"],
    ).to_frame(index=False)

    out = grid.merge(daily, on=["event_date", "commodity"], how="left")
    out = out.merge(
        market_qc,
        on=["event_date", "commodity"],
        how="left",
    )

    av = availability.copy()
    av["event_date"] = pd.to_numeric(
        av["event_date"], errors="coerce"
    ).astype("Int64")
    av["raw_event_count"] = pd.to_numeric(
        av["raw_event_count"], errors="coerce"
    ).fillna(0).astype(int)

    if "source_status" not in av.columns:
        av["source_status"] = np.where(
            av["raw_event_count"] > 0,
            "GDELT2_HTTP_OK",
            "GDELT2_SOURCE_MISSING",
        )

    av["source_status"] = av["source_status"].fillna(
        "GDELT2_SOURCE_MISSING"
    )
    av["source_complete"] = av["source_status"].eq("GDELT2_HTTP_OK")
    av["source_partial"] = av["source_status"].eq("GDELT2_HTTP_PARTIAL")
    av["source_available"] = av[
        "source_complete"
    ] | av["source_partial"]

    out = out.merge(
        av[[
            "event_date", "raw_event_count", "source_status",
            "source_complete", "source_partial", "source_available",
        ]],
        on="event_date",
        how="left",
    )

    out["source_status"] = out["source_status"].fillna(
        "GDELT2_SOURCE_MISSING"
    )
    for col in ["source_complete", "source_partial", "source_available"]:
        out[col] = out[col].fillna(False).astype(bool)
    out["raw_event_count"] = out["raw_event_count"].fillna(0).astype(int)
    out["candidate_event_count"] = out[
        "candidate_event_count"
    ].fillna(0).astype(int)
    out["article_success_count"] = out[
        "article_success_count"
    ].fillna(0).astype(int)

    # source가 완전한 날에 후보 event가 0개인 경우에만 genuine zero로 본다.
    out["genuine_zero"] = (
        out["source_complete"]
        & out["candidate_event_count"].eq(0)
    )
    out["partial_no_candidate_unknown"] = (
        out["source_partial"]
        & out["candidate_event_count"].eq(0)
    )

    out.loc[out["genuine_zero"], "article_coverage_rate"] = 1.0
    out.loc[out["genuine_zero"], "daily_event_risk_observed"] = 0.0
    out.loc[
        out["genuine_zero"],
        ["event_count", "max_event_risk", "mean_event_risk"],
    ] = 0.0

    # partial source에서 후보가 0개인 날은 '진짜 0'이라고 단정하지 않는다.
    out.loc[
        out["partial_no_candidate_unknown"],
        ["article_coverage_rate", "daily_event_risk_observed"],
    ] = np.nan

    out["daily_event_risk_raw"] = out[
        "daily_event_risk_observed"
    ].where(out["source_available"] | out["genuine_zero"])

    out["strict70_usable"] = (
        out["source_complete"]
        & out["daily_event_risk_observed"].notna()
        & out["article_coverage_rate"].ge(min_article_coverage)
    )
    out["daily_event_risk_strict70"] = out[
        "daily_event_risk_observed"
    ].where(out["strict70_usable"])

    out["qc_status"] = np.select(
        [
            out["genuine_zero"],
            out["source_status"].eq("GDELT2_SOURCE_MISSING"),
            out["partial_no_candidate_unknown"],
            out["source_partial"]
            & out["daily_event_risk_observed"].notna(),
            out["daily_event_risk_observed"].isna()
            & out["candidate_event_count"].gt(0),
            out["article_coverage_rate"].lt(min_article_coverage),
        ],
        [
            "GENUINE_ZERO",
            "SOURCE_MISSING",
            "PARTIAL_SOURCE_NO_CANDIDATE_UNKNOWN",
            "PARTIAL_SOURCE_OBSERVED",
            "ARTICLE_EXTRACTION_FAILED",
            "LOW_ARTICLE_COVERAGE",
        ],
        default="OK",
    )

    out["methodology_version"] = METHODOLOGY_VERSION
    out["period"] = np.where(
        pd.to_datetime(out["event_date"].astype(str), format="%Y%m%d")
        <= pd.Timestamp(TRAIN_END),
        "TRAIN_CALIBRATION",
        "TEST_EVALUATION",
    )

    out = out.sort_values(["event_date", "commodity"]).reset_index(drop=True)
    out.to_csv(
        paths["outputs"] / "dgri_input.csv",
        index=False,
        encoding="utf-8-sig",
    )

    qc_cols = [
        "event_date", "commodity", "source_status", "source_complete",
        "source_partial", "source_available", "raw_event_count",
        "candidate_event_count", "article_success_count",
        "article_coverage_rate", "genuine_zero",
        "partial_no_candidate_unknown", "strict70_usable", "qc_status",
    ]
    out[qc_cols].to_csv(
        paths["outputs"] / "daily_qc_market.csv",
        index=False,
        encoding="utf-8-sig",
    )
    av.to_csv(
        paths["outputs"] / "gdelt2_daily_source_availability.csv",
        index=False,
        encoding="utf-8-sig",
    )

    for value_col, filename in [
        ("daily_event_risk_observed", "daily_event_risk_observed.csv"),
        ("daily_event_risk_raw", "daily_event_risk_raw.csv"),
        ("daily_event_risk_strict70", "daily_event_risk_strict70.csv"),
    ]:
        wide = out.pivot(
            index="event_date",
            columns="commodity",
            values=value_col,
        ).reset_index()
        wide.columns.name = None
        wide.to_csv(
            paths["outputs"] / filename,
            index=False,
            encoding="utf-8-sig",
        )

    return out



def run_direct():
    paths = ensure_dirs(ROOT)
    setup_logging(paths["logs"] / "pipeline_gdelt2_http.log")

    if TRAIN_END < START_DATE or TRAIN_END >= END_DATE:
        raise ValueError("TRAIN_END must be within [START_DATE, END_DATE)")

    print("=" * 72)
    print("GDELT 2.0 EventRisk DIRECT RUN")
    print(f"Study period : {START_DATE} ~ {END_DATE}")
    print(f"Train        : {START_DATE} ~ {TRAIN_END}")
    print(f"Test         : {TEST_START} ~ {END_DATE}")
    print(f"Version      : {METHODOLOGY_VERSION}")
    print(f"Translated   : {INCLUDE_TRANSLATED}")
    print("Google Cloud / BigQuery / gcloud : NOT USED")
    print("=" * 72)

    raw, availability = fetch_gdelt2_daily(START_DATE, END_DATE, paths)
    if raw.empty:
        raise RuntimeError(
            "GDELT 2.0 다운로드는 완료됐지만 Oil/Gas/Power 후보 사건이 하나도 없습니다. "
            "logs/gdelt2_daily_download_log.csv를 확인하세요."
        )

    filtered = prefilter_commodity_events(raw)
    if filtered.empty:
        raise RuntimeError("No Oil/Gas/Power candidate events after exact local filtering")
    filtered.to_csv(paths["outputs"] / "commodity_filtered_events.csv", index=False, encoding="utf-8-sig")

    summary, urls = build_event_groups(filtered, MAX_URLS_PER_EVENT)
    summary = calculate_severity(summary, TRAIN_END)
    urls.to_csv(paths["outputs"] / "representative_urls.csv", index=False, encoding="utf-8-sig")

    attempt_stem = paths["articles"] / "article_attempts"
    try:
        cached_attempts = read_saved_table(attempt_stem)
    except FileNotFoundError:
        cached_attempts = pd.DataFrame()

    cached_keys = set()
    if not cached_attempts.empty and {"event_id", "normalized_url"}.issubset(cached_attempts.columns):
        cached_keys = set(zip(cached_attempts["event_id"].astype(str), cached_attempts["normalized_url"].astype(str)))

    article_tasks = []
    for r in urls.itertuples(index=False):
        key = (str(r.event_id), str(r.normalized_url))
        if key not in cached_keys:
            article_tasks.append((r.event_id, int(r.url_rank), r.SOURCEURL, r.normalized_url))

    new_attempts = []
    with ThreadPoolExecutor(max_workers=ARTICLE_WORKERS) as ex:
        futures = [ex.submit(fetch_article, t) for t in article_tasks]
        for f in as_completed(futures):
            result = f.result()
            new_attempts.append(result)
            logging.info(
                "article | event=%s | rank=%s | extracted=%s",
                result["event_id"], result["url_rank"], result["text_extracted"]
            )

    new_attempts = pd.DataFrame(new_attempts)
    attempts = pd.concat([cached_attempts, new_attempts], ignore_index=True) if not cached_attempts.empty else new_attempts
    if attempts.empty:
        attempts = pd.DataFrame(columns=[
            "event_id", "url_rank", "url", "normalized_url", "article_text", "text_extracted", "error"
        ])
    else:
        attempts["event_id"] = attempts["event_id"].astype(str)
        attempts = attempts.drop_duplicates(["event_id", "normalized_url"], keep="last")
    save_table(attempts, attempt_stem)

    selected = select_best_articles(attempts, summary)
    save_table(selected, paths["articles"] / "selected_articles")

    results = build_eventrisk(summary, selected, TRAIN_END)
    daily_out = create_outputs(
        results, START_DATE, END_DATE, paths, availability, MIN_ARTICLE_COVERAGE
    )

    stats = pd.DataFrame([
        {"stage": "gdelt2_http_candidate_rows", "count": len(raw)},
        {"stage": "commodity_prefilter_rows", "count": len(filtered)},
        {"stage": "unique_events", "count": len(summary)},
        {"stage": "representative_urls", "count": len(urls)},
        {"stage": "events_with_article", "count": int(selected["text_extracted"].sum())},
        {"stage": "source_missing_or_partial_days", "count": int((~availability["source_ok"].astype(bool)).sum())},
    ])
    stats.to_csv(paths["outputs"] / "pipeline_statistics.csv", index=False, encoding="utf-8-sig")

    print("\n01 completed successfully.")
    manifest = {
        "methodology_version": METHODOLOGY_VERSION,
        "study_start": str(START_DATE),
        "train_end": str(TRAIN_END),
        "test_start": str(TEST_START),
        "study_end": str(END_DATE),
        "min_article_coverage": MIN_ARTICLE_COVERAGE,
        "include_translated": INCLUDE_TRANSLATED,
        "dgri_input": str((paths["outputs"] / "dgri_input.csv").resolve()),
        "n_daily_rows": int(len(daily_out)),
        "n_source_missing_rows": int(daily_out["qc_status"].eq("SOURCE_MISSING").sum()),
        "n_partial_observed_rows": int(daily_out["qc_status"].eq("PARTIAL_SOURCE_OBSERVED").sum()),
        "n_low_coverage_rows": int(daily_out["qc_status"].eq("LOW_ARTICLE_COVERAGE").sum()),
        "run_timestamp_utc": pd.Timestamp.utcnow().isoformat(),
    }
    (paths["outputs"] / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"DGRI input: {paths['outputs'] / 'dgri_input.csv'}")
    print("Next: open 02_build_dgri.py and press ▶")


if __name__ == "__main__":
    run_direct()