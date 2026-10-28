#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03_analyze_risk_transmission.py
========================================

고정 연구설계
------------
- 전체: 2024-10-01 ~ 2025-05-31
- Train: 2024-10-01 ~ 2025-03-31
- Test : 2025-04-01 ~ 2025-05-31
- 메인 EventRisk: SOFT-70

분석
----
A. Test shock occurrence: 90/95/97.5 percentile 민감도
B. Test risk transmission: log(1+EventRisk) Train z-score VAR/GFEVD
   - percentile VAR/GFEVD는 robustness check
   - TCI와 residual-bootstrap confidence interval 포함
C. Current accumulated risk: 상대 leader + 절대 LOW/NORMAL/ELEVATED 상태
D. OOS validation: Train-mean/persistence benchmark, 최소 표본 기준 포함
E. Daily rolling connectedness: 직전 30일을 이용해 매일 VAR(1)/GFEVD 재추정
F. Episode matching: 동일 사건의 cross-market onset을 보는 선택적 보조분석

04에서는 EventRisk, phi, DGRI를 재계산하지 않고 02 V3 output만 읽는다.
"""

from __future__ import annotations

import importlib
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path


def _ensure_package(import_name: str, pip_name: str) -> None:
    try:
        importlib.import_module(import_name)
    except ImportError:
        print(f"[SETUP] {pip_name} 설치 중... (최초 1회)")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])


for _import_name, _pip_name in [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy.stats", "scipy"),
    ("sklearn.feature_extraction.text", "scikit-learn"),
    ("sklearn.metrics.pairwise", "scikit-learn"),
    ("matplotlib.pyplot", "matplotlib"),
]:
    _ensure_package(_import_name, _pip_name)

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


# =============================================================================
# 0. 고정 연구 설정
# =============================================================================
COMMODITIES = ["Oil", "Gas", "Power"]
METHODOLOGY_VERSION = "2024OCT_2025MAY_TRAIN6M_TEST2M_SOFT70_V3"
STUDY_START = pd.Timestamp("2024-10-01")
CALIBRATION_END = date(2025, 3, 31)
TEST_START = pd.Timestamp("2025-04-01")
STUDY_END = pd.Timestamp("2025-05-31")

# A. Shock occurrence / optional episode
SHOCK_THRESHOLDS = [90.0, 95.0, 97.5]
MAIN_SHOCK_PERCENTILE = 95.0
EVENT_CONTRIBUTION_SHARE = 0.70
MAX_EVENTS_PER_SHOCK = 10
EPISODE_MAX_GAP_DAYS = 3
EPISODE_MAX_LENGTH_DAYS = 7
ACTOR_JACCARD_MIN = 0.25
TEXT_COSINE_MIN = 0.15
TEXT_COSINE_STRONG = 0.25

# B. VAR / GFEVD
VAR_MAX_LAG = 3
MIN_VAR_EFFECTIVE_OBS = 30
FEVD_HORIZON = 5
CURRENT_WINDOW_DAYS = 30

# Daily rolling transmission history
# - Every as-of date uses ONLY the most recent 30 calendar days, including that date.
# - The 30-day window is moved by one day and re-estimated every day.
# - To keep each daily network comparable and estimable in a short window,
#   the rolling model is fixed at VAR(1). A strict 30-day slice yields at most
#   29 regression observations, so at least 25 valid consecutive observations
#   are required.
# - Daily NET/TCI are point estimates. Residual-bootstrap confidence intervals
#   are calculated at month-end checkpoints (including the final date), rather
#   than at every date, to keep runtime practical.
ROLLING_WINDOW_DAYS = 30
ROLLING_STEP_DAYS = 1
ROLLING_VAR_LAG = 1
ROLLING_MIN_EFFECTIVE_OBS = 25
ROLLING_BOOTSTRAP_REPS = 200
ROLLING_BOOTSTRAP_MONTH_END_ONLY = True
ROLLING_INCLUDE_FINAL_DATE = True

MAIN_VAR_COLUMN = "eventrisk_log1p_zscore_train_soft70"
ROBUST_VAR_COLUMN = "eventrisk_percentile_train_soft70"
BOOTSTRAP_REPS = 300
BOOTSTRAP_SEED = 20250815

# C. OOS validation
OOS_HORIZONS = [1, 3, 5]
DGRI_THRESHOLDS = [70.0, 80.0, 90.0]
MAIN_DGRI_THRESHOLD = 80.0
MIN_HIGH_DGRI_DAYS = 5
MIN_OTHER_DAYS = 10
MIN_LEADER_DAYS = 5

SCRIPT_DIR = Path(__file__).resolve().parent
EVENT_INPUT = (
    SCRIPT_DIR / "eventrisk_research_final_output" / "outputs"
    / METHODOLOGY_VERSION / "event_risk_results.csv"
)
DAILY_INPUT = (
    SCRIPT_DIR / "dgri_output" / METHODOLOGY_VERSION
    / "shock_decay_dgri_long_final.csv"
)
ANALYSIS_VERSION = (
    f"{METHODOLOGY_VERSION}_ROLL{ROLLING_WINDOW_DAYS}_DAILY_VAR{ROLLING_VAR_LAG}"
)
OUTPUT_DIR = SCRIPT_DIR / "risk_leadership_output" / ANALYSIS_VERSION
GRAPH_DIR = OUTPUT_DIR / "graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. 파일 탐색 / 공통 유틸
# =============================================================================
def resolve_inputs():
    if not DAILY_INPUT.exists():
        raise FileNotFoundError(
            "02 V3 output을 찾지 못했습니다. 먼저 02_build_dgri.py를 "
            f"실행하세요.\nExpected: {DAILY_INPUT}"
        )
    event_path = EVENT_INPUT if EVENT_INPUT.exists() else None
    if event_path is None:
        print(
            "[WARN] event_risk_results.csv가 없어 episode 보조분석은 생략합니다."
        )
    return event_path, DAILY_INPUT


def ecdf_midpoint(values, reference):
    """Calibration ECDF midpoint percentile, 0~1."""
    ref = np.asarray(reference, dtype=float)
    ref = np.sort(ref[np.isfinite(ref)])
    vals = np.asarray(values, dtype=float)
    out = np.full(len(vals), np.nan, dtype=float)
    if len(ref) == 0:
        return out
    for i, v in enumerate(vals):
        if not np.isfinite(v):
            continue
        left = np.searchsorted(ref, v, side="left")
        right = np.searchsorted(ref, v, side="right")
        out[i] = (left + 0.5 * (right - left)) / len(ref)
    return out


def normalize_text_tokens(value) -> set[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    text = str(value).lower()
    return set(re.findall(r"[a-z0-9가-힣]{2,}", text))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def geo_tokens(value) -> set[str]:
    """ActionGeo_FullName의 comma component + 단어 token을 함께 사용."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    text = str(value).lower()
    comps = {re.sub(r"\s+", " ", x.strip()) for x in text.split(",") if len(x.strip()) >= 2}
    words = normalize_text_tokens(text)
    return comps | words


def clean_event_id(v) -> str:
    if pd.isna(v):
        return ""
    s = str(v)
    return s[:-2] if s.endswith(".0") else s


# =============================================================================
# 2. 02 결과 로드 — DGRI 재계산 금지
# =============================================================================
def load_daily_from_02(daily_path: Path):
    df = pd.read_csv(daily_path, low_memory=False)

    required = [
        "date", "commodity", "daily_event_risk_soft70",
        "eventrisk_percentile_train_soft70", "eventrisk_log1p_zscore_train_soft70",
        "dgri_soft70", "dgri_percentile_train_soft70",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            "02 V3 output에 필요한 열이 없습니다: " + ", ".join(missing)
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["commodity"] = (
        df["commodity"].astype(str).str.strip().str.capitalize()
    )

    numeric_cols = [
        "daily_event_risk_soft70", "eventrisk_percentile_train_soft70",
        "eventrisk_log1p_zscore_train_soft70", "dgri_soft70",
        "dgri_percentile_train_soft70", "dgri_zscore_train_soft70",
        "decay_phi_soft70", "decay_phi_raw_soft70", "decay_intercept_soft70",
        "article_coverage_rate", "soft70_weight",
        "daily_event_risk_observed",
        "eventrisk_percentile_train_raw",
        "eventrisk_percentile_train_soft70",
        "eventrisk_percentile_train_strict70",
        "eventrisk_log1p_zscore_train_raw",
        "eventrisk_log1p_zscore_train_soft70",
        "eventrisk_log1p_zscore_train_strict70",
        "dgri_percentile_train_raw",
        "dgri_percentile_train_soft70",
        "dgri_percentile_train_strict70",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[
        df["commodity"].isin(COMMODITIES)
        & df["date"].between(STUDY_START, STUDY_END)
    ].copy()

    if "methodology_version" not in df.columns:
        raise RuntimeError("02 output에 methodology_version이 없습니다.")
    versions = set(df["methodology_version"].dropna().astype(str).unique())
    if versions != {METHODOLOGY_VERSION}:
        raise RuntimeError(
            "02/04 methodology version이 다릅니다.\n"
            f"Found: {sorted(versions)}\nExpected: {METHODOLOGY_VERSION}"
        )

    dup = df.duplicated(["date", "commodity"], keep=False)
    if dup.any():
        bad = df.loc[dup, ["date", "commodity"]].drop_duplicates()
        raise RuntimeError(
            "02 output에 date/commodity 중복이 있습니다:\n"
            + bad.to_string(index=False)
        )

    return df.sort_values(["date", "commodity"]).reset_index(drop=True)


def build_decay_parameter_summary(daily: pd.DataFrame):
    rows=[]
    for commodity in COMMODITIES:
        g=daily[daily["commodity"].eq(commodity)].copy()
        if g.empty: continue
        def first_finite(col):
            if col not in g.columns: return np.nan
            x=pd.to_numeric(g[col],errors="coerce").dropna()
            return float(x.iloc[0]) if not x.empty else np.nan
        phi=first_finite("decay_phi_soft70")
        half_life=math.log(0.5)/math.log(phi) if np.isfinite(phi) and 0<phi<1 else (0.0 if phi==0 else np.nan)
        rows.append({"commodity":commodity,"intercept":first_finite("decay_intercept_soft70"),"phi_raw":first_finite("decay_phi_raw_soft70"),"phi_used":phi,"half_life_days":half_life,"specification":"SOFT70_V3_FROM_02"})
    return pd.DataFrame(rows)

# =============================================================================
# 3. Shock day 탐지
# =============================================================================
def detect_shock_days(daily: pd.DataFrame, threshold: float):
    mask = (
        daily["daily_event_risk_soft70"].notna()
        & daily["eventrisk_percentile_train_soft70"].ge(float(threshold))
        & daily["date"].ge(TEST_START)
    )
    keep = [
        "date", "commodity", "daily_event_risk_soft70",
        "eventrisk_percentile_train_soft70",
    ]
    for optional in [
        "article_coverage_rate", "soft70_weight", "event_count",
        "max_event_risk",
    ]:
        if optional in daily.columns:
            keep.append(optional)
    shocks = (
        daily.loc[mask, keep]
        .sort_values(["date", "commodity"])
        .reset_index(drop=True)
    )
    shocks["shock_threshold"] = float(threshold)
    shocks["shock_id"] = [f"S{i:04d}" for i in range(1, len(shocks) + 1)]
    return shocks


def summarize_shock_occurrence(
    shocks: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    rows = []
    for commodity in COMMODITIES:
        g = shocks[shocks["commodity"].eq(commodity)]
        rows.append({
            "threshold": float(threshold),
            "commodity": commodity,
            "shock_days": int(len(g)),
            "mean_shock_percentile": (
                float(g["eventrisk_percentile_train_soft70"].mean())
                if not g.empty else np.nan
            ),
            "max_shock_percentile": (
                float(g["eventrisk_percentile_train_soft70"].max())
                if not g.empty else np.nan
            ),
            "first_shock_date": (
                pd.Timestamp(g["date"].min()).date().isoformat()
                if not g.empty else ""
            ),
            "last_shock_date": (
                pd.Timestamp(g["date"].max()).date().isoformat()
                if not g.empty else ""
            ),
        })
    out = pd.DataFrame(rows)
    total = int(out["shock_days"].sum())
    out["share_of_all_shock_days"] = (
        out["shock_days"] / total if total > 0 else 0.0
    )
    return out.sort_values(
        ["shock_days", "max_shock_percentile", "commodity"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def shock_threshold_sensitivity(daily: pd.DataFrame):
    summaries = []
    shocks_by_threshold = {}
    for threshold in SHOCK_THRESHOLDS:
        shocks = detect_shock_days(daily, threshold)
        shocks_by_threshold[threshold] = shocks
        summaries.append(summarize_shock_occurrence(shocks, threshold))
    return pd.concat(summaries, ignore_index=True), shocks_by_threshold


# =============================================================================
# 4. shock day를 만든 event-level node 추출
# =============================================================================
def canonicalize_event_columns(events: pd.DataFrame):
    # 현재 01 pipeline 출력명을 우선 사용하되 대소문자/예전 이름 일부 흡수
    lookup = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in events.columns}
    wanted_alias = {
        "event_id": ["eventid", "globaleventid"],
        "first_seen_date": ["firstseendate", "eventdate", "dateadded"],
        "actor1": ["actor1", "actor1name"],
        "actor2": ["actor2", "actor2name"],
        "event_code": ["eventcode"],
        "event_root_code": ["eventrootcode", "eventroot"],
        "location": ["location", "actiongeofullname"],
        "article_text": ["articletext", "text"],
        "article_ok": ["articleok", "textextracted"],
        "oil_event_risk": ["oileventrisk"],
        "gas_event_risk": ["gaseventrisk"],
        "power_event_risk": ["powereventrisk"],
    }
    rename = {}
    for target, aliases in wanted_alias.items():
        if target in events.columns:
            continue
        for alias in aliases:
            if alias in lookup:
                rename[lookup[alias]] = target
                break
    events = events.rename(columns=rename).copy()

    required = [
        "event_id", "first_seen_date", "actor1", "actor2", "event_root_code", "location",
        "article_text", "oil_event_risk", "gas_event_risk", "power_event_risk"
    ]
    missing = [c for c in required if c not in events.columns]
    if missing:
        raise RuntimeError(
            "event_risk_results.csv에 episode 분석에 필요한 열이 없습니다: " + ", ".join(missing)
            + "\n실제 열 일부: " + ", ".join(map(str, events.columns[:80]))
        )
    return events


def extract_shock_event_nodes(event_path: Path, shocks: pd.DataFrame):
    events = canonicalize_event_columns(pd.read_csv(event_path, low_memory=False))
    events["event_id"] = events["event_id"].map(clean_event_id)
    # first_seen_date is YYYYMMDD in current pipeline
    fs = pd.to_numeric(events["first_seen_date"], errors="coerce").astype("Int64").astype(str).str[:8]
    events["date"] = pd.to_datetime(fs, format="%Y%m%d", errors="coerce")
    for c in ["oil_event_risk", "gas_event_risk", "power_event_risk"]:
        events[c] = pd.to_numeric(events[c], errors="coerce")

    nodes = []
    for shock in shocks.itertuples(index=False):
        market_col = f"{shock.commodity.lower()}_event_risk"
        cand = events.loc[
            events["date"].eq(shock.date)
            & events[market_col].gt(0)
        ].copy()
        if cand.empty:
            continue
        cand = cand.sort_values(market_col, ascending=False)
        total = float(cand[market_col].sum())
        if total <= 0:
            continue
        cand["contribution_share"] = cand[market_col] / total
        cand["cum_share"] = cand["contribution_share"].cumsum()
        # threshold를 넘기는 최초 event까지 포함
        keep_n = int(np.searchsorted(cand["cum_share"].to_numpy(), EVENT_CONTRIBUTION_SHARE, side="left") + 1)
        keep_n = max(1, min(keep_n, MAX_EVENTS_PER_SHOCK, len(cand)))
        cand = cand.head(keep_n)
        for row in cand.itertuples(index=False):
            nodes.append({
                "shock_id": shock.shock_id,
                "shock_date": shock.date,
                "commodity": shock.commodity,
                "daily_eventrisk_percentile": float(
                    shock.eventrisk_percentile_train_soft70
                ),
                "daily_eventrisk": float(shock.daily_event_risk_soft70),
                "event_id": clean_event_id(row.event_id),
                "market_event_risk": float(getattr(row, market_col)),
                "contribution_share": float(row.contribution_share),
                "actor1": getattr(row, "actor1", ""),
                "actor2": getattr(row, "actor2", ""),
                "event_code": getattr(row, "event_code", np.nan) if "event_code" in cand.columns else np.nan,
                "event_root_code": getattr(row, "event_root_code", np.nan),
                "location": getattr(row, "location", ""),
                "article_text": getattr(row, "article_text", "") if isinstance(getattr(row, "article_text", ""), str) else "",
            })
    nodes_df = pd.DataFrame(nodes)
    if nodes_df.empty:
        return nodes_df
    nodes_df = nodes_df.sort_values(["shock_date", "commodity", "market_event_risk"], ascending=[True, True, False]).reset_index(drop=True)
    nodes_df["node_id"] = [f"N{i:05d}" for i in range(1, len(nodes_df)+1)]
    return nodes_df


# =============================================================================
# 5. event-level episode clustering
# =============================================================================
@dataclass
class Episode:
    episode_id: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    member_indices: list[int] = field(default_factory=list)


def prepare_node_features(nodes: pd.DataFrame):
    n = nodes.copy().reset_index(drop=True)
    n["actor_tokens"] = [
        normalize_text_tokens(str(a1) + " " + str(a2))
        for a1, a2 in zip(n["actor1"], n["actor2"])
    ]
    n["geo_tokens"] = n["location"].map(geo_tokens)
    n["root_code_norm"] = n["event_root_code"].astype(str).replace({"nan":"", "<NA>":""})

    texts = n["article_text"].fillna("").astype(str).str.slice(0, 8000).tolist()
    if any(t.strip() for t in texts):
        vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", max_features=6000, ngram_range=(1,2), min_df=1
        )
        tfidf = vectorizer.fit_transform(texts)
        text_sim = cosine_similarity(tfidf)
    else:
        text_sim = np.zeros((len(n), len(n)), dtype=float)
    return n, text_sim


def node_link_score(a, b, text_similarity: float):
    same_event = bool(a.event_id and b.event_id and a.event_id == b.event_id)
    actor_j = jaccard(a.actor_tokens, b.actor_tokens)
    actor_match = actor_j >= ACTOR_JACCARD_MIN
    geo_j = jaccard(a.geo_tokens, b.geo_tokens)
    geo_match = geo_j > 0
    code_match = bool(a.root_code_norm and b.root_code_norm and a.root_code_norm == b.root_code_norm)
    text_match = text_similarity >= TEXT_COSINE_MIN

    if same_event:
        qualifies = True
    else:
        supporting = sum([actor_match, code_match, text_match])
        if geo_match:
            qualifies = supporting >= 2
        else:
            # location 누락/표현 차이 시 더 강한 actor + text 증거 필요
            qualifies = actor_j >= 0.50 and code_match and text_similarity >= TEXT_COSINE_STRONG

    score = (
        4.0 * same_event
        + 1.5 * geo_match
        + 1.2 * actor_match
        + 1.0 * code_match
        + 2.0 * text_similarity
    )
    return qualifies, score, {
        "same_event": same_event,
        "actor_jaccard": actor_j,
        "geo_jaccard": geo_j,
        "code_match": code_match,
        "text_cosine": text_similarity,
    }


def cluster_episodes(nodes: pd.DataFrame):
    if nodes.empty:
        return nodes.copy(), pd.DataFrame(), pd.DataFrame()
    n, text_sim = prepare_node_features(nodes)
    episodes: list[Episode] = []
    link_diagnostics = []

    for idx, row in n.iterrows():
        best_ep = None
        best_score = -np.inf
        best_detail = None
        for ep in episodes:
            if (row["shock_date"] - ep.end_date).days > EPISODE_MAX_GAP_DAYS:
                continue
            if (row["shock_date"] - ep.start_date).days > EPISODE_MAX_LENGTH_DAYS:
                continue
            # episode의 어느 member와든 충분히 유사하면 연결. 가장 높은 score를 사용.
            local_best_score = -np.inf
            local_best_detail = None
            local_best_member = None
            for j in ep.member_indices:
                qualifies, score, detail = node_link_score(
                    n.loc[idx], n.loc[j], float(text_sim[idx, j])
                )
                if qualifies and score > local_best_score:
                    local_best_score = score
                    local_best_detail = detail
                    local_best_member = j
            if local_best_score > best_score:
                best_score = local_best_score
                if local_best_member is not None:
                    best_ep = ep
                    best_detail = (local_best_member, local_best_detail)

        if best_ep is None:
            ep = Episode(
                episode_id=f"E{len(episodes)+1:03d}",
                start_date=row["shock_date"],
                end_date=row["shock_date"],
                member_indices=[idx],
            )
            episodes.append(ep)
            n.loc[idx, "episode_id"] = ep.episode_id
        else:
            best_ep.member_indices.append(idx)
            best_ep.end_date = max(best_ep.end_date, row["shock_date"])
            n.loc[idx, "episode_id"] = best_ep.episode_id
            j, detail = best_detail
            link_diagnostics.append({
                "episode_id": best_ep.episode_id,
                "from_node": n.loc[j, "node_id"],
                "to_node": row["node_id"],
                "score": best_score,
                **detail,
            })

    episode_rows = []
    for ep in episodes:
        g = n.loc[ep.member_indices].copy()
        first_by_market = g.groupby("commodity")["shock_date"].min().to_dict()
        earliest = min(first_by_market.values())
        origins = sorted([m for m, d in first_by_market.items() if d == earliest])
        origin = origins[0] if len(origins) == 1 else "Joint:" + "+".join(origins)
        commodities = sorted(g["commodity"].unique().tolist())
        top_locations = g["location"].fillna("").astype(str).value_counts().head(3).index.tolist()
        top_actors = pd.concat([
            g["actor1"].fillna("").astype(str), g["actor2"].fillna("").astype(str)
        ]).replace("", np.nan).dropna().value_counts().head(5).index.tolist()

        episode_rows.append({
            "episode_id": ep.episode_id,
            "start_date": ep.start_date,
            "end_date": ep.end_date,
            "duration_days": int((ep.end_date - ep.start_date).days + 1),
            "origin_leader": origin,
            "cross_commodity": len(commodities) >= 2,
            "n_commodities": len(commodities),
            "commodities": "|".join(commodities),
            "oil_first_date": first_by_market.get("Oil", pd.NaT),
            "gas_first_date": first_by_market.get("Gas", pd.NaT),
            "power_first_date": first_by_market.get("Power", pd.NaT),
            "n_event_nodes": len(g),
            "n_unique_events": g["event_id"].nunique(),
            "peak_daily_eventrisk_percentile": float(g["daily_eventrisk_percentile"].max()),
            "top_locations": " | ".join(top_locations),
            "top_actors": " | ".join(top_actors),
        })

    episodes_df = pd.DataFrame(episode_rows).sort_values(["start_date", "episode_id"]).reset_index(drop=True)
    diagnostics_df = pd.DataFrame(link_diagnostics)
    return n, episodes_df, diagnostics_df


def summarize_onset(episodes: pd.DataFrame):
    if episodes.empty:
        return pd.DataFrame(columns=["origin_leader","episode_count","share"])
    cross = episodes[episodes["cross_commodity"]].copy()
    if cross.empty:
        return pd.DataFrame(columns=["origin_leader","episode_count","share"])
    s = cross["origin_leader"].value_counts().rename_axis("origin_leader").reset_index(name="episode_count")
    s["share"] = s["episode_count"] / s["episode_count"].sum()
    return s


# =============================================================================
# 6. Calendar-consecutive VAR (결측을 압축하지 않음)
# =============================================================================
def make_eventrisk_panel(daily: pd.DataFrame, value_col: str):
    if value_col not in daily.columns:
        raise ValueError(f"VAR input column not found: {value_col}")
    p = daily.pivot(
        index="date", columns="commodity", values=value_col
    ).sort_index()
    return p.reindex(columns=COMMODITIES)


def build_var_design(panel: pd.DataFrame, p: int, start_date=None, end_date=None):
    idx_set = set(panel.index)
    X_rows, Y_rows, used_dates = [], [], []
    for t in panel.index:
        if start_date is not None and t < start_date:
            continue
        if end_date is not None and t > end_date:
            continue
        y = panel.loc[t].to_numpy(float)
        if not np.isfinite(y).all():
            continue
        lags = []
        ok = True
        for lag in range(1, p+1):
            d = t - pd.Timedelta(days=lag)
            if d not in idx_set:
                ok = False; break
            vals = panel.loc[d].to_numpy(float)
            if not np.isfinite(vals).all():
                ok = False; break
            lags.extend(vals.tolist())
        if not ok:
            continue
        X_rows.append([1.0] + lags)
        Y_rows.append(y.tolist())
        used_dates.append(t)
    if not X_rows:
        return np.empty((0, 1+len(COMMODITIES)*p)), np.empty((0, len(COMMODITIES))), []
    return np.asarray(X_rows, float), np.asarray(Y_rows, float), used_dates


def fit_var_ols(panel: pd.DataFrame, p: int, start_date=None, end_date=None):
    X, Y, used_dates = build_var_design(panel, p, start_date, end_date)
    n, q = X.shape
    k = Y.shape[1] if Y.ndim == 2 and Y.size else len(COMMODITIES)
    if n <= q + 2:
        raise RuntimeError(f"VAR({p}) 유효관측치 {n}개로 파라미터 {q}개를 안정적으로 추정하기 어렵습니다.")
    B = np.linalg.lstsq(X, Y, rcond=None)[0]  # q x k
    E = Y - X @ B
    sigma = (E.T @ E) / n
    sign, logdet = np.linalg.slogdet(sigma)
    if sign <= 0:
        logdet = np.log(max(np.linalg.det(sigma + np.eye(k)*1e-10), 1e-300))
    total_params = k * q
    aic = logdet + 2.0 * total_params / n
    bic = logdet + math.log(n) * total_params / n

    A = []
    for lag in range(p):
        block = B[1 + lag*k : 1 + (lag+1)*k, :]  # source rows x target cols
        A.append(block.T.copy())  # target x source

    return {
        "p": p, "X": X, "Y": Y, "B": B, "resid": E, "sigma": sigma,
        "A": A, "nobs": n, "q": q, "aic": aic, "bic": bic,
        "dates": used_dates,
    }


def select_var_lag(
    panel: pd.DataFrame,
    start_date=None,
    end_date=None,
):
    candidates = []
    fits = {}
    k = len(COMMODITIES)

    for p in range(1, VAR_MAX_LAG + 1):
        X, Y, dates_used = build_var_design(
            panel,
            p,
            start_date,
            end_date,
        )

        n = len(X)
        q = 1 + k * p

        estimable = (
            n >= MIN_VAR_EFFECTIVE_OBS
            and n > q + 5
        )

        row = {
            "lag": p,
            "effective_obs": n,
            "parameters_per_equation": q,
            "estimable": estimable,
        }

        if estimable:
            fit = fit_var_ols(
                panel,
                p,
                start_date,
                end_date,
            )
            row.update({
                "aic": fit["aic"],
                "bic": fit["bic"],
            })
            fits[p] = fit
        else:
            row.update({
                "aic": np.nan,
                "bic": np.nan,
            })

        candidates.append(row)

    table = pd.DataFrame(candidates)
    valid = table[table["estimable"]]

    if valid.empty:
        mask = pd.Series(
            True,
            index=panel.index,
        )
        if start_date is not None:
            mask &= panel.index >= pd.Timestamp(start_date)
        if end_date is not None:
            mask &= panel.index <= pd.Timestamp(end_date)

        finite_counts = panel.loc[mask].notna().sum()

        start_label = (
            pd.Timestamp(start_date).date()
            if start_date is not None
            else panel.index.min().date()
        )
        end_label = (
            pd.Timestamp(end_date).date()
            if end_date is not None
            else panel.index.max().date()
        )

        raise RuntimeError(
            "VAR 유효관측치가 부족합니다.\n\n"
            f"분석구간: {start_label} ~ {end_label}\n"
            f"최소 요구 effective obs: {MIN_VAR_EFFECTIVE_OBS}\n\n"
            "Lag별 실제 유효관측치:\n"
            + table.to_string(index=False)
            + "\n\n시장별 finite EventRisk percentile 개수:\n"
            + finite_counts.to_string()
            + "\n\n"
            "이번 2024-10~2025-05 SOFT-70 자료에서 test VAR 유효관측치가 "
            "30개를 넘는 것이 정상입니다.\n"
            "이 오류가 계속 나오면 04가 예전 02 output을 읽고 있는 것입니다."
        )

    chosen_p = int(
        valid.sort_values(
            ["bic", "lag"]
        ).iloc[0]["lag"]
    )

    return fits[chosen_p], table


# =============================================================================
# 7. Granger predictive test
# =============================================================================
def granger_tests(var_fit):
    X = var_fit["X"]; Y = var_fit["Y"]; p = var_fit["p"]
    n, q_u = X.shape
    rows = []
    for target_idx, target in enumerate(COMMODITIES):
        y = Y[:, target_idx]
        beta_u = np.linalg.lstsq(X, y, rcond=None)[0]
        resid_u = y - X @ beta_u
        sse_u = float(resid_u @ resid_u)
        for source_idx, source in enumerate(COMMODITIES):
            if source == target:
                continue
            remove_cols = []
            for lag in range(p):
                remove_cols.append(1 + lag*len(COMMODITIES) + source_idx)
            keep = [j for j in range(q_u) if j not in remove_cols]
            Xr = X[:, keep]
            beta_r = np.linalg.lstsq(Xr, y, rcond=None)[0]
            resid_r = y - Xr @ beta_r
            sse_r = float(resid_r @ resid_r)
            m = len(remove_cols)
            df2 = n - q_u
            numerator = max((sse_r - sse_u) / m, 0.0)
            denominator = sse_u / df2 if df2 > 0 else np.nan
            F = numerator / denominator if denominator > 0 else np.nan
            pvalue = float(scipy_stats.f.sf(F, m, df2)) if np.isfinite(F) else np.nan
            rows.append({
                "source": source,
                "target": target,
                "lag_order": p,
                "F_stat": F,
                "p_value": pvalue,
                "predictive_at_5pct": bool(np.isfinite(pvalue) and pvalue < 0.05),
            })
    return pd.DataFrame(rows)


# =============================================================================
# 8. Generalized FEVD (row = target, column = shock source)
# =============================================================================
def ma_matrices(A: list[np.ndarray], H: int):
    k = A[0].shape[0]
    p = len(A)
    phi = [np.eye(k)]
    for h in range(1, H):
        cur = np.zeros((k, k))
        for lag in range(1, min(p, h)+1):
            cur += A[lag-1] @ phi[h-lag]
        phi.append(cur)
    return phi


def generalized_fevd(var_fit, H: int):
    A = var_fit["A"]
    sigma = var_fit["sigma"]
    k = len(COMMODITIES)
    phi = ma_matrices(A, H)
    theta = np.zeros((k, k), dtype=float)

    for i in range(k):  # target
        denom = 0.0
        for h in range(H):
            v = phi[h]
            denom += float((v @ sigma @ v.T)[i, i])
        for j in range(k):  # shock source
            sigma_jj = float(sigma[j, j])
            num = 0.0
            for h in range(H):
                v = phi[h]
                impact = float((v @ sigma)[i, j])
                num += impact * impact
            theta[i, j] = (num / sigma_jj) / denom if sigma_jj > 0 and denom > 0 else np.nan

    # generalized FEVD는 row sum이 자동으로 1이 아닐 수 있으므로 Diebold-Yilmaz 방식으로 row-normalize
    row_sums = np.nansum(theta, axis=1, keepdims=True)
    theta_norm = np.divide(theta, row_sums, out=np.full_like(theta, np.nan), where=row_sums > 0)
    return theta_norm


def fevd_table(theta: np.ndarray, label: str):
    m = pd.DataFrame(100 * theta, index=COMMODITIES, columns=COMMODITIES)
    m.index.name = "target"
    long = m.reset_index().melt(id_vars="target", var_name="shock_source", value_name="share_pct")
    long["sample"] = label
    return m, long


def directional_connectedness(theta: np.ndarray, sample_label: str):
    k = len(COMMODITIES)
    offdiag_sum = float(
        sum(theta[i, j] for i in range(k) for j in range(k) if i != j)
    )
    tci = 100.0 * offdiag_sum / k
    rows = []
    for j, commodity in enumerate(COMMODITIES):
        to_others = 100 * sum(theta[i, j] for i in range(k) if i != j)
        from_others = 100 * sum(theta[j, s] for s in range(k) if s != j)
        rows.append({
            "sample": sample_label,
            "commodity": commodity,
            "TO_others_pct": to_others,
            "FROM_others_pct": from_others,
            "NET_sender_pct": to_others - from_others,
            "TCI_pct": tci,
        })
    return pd.DataFrame(rows)


def _fit_var_arrays(series: np.ndarray, p: int):
    n_total, k = series.shape
    X, Y = [], []
    for t in range(p, n_total):
        row = [1.0]
        for lag in range(1, p + 1):
            row.extend(series[t - lag].tolist())
        X.append(row)
        Y.append(series[t].tolist())
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    B = np.linalg.lstsq(X, Y, rcond=None)[0]
    E = Y - X @ B
    sigma = (E.T @ E) / len(E)
    A = []
    for lag in range(p):
        block = B[1 + lag * k: 1 + (lag + 1) * k, :]
        A.append(block.T.copy())
    return {"p": p, "B": B, "resid": E, "sigma": sigma, "A": A}


def bootstrap_connectedness(
    var_fit,
    sample_label: str,
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
):
    """Residual bootstrap CI for NET and TCI. Small-sample uncertainty check."""
    rng = np.random.default_rng(seed)
    p = int(var_fit["p"])
    k = len(COMMODITIES)
    X = np.asarray(var_fit["X"], float)
    Y = np.asarray(var_fit["Y"], float)
    B = np.asarray(var_fit["B"], float)
    resid = np.asarray(var_fit["resid"], float)
    resid = resid - resid.mean(axis=0, keepdims=True)

    # First design row contains lag1, lag2, ...; reverse to oldest->newest.
    lag_blocks = [
        X[0, 1 + lag * k:1 + (lag + 1) * k]
        for lag in range(p)
    ]
    history = list(reversed([b.copy() for b in lag_blocks]))
    intercept = B[0]
    A = var_fit["A"]

    records = []
    for _ in range(int(reps)):
        generated = [h.copy() for h in history]
        sampled = resid[rng.integers(0, len(resid), size=len(Y))]
        try:
            for t in range(len(Y)):
                value = intercept.copy()
                for lag in range(1, p + 1):
                    value = value + A[lag - 1] @ generated[-lag]
                generated.append(value + sampled[t])
            sim = np.asarray(generated, float)
            fit_b = _fit_var_arrays(sim, p)
            theta_b = generalized_fevd(fit_b, FEVD_HORIZON)
            conn_b = directional_connectedness(theta_b, sample_label)
            for row in conn_b.itertuples(index=False):
                records.append({
                    "commodity": row.commodity,
                    "NET_sender_pct": row.NET_sender_pct,
                    "TCI_pct": row.TCI_pct,
                })
        except Exception:
            continue

    if not records:
        return pd.DataFrame(columns=[
            "sample", "commodity", "bootstrap_successful_reps",
            "NET_ci_low", "NET_ci_high", "TCI_ci_low", "TCI_ci_high",
            "bootstrap_status",
        ])

    b = pd.DataFrame(records)
    rows = []
    success_reps = int(len(b) / k)
    for commodity in COMMODITIES:
        g = b[b["commodity"].eq(commodity)]
        rows.append({
            "sample": sample_label,
            "commodity": commodity,
            "bootstrap_successful_reps": success_reps,
            "NET_ci_low": float(g["NET_sender_pct"].quantile(0.025)),
            "NET_ci_high": float(g["NET_sender_pct"].quantile(0.975)),
            "TCI_ci_low": float(g["TCI_pct"].quantile(0.025)),
            "TCI_ci_high": float(g["TCI_pct"].quantile(0.975)),
            "bootstrap_status": (
                "OK" if success_reps >= max(50, reps // 2)
                else "LOW_SUCCESSFUL_REPS"
            ),
        })
    return pd.DataFrame(rows)


# =============================================================================
# 9. Current-window GFEVD
# =============================================================================
def fit_current_window(panel: pd.DataFrame, evaluation_start: pd.Timestamp):
    """Fit the final current network on a strict 30-calendar-day TEST window.

    The window contains only dates from [end-29, end]. VAR(1) is fixed, so the
    first date supplies the lag and the remaining dates can contribute up to 29
    regression observations. The window never expands into TRAIN.
    """
    end = pd.Timestamp(panel.index.max()).normalize()
    evaluation_start = pd.Timestamp(evaluation_start).normalize()
    start = max(
        evaluation_start,
        end - pd.Timedelta(days=CURRENT_WINDOW_DAYS - 1),
    )
    window_panel = panel.loc[(panel.index >= start) & (panel.index <= end)].copy()

    fit = fit_var_ols(window_panel, p=ROLLING_VAR_LAG)
    nobs = int(fit["nobs"])
    if nobs < ROLLING_MIN_EFFECTIVE_OBS:
        raise RuntimeError(
            f"Current {CURRENT_WINDOW_DAYS}D VAR({ROLLING_VAR_LAG}) effective "
            f"obs={nobs}; minimum={ROLLING_MIN_EFFECTIVE_OBS}."
        )

    selection = pd.DataFrame([{
        "lag": ROLLING_VAR_LAG,
        "effective_obs": nobs,
        "parameters_per_equation": 1 + len(COMMODITIES) * ROLLING_VAR_LAG,
        "estimable": True,
        "aic": fit["aic"],
        "bic": fit["bic"],
        "selection_method": "FIXED_VAR1_STRICT_30D",
        "window_start": start,
        "window_end": end,
        "window_days": int((end - start).days + 1),
    }])
    return fit, selection, start, end


# =============================================================================
# 10. OUT-OF-SAMPLE VALIDATION
# =============================================================================

def _future_complete_mean(series: pd.Series, day: pd.Timestamp, horizon: int):
    """
    t+1 ... t+h의 calendar-day 값이 모두 존재하고 finite일 때만 평균을 반환.
    결측을 건너뛰어 서로 다른 날짜를 붙이지 않는다.
    """
    vals = []
    for step in range(1, horizon + 1):
        d = pd.Timestamp(day) + pd.Timedelta(days=step)
        if d not in series.index:
            return np.nan
        v = series.loc[d]
        if isinstance(v, pd.Series):
            return np.nan
        if not np.isfinite(v):
            return np.nan
        vals.append(float(v))
    return float(np.mean(vals))


def oos_ar1_validation(daily: pd.DataFrame):
    eval_start = TEST_START
    rows = []
    for commodity in COMMODITIES:
        g = daily[daily["commodity"].eq(commodity)].sort_values("date").copy()
        if g.empty:
            continue

        c_vals = pd.to_numeric(g.get("decay_intercept_soft70"), errors="coerce").dropna()
        phi_vals = pd.to_numeric(g.get("decay_phi_raw_soft70"), errors="coerce").dropna()
        if c_vals.empty or phi_vals.empty:
            rows.append({
                "commodity": commodity, "n_forecasts": 0,
                "mae": np.nan, "rmse": np.nan, "mean_error": np.nan,
                "oos_r2_vs_train_mean": np.nan,
                "oos_r2_vs_persistence": np.nan,
                "oos_r2_vs_test_mean_expost": np.nan,
                "computation_status": "MISSING_TRAIN_AR_PARAMS",
                "skill_vs_train_mean": "UNAVAILABLE",
                "skill_vs_persistence": "UNAVAILABLE",
            })
            continue

        c, phi = float(c_vals.iloc[0]), float(phi_vals.iloc[0])
        er = g.set_index("date")["daily_event_risk_soft70"].sort_index().astype(float)
        train_mean = float(er.loc[er.index < eval_start].dropna().mean())
        actuals, model_preds, persistence_preds = [], [], []
        for day in er.index:
            if day < eval_start:
                continue
            prev = day - pd.Timedelta(days=1)
            if prev not in er.index:
                continue
            y, x = er.loc[day], er.loc[prev]
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            actuals.append(float(y))
            model_preds.append(c + phi * float(x))
            persistence_preds.append(float(x))

        actuals = np.asarray(actuals, float)
        model_preds = np.asarray(model_preds, float)
        persistence_preds = np.asarray(persistence_preds, float)
        if len(actuals) == 0:
            rows.append({
                "commodity": commodity, "n_forecasts": 0,
                "mae": np.nan, "rmse": np.nan, "mean_error": np.nan,
                "oos_r2_vs_train_mean": np.nan,
                "oos_r2_vs_persistence": np.nan,
                "oos_r2_vs_test_mean_expost": np.nan,
                "computation_status": "NO_VALID_CONSECUTIVE_TEST_PAIRS",
                "skill_vs_train_mean": "UNAVAILABLE",
                "skill_vs_persistence": "UNAVAILABLE",
            })
            continue

        err = actuals - model_preds
        sse = float(np.sum(err ** 2))
        sse_train_mean = float(np.sum((actuals - train_mean) ** 2))
        sse_persistence = float(np.sum((actuals - persistence_preds) ** 2))
        test_mean = float(np.mean(actuals))
        sse_test_mean = float(np.sum((actuals - test_mean) ** 2))

        r2_train = np.nan if sse_train_mean <= 0 else 1 - sse / sse_train_mean
        r2_persist = np.nan if sse_persistence <= 0 else 1 - sse / sse_persistence
        r2_test = np.nan if sse_test_mean <= 0 else 1 - sse / sse_test_mean

        rows.append({
            "commodity": commodity,
            "n_forecasts": int(len(actuals)),
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mean_error": float(np.mean(err)),
            "train_mean_benchmark": train_mean,
            "oos_r2_vs_train_mean": r2_train,
            "oos_r2_vs_persistence": r2_persist,
            "oos_r2_vs_test_mean_expost": r2_test,
            "computation_status": "OK",
            "skill_vs_train_mean": (
                "POSITIVE" if np.isfinite(r2_train) and r2_train > 0 else "NEGATIVE"
            ),
            "skill_vs_persistence": (
                "POSITIVE" if np.isfinite(r2_persist) and r2_persist > 0 else "NEGATIVE"
            ),
        })
    return pd.DataFrame(rows)


def oos_dgri_predictive_validation(daily: pd.DataFrame):
    rows = []
    for commodity in COMMODITIES:
        g = daily[daily["commodity"].eq(commodity)].sort_values("date").copy()
        if g.empty:
            continue
        er = g.set_index("date")["daily_event_risk_soft70"].astype(float).sort_index()
        dgri_pct = g.set_index("date")["dgri_percentile_train_soft70"].astype(float).sort_index()

        for threshold in DGRI_THRESHOLDS:
            for h in OOS_HORIZONS:
                obs = []
                for day in dgri_pct.index:
                    if day < TEST_START:
                        continue
                    x = dgri_pct.loc[day]
                    if not np.isfinite(x):
                        continue
                    future = _future_complete_mean(er, day, h)
                    if np.isfinite(future):
                        obs.append((float(x), float(future)))

                if not obs:
                    rows.append({
                        "commodity": commodity, "threshold": threshold,
                        "horizon_days": h, "n_obs": 0, "n_high_dgri": 0,
                        "n_other": 0, "high_future_eventrisk_mean": np.nan,
                        "other_future_eventrisk_mean": np.nan,
                        "raw_high_minus_other": np.nan,
                        "validated_high_minus_other": np.nan,
                        "spearman_rho": np.nan, "spearman_pvalue": np.nan,
                        "group_comparison_status": "NO_COMPLETE_TEST_HORIZONS",
                    })
                    continue

                arr = np.asarray(obs, float)
                x, y = arr[:, 0], arr[:, 1]
                high = x >= threshold
                n_high, n_other = int(high.sum()), int((~high).sum())
                high_mean = float(np.mean(y[high])) if n_high else np.nan
                other_mean = float(np.mean(y[~high])) if n_other else np.nan
                raw_diff = (
                    high_mean - other_mean
                    if np.isfinite(high_mean) and np.isfinite(other_mean)
                    else np.nan
                )
                enough = n_high >= MIN_HIGH_DGRI_DAYS and n_other >= MIN_OTHER_DAYS
                rho, pval = (np.nan, np.nan)
                if len(x) >= 10 and np.nanstd(x) > 0 and np.nanstd(y) > 0:
                    rho, pval = scipy_stats.spearmanr(x, y)
                rows.append({
                    "commodity": commodity, "threshold": threshold,
                    "horizon_days": h, "n_obs": int(len(x)),
                    "n_high_dgri": n_high, "n_other": n_other,
                    "high_future_eventrisk_mean": high_mean,
                    "other_future_eventrisk_mean": other_mean,
                    "raw_high_minus_other": raw_diff,
                    "validated_high_minus_other": raw_diff if enough else np.nan,
                    "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                    "spearman_pvalue": float(pval) if np.isfinite(pval) else np.nan,
                    "group_comparison_status": (
                        "OK" if enough else "INSUFFICIENT_GROUP_SAMPLE"
                    ),
                })
    return pd.DataFrame(rows)


def oos_leadership_validation(daily: pd.DataFrame):
    dgri = daily.pivot(
        index="date", columns="commodity", values="dgri_percentile_train_soft70"
    ).reindex(columns=COMMODITIES).sort_index()
    er_pct = daily.pivot(
        index="date", columns="commodity", values="eventrisk_percentile_train_soft70"
    ).reindex(columns=COMMODITIES).sort_index()

    detail_rows = []
    for threshold in DGRI_THRESHOLDS:
        for day in dgri.index:
            if day < TEST_START:
                continue
            current_dgri = dgri.loc[day]
            if current_dgri.isna().any():
                continue
            leader = str(current_dgri.idxmax())
            leader_pct = float(current_dgri.max())
            if leader_pct < threshold:
                continue
            others = [c for c in COMMODITIES if c != leader]
            cur_other = er_pct.loc[day, others]
            if cur_other.isna().any():
                continue
            current_other_mean = float(cur_other.mean())
            for h in OOS_HORIZONS:
                future_vals, valid = [], True
                for step in range(1, h + 1):
                    future_day = day + pd.Timedelta(days=step)
                    if future_day not in er_pct.index:
                        valid = False; break
                    vals = er_pct.loc[future_day, others]
                    if vals.isna().any():
                        valid = False; break
                    future_vals.extend([float(v) for v in vals.values])
                if valid and future_vals:
                    future_mean = float(np.mean(future_vals))
                    detail_rows.append({
                        "date": day, "threshold": threshold, "leader": leader,
                        "leader_dgri_percentile": leader_pct,
                        "horizon_days": h,
                        "current_other_eventrisk_pct_mean": current_other_mean,
                        "future_other_eventrisk_pct_mean": future_mean,
                        "future_minus_current_pp": future_mean - current_other_mean,
                    })

    details = pd.DataFrame(detail_rows)
    if details.empty:
        return details, pd.DataFrame(columns=[
            "threshold", "leader", "horizon_days", "n_leader_days",
            "mean_future_minus_current_pp", "median_future_minus_current_pp",
            "positive_change_share", "validation_status",
        ])

    summary = (
        details.groupby(["threshold", "leader", "horizon_days"], as_index=False)
        .agg(
            n_leader_days=("date", "nunique"),
            mean_future_minus_current_pp=("future_minus_current_pp", "mean"),
            median_future_minus_current_pp=("future_minus_current_pp", "median"),
            positive_change_share=(
                "future_minus_current_pp", lambda s: float((s > 0).mean())
            ),
        )
    )
    summary["validation_status"] = np.where(
        summary["n_leader_days"] >= MIN_LEADER_DAYS,
        "DESCRIPTIVE_SAMPLE_OK",
        "INSUFFICIENT_LEADER_DAYS",
    )
    return details, summary.sort_values(
        ["threshold", "horizon_days", "leader"]
    ).reset_index(drop=True)


# =============================================================================
# 11. Current Risk State: 최신 공통 DGRI 날짜
# =============================================================================
def current_risk_state(daily: pd.DataFrame):
    p = daily.pivot(
        index="date", columns="commodity", values="dgri_percentile_train_soft70"
    ).reindex(columns=COMMODITIES)
    common = p.dropna(how="any")
    if common.empty:
        raise RuntimeError("Oil/Gas/Power 모두 DGRI가 존재하는 공통 날짜가 없습니다.")
    current_date = common.index.max()
    vals = common.loc[current_date]

    status_map = (
        daily[daily["date"].eq(current_date)]
        .set_index("commodity")["dgri_status_soft70"]
        .to_dict()
        if "dgri_status_soft70" in daily.columns else {}
    )
    max_pct = float(vals.max())
    system_state = "LOW" if max_pct < 20 else ("NORMAL" if max_pct < 80 else "ELEVATED")

    rows = []
    for commodity in COMMODITIES:
        raw_status = str(status_map.get(commodity, "UNKNOWN"))
        update_type = (
            "OBSERVED_UPDATE" if "OBSERVED" in raw_status
            else ("DECAY_ONLY_ESTIMATE" if "DECAY_ONLY" in raw_status else "UNKNOWN")
        )
        rows.append({
            "date": current_date,
            "commodity": commodity,
            "dgri_percentile": float(vals[commodity]),
            "dgri_update_type": update_type,
            "system_risk_state": system_state,
            "system_max_dgri_percentile": max_pct,
        })
    table = pd.DataFrame(rows).sort_values(
        "dgri_percentile", ascending=False
    ).reset_index(drop=True)
    table["risk_rank"] = np.arange(1, len(table) + 1)
    table["current_risk_leader"] = table["risk_rank"].eq(1)
    return table


# =============================================================================
# 12. Graphs
# =============================================================================
def save_graphs(shock_summary, transmission_current, current_state):
    if not shock_summary.empty:
        d = shock_summary.sort_values("shock_days", ascending=False)
        fig, ax = plt.subplots(figsize=(7,4))
        ax.bar(d["commodity"], d["shock_days"])
        ax.set_title("Extreme EventRisk Shock Days – TEST")
        ax.set_ylabel("Shock days (>= 95th percentile)")
        ax.set_xlabel("Commodity")
        fig.tight_layout()
        fig.savefig(GRAPH_DIR / "shock_occurrence_by_commodity.png", dpi=160)
        plt.close(fig)

    if not transmission_current.empty:
        d = transmission_current.sort_values("NET_sender_pct", ascending=False)
        fig, ax = plt.subplots(figsize=(7,4))
        ax.bar(d["commodity"], d["NET_sender_pct"])
        ax.axhline(0, linewidth=1)
        ax.set_title("Current-Window Net Risk Transmission – GFEVD")
        ax.set_ylabel("TO − FROM (percentage points)")
        fig.tight_layout()
        fig.savefig(GRAPH_DIR / "current_net_transmission.png", dpi=160)
        plt.close(fig)

    if not current_state.empty:
        d = current_state.sort_values("dgri_percentile", ascending=False)
        fig, ax = plt.subplots(figsize=(7,4))
        ax.bar(d["commodity"], d["dgri_percentile"])
        ax.set_ylim(0,100)
        ax.set_title("Current Accumulated Risk State – DGRI Percentile")
        ax.set_ylabel("Calibration-based DGRI percentile")
        fig.tight_layout()
        fig.savefig(GRAPH_DIR / "current_dgri_risk_state.png", dpi=160)
        plt.close(fig)


# =============================================================================
# 13. Main
# =============================================================================
def _save_csv(df: pd.DataFrame, filename: str, index: bool = False):
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=index, encoding="utf-8-sig")
    return path


def _run_var_spec(daily: pd.DataFrame, value_col: str, label: str):
    panel = make_eventrisk_panel(daily, value_col)
    eval_start = TEST_START
    full_fit = None
    cur_fit = None
    outputs = {
        "panel": panel,
        "full_fit": None, "cur_fit": None,
        "lag_sel": pd.DataFrame(), "granger": pd.DataFrame(),
        "full_matrix": pd.DataFrame(), "full_long": pd.DataFrame(),
        "full_conn": pd.DataFrame(), "full_boot": pd.DataFrame(),
        "cur_lag_sel": pd.DataFrame(), "cur_matrix": pd.DataFrame(),
        "cur_long": pd.DataFrame(), "cur_conn": pd.DataFrame(),
        "cur_boot": pd.DataFrame(), "cur_start": eval_start,
        "cur_end": panel.index.max(),
    }
    try:
        full_fit, lag_sel = select_var_lag(
            panel, start_date=eval_start, end_date=panel.index.max()
        )
        theta = generalized_fevd(full_fit, FEVD_HORIZON)
        matrix, long = fevd_table(theta, f"{label}_test_full")
        conn = directional_connectedness(theta, f"{label}_test_full")
        boot = bootstrap_connectedness(full_fit, f"{label}_test_full")
        outputs.update({
            "full_fit": full_fit, "lag_sel": lag_sel,
            "granger": granger_tests(full_fit), "full_matrix": matrix,
            "full_long": long, "full_conn": conn, "full_boot": boot,
        })
    except RuntimeError as exc:
        print(f"[WARN] {label} full VAR/GFEVD 생략: {exc}")
        return outputs

    try:
        cur_fit, cur_lag_sel, cur_start, cur_end = fit_current_window(
            panel, evaluation_start=eval_start
        )
        theta = generalized_fevd(cur_fit, FEVD_HORIZON)
        matrix, long = fevd_table(theta, f"{label}_test_current")
        conn = directional_connectedness(theta, f"{label}_test_current")
        boot = bootstrap_connectedness(
            cur_fit, f"{label}_test_current", seed=BOOTSTRAP_SEED + 1
        )
        outputs.update({
            "cur_fit": cur_fit, "cur_lag_sel": cur_lag_sel,
            "cur_start": cur_start, "cur_end": cur_end,
            "cur_matrix": matrix, "cur_long": long,
            "cur_conn": conn, "cur_boot": boot,
        })
    except RuntimeError as exc:
        print(f"[WARN] {label} current-window GFEVD 생략: {exc}")
    return outputs



# =============================================================================
# 14. Daily rolling transmission: strict 30-day window, re-estimated every day
# =============================================================================
def _rolling_period_type(window_start: pd.Timestamp, window_end: pd.Timestamp) -> str:
    """Classify a rolling window by whether it lies in Train, Test, or both."""
    train_end = pd.Timestamp(CALIBRATION_END).normalize()
    test_start = pd.Timestamp(TEST_START).normalize()
    if window_end <= train_end:
        return "TRAIN"
    if window_start >= test_start:
        return "TEST"
    return "TRANSITION"


def _rolling_asof_dates(panel: pd.DataFrame) -> list[pd.Timestamp]:
    """Generate every daily as-of date after the strict 30-day warm-up."""
    if panel.empty:
        return []
    first_date = pd.Timestamp(panel.index.min()).normalize()
    final_date = pd.Timestamp(panel.index.max()).normalize()
    first_asof = first_date + pd.Timedelta(days=ROLLING_WINDOW_DAYS - 1)
    if first_asof > final_date:
        return []
    return [
        pd.Timestamp(d).normalize()
        for d in pd.date_range(first_asof, final_date, freq="D")
    ]


def _connectedness_maps(conn: pd.DataFrame) -> tuple[dict, dict, dict]:
    if conn.empty:
        return {}, {}, {}
    net = conn.set_index("commodity")["NET_sender_pct"].to_dict()
    to_map = conn.set_index("commodity")["TO_others_pct"].to_dict()
    from_map = conn.set_index("commodity")["FROM_others_pct"].to_dict()
    return net, to_map, from_map


def _rolling_should_bootstrap(as_of_date: pd.Timestamp, final_date: pd.Timestamp) -> bool:
    """Bootstrap only at month-end checkpoints and the final study date."""
    if not ROLLING_BOOTSTRAP_MONTH_END_ONLY:
        return True
    day = pd.Timestamp(as_of_date).normalize()
    return bool(day.is_month_end or day == pd.Timestamp(final_date).normalize())


def rolling_transmission_analysis(
    daily: pd.DataFrame,
    value_col: str = MAIN_VAR_COLUMN,
):
    """Estimate a strict 30-day VAR(1)/GFEVD network for every calendar date.

    For an as-of date t, the data slice is [t-29, t]. Because VAR(1) is fit on
    that sliced panel, only lags that also lie inside the 30-day window are used,
    yielding at most 29 effective observations. No prior-to-window observation
    leaks into the estimate.
    """
    panel = make_eventrisk_panel(daily, value_col)
    panel = panel.loc[
        (panel.index >= STUDY_START) & (panel.index <= STUDY_END)
    ].copy()

    train_fit = None
    train_theta = None
    train_matrix = pd.DataFrame()
    train_conn = pd.DataFrame()
    train_lag_sel = pd.DataFrame()
    try:
        train_fit, train_lag_sel = select_var_lag(
            panel,
            start_date=STUDY_START,
            end_date=pd.Timestamp(CALIBRATION_END),
        )
        train_theta = generalized_fevd(train_fit, FEVD_HORIZON)
        train_matrix, _ = fevd_table(train_theta, "train_full_benchmark")
        train_conn = directional_connectedness(
            train_theta, "train_full_benchmark"
        )
        train_conn["lag"] = int(train_fit["p"])
        train_conn["effective_obs"] = int(train_fit["nobs"])
        train_conn["window_start"] = STUDY_START
        train_conn["window_end"] = pd.Timestamp(CALIBRATION_END)
    except RuntimeError as exc:
        print(f"[WARN] Train benchmark network unavailable: {exc}")

    train_net, _, _ = _connectedness_maps(train_conn)
    train_leader = (
        max(train_net, key=train_net.get) if train_net else "UNAVAILABLE"
    )
    train_tci = (
        float(train_conn["TCI_pct"].iloc[0]) if not train_conn.empty else np.nan
    )

    history_rows = []
    matrix_rows = []
    lag_rows = []
    boot_rows = []
    stability_rows = []
    final_date = pd.Timestamp(panel.index.max()).normalize()

    for update_index, as_of_date in enumerate(_rolling_asof_dates(panel), start=1):
        window_end = pd.Timestamp(as_of_date).normalize()
        window_start = window_end - pd.Timedelta(days=ROLLING_WINDOW_DAYS - 1)
        period_type = _rolling_period_type(window_start, window_end)
        window_panel = panel.loc[
            (panel.index >= window_start) & (panel.index <= window_end)
        ].copy()

        base_row = {
            "update_index": update_index,
            "as_of_date": window_end,
            "signal_apply_date": window_end + pd.Timedelta(days=1),
            "window_start": window_start,
            "window_end": window_end,
            "window_days": ROLLING_WINDOW_DAYS,
            "update_interval_days": ROLLING_STEP_DAYS,
            "period_type": period_type,
            "var_input_column": value_col,
            "rolling_model": f"FIXED_VAR({ROLLING_VAR_LAG})",
        }

        try:
            fit = fit_var_ols(window_panel, p=ROLLING_VAR_LAG)
            nobs = int(fit["nobs"])
            q = int(fit["q"])
            estimable = (
                nobs >= ROLLING_MIN_EFFECTIVE_OBS
                and nobs > q + 2
            )
            lag_rows.append({
                "as_of_date": window_end,
                "window_start": window_start,
                "window_end": window_end,
                "period_type": period_type,
                "lag": ROLLING_VAR_LAG,
                "effective_obs": nobs,
                "parameters_per_equation": q,
                "estimable": estimable,
                "aic": fit["aic"],
                "bic": fit["bic"],
                "selection_method": "FIXED_VAR1_DAILY_30D",
            })
            if not estimable:
                raise RuntimeError(
                    f"Daily 30D VAR(1) effective obs={nobs}; "
                    f"minimum={ROLLING_MIN_EFFECTIVE_OBS}."
                )

            theta = generalized_fevd(fit, FEVD_HORIZON)
            conn = directional_connectedness(
                theta, f"rolling_daily_{window_end:%Y%m%d}"
            )
            net, to_map, from_map = _connectedness_maps(conn)
            leader = max(net, key=net.get)
            tci = float(conn["TCI_pct"].iloc[0])

            do_bootstrap = _rolling_should_bootstrap(window_end, final_date)
            boot = pd.DataFrame()
            if do_bootstrap and ROLLING_BOOTSTRAP_REPS > 0:
                boot = bootstrap_connectedness(
                    fit,
                    f"rolling_daily_{window_end:%Y%m%d}",
                    reps=ROLLING_BOOTSTRAP_REPS,
                    seed=BOOTSTRAP_SEED + 10000 + update_index,
                )
                if not boot.empty:
                    boot = boot.copy()
                    boot.insert(0, "as_of_date", window_end)
                    boot.insert(1, "window_start", window_start)
                    boot.insert(2, "window_end", window_end)
                    boot.insert(3, "period_type", period_type)
                    boot_rows.extend(boot.to_dict("records"))

            leader_ci_low = np.nan
            leader_ci_high = np.nan
            tci_ci_low = np.nan
            tci_ci_high = np.nan
            transmission_confidence = "POINT_ESTIMATE_ONLY"
            if not boot.empty:
                b_leader = boot[boot["commodity"].eq(leader)]
                if not b_leader.empty:
                    leader_ci_low = float(b_leader.iloc[0]["NET_ci_low"])
                    leader_ci_high = float(b_leader.iloc[0]["NET_ci_high"])
                    tci_ci_low = float(b_leader.iloc[0]["TCI_ci_low"])
                    tci_ci_high = float(b_leader.iloc[0]["TCI_ci_high"])
                    transmission_confidence = (
                        "POSITIVE_NET_95CI" if leader_ci_low > 0
                        else "UNCERTAIN_CI_INCLUDES_ZERO"
                    )

            history_rows.append({
                **base_row,
                "status": "OK",
                "selected_lag": ROLLING_VAR_LAG,
                "effective_obs": nobs,
                "effective_start": (
                    min(fit["dates"]) if fit["dates"] else pd.NaT
                ),
                "effective_end": (
                    max(fit["dates"]) if fit["dates"] else pd.NaT
                ),
                "oil_to_others_pct": to_map.get("Oil", np.nan),
                "gas_to_others_pct": to_map.get("Gas", np.nan),
                "power_to_others_pct": to_map.get("Power", np.nan),
                "oil_from_others_pct": from_map.get("Oil", np.nan),
                "gas_from_others_pct": from_map.get("Gas", np.nan),
                "power_from_others_pct": from_map.get("Power", np.nan),
                "oil_net_sender_pct": net.get("Oil", np.nan),
                "gas_net_sender_pct": net.get("Gas", np.nan),
                "power_net_sender_pct": net.get("Power", np.nan),
                "transmission_leader": leader,
                "leader_net_sender_pct": net.get(leader, np.nan),
                "leader_net_ci_low": leader_ci_low,
                "leader_net_ci_high": leader_ci_high,
                "transmission_confidence": transmission_confidence,
                "TCI_pct": tci,
                "TCI_ci_low": tci_ci_low,
                "TCI_ci_high": tci_ci_high,
                "bootstrap_computed": bool(do_bootstrap and not boot.empty),
            })

            for i, target in enumerate(COMMODITIES):
                for j, source in enumerate(COMMODITIES):
                    matrix_rows.append({
                        "as_of_date": window_end,
                        "window_start": window_start,
                        "window_end": window_end,
                        "period_type": period_type,
                        "target": target,
                        "shock_source": source,
                        "share_pct": float(100.0 * theta[i, j]),
                    })

            if train_theta is not None and train_net:
                sign_matches = []
                for commodity in COMMODITIES:
                    train_value = float(train_net.get(commodity, np.nan))
                    rolling_value = float(net.get(commodity, np.nan))
                    if np.isfinite(train_value) and np.isfinite(rolling_value):
                        sign_matches.append(
                            int(np.sign(train_value) == np.sign(rolling_value))
                        )

                stability_rows.append({
                    "as_of_date": window_end,
                    "window_start": window_start,
                    "window_end": window_end,
                    "period_type": period_type,
                    "train_transmission_leader": train_leader,
                    "rolling_transmission_leader": leader,
                    "leader_match_to_train": bool(leader == train_leader),
                    "train_TCI_pct": train_tci,
                    "rolling_TCI_pct": tci,
                    "delta_TCI_pct": tci - train_tci,
                    "gfevd_frobenius_distance": float(
                        np.linalg.norm(theta - train_theta, ord="fro")
                    ),
                    "net_sign_match_count": int(sum(sign_matches)),
                    "net_sign_match_share": (
                        float(np.mean(sign_matches)) if sign_matches else np.nan
                    ),
                    "train_oil_net_pct": train_net.get("Oil", np.nan),
                    "rolling_oil_net_pct": net.get("Oil", np.nan),
                    "train_gas_net_pct": train_net.get("Gas", np.nan),
                    "rolling_gas_net_pct": net.get("Gas", np.nan),
                    "train_power_net_pct": train_net.get("Power", np.nan),
                    "rolling_power_net_pct": net.get("Power", np.nan),
                })

        except RuntimeError as exc:
            if not any(
                row.get("as_of_date") == window_end for row in lag_rows
            ):
                X, _, _ = build_var_design(
                    window_panel, ROLLING_VAR_LAG
                )
                lag_rows.append({
                    "as_of_date": window_end,
                    "window_start": window_start,
                    "window_end": window_end,
                    "period_type": period_type,
                    "lag": ROLLING_VAR_LAG,
                    "effective_obs": int(len(X)),
                    "parameters_per_equation": 1 + len(COMMODITIES) * ROLLING_VAR_LAG,
                    "estimable": False,
                    "aic": np.nan,
                    "bic": np.nan,
                    "selection_method": "FIXED_VAR1_DAILY_30D",
                })
            history_rows.append({
                **base_row,
                "status": "INSUFFICIENT_VAR_OBS",
                "error_message": str(exc).split("\n")[0],
                "selected_lag": ROLLING_VAR_LAG,
                "effective_obs": 0,
                "transmission_leader": "UNAVAILABLE",
                "transmission_confidence": "UNAVAILABLE",
                "TCI_pct": np.nan,
                "bootstrap_computed": False,
            })

    history = pd.DataFrame(history_rows).sort_values(
        "as_of_date"
    ).reset_index(drop=True)
    matrices = pd.DataFrame(matrix_rows)
    lag_diagnostics = pd.DataFrame(lag_rows)
    rolling_bootstrap = pd.DataFrame(boot_rows)
    stability = pd.DataFrame(stability_rows)

    return {
        "history": history,
        "matrices": matrices,
        "lag_diagnostics": lag_diagnostics,
        "bootstrap": rolling_bootstrap,
        "stability": stability,
        "train_fit": train_fit,
        "train_lag_selection": train_lag_sel,
        "train_matrix": train_matrix,
        "train_connectedness": train_conn,
        "train_leader": train_leader,
        "train_tci": train_tci,
    }


def build_daily_risk_and_transmission_state(
    daily: pd.DataFrame,
    rolling_history: pd.DataFrame,
) -> pd.DataFrame:
    """Combine daily risk state with the same-day 30-day rolling network."""
    dates = pd.DataFrame({
        "date": sorted(pd.to_datetime(daily["date"]).dropna().unique())
    })
    dates["date"] = pd.to_datetime(dates["date"]).dt.normalize()
    dates["sample_period"] = np.where(
        dates["date"].le(pd.Timestamp(CALIBRATION_END)), "TRAIN", "TEST"
    )

    def add_wide(value_col: str, suffix: str):
        nonlocal dates
        if value_col not in daily.columns:
            return
        p = daily.pivot(
            index="date", columns="commodity", values=value_col
        ).reindex(columns=COMMODITIES)
        p.columns = [f"{c.lower()}_{suffix}" for c in p.columns]
        dates = dates.merge(p.reset_index(), on="date", how="left")

    add_wide("daily_event_risk_soft70", "eventrisk_soft70")
    add_wide("eventrisk_percentile_train_soft70", "eventrisk_pct_train")
    add_wide("dgri_soft70", "dgri")
    add_wide("dgri_percentile_train_soft70", "dgri_pct_train")

    dgri_pct_cols = [f"{c.lower()}_dgri_pct_train" for c in COMMODITIES]
    if all(c in dates.columns for c in dgri_pct_cols):
        labels = pd.Series(COMMODITIES, index=dgri_pct_cols)
        dates["risk_leader"] = dates[dgri_pct_cols].idxmax(axis=1).map(labels)
        dates["risk_leader_dgri_pct"] = dates[dgri_pct_cols].max(axis=1)
        dates.loc[dates[dgri_pct_cols].isna().all(axis=1), "risk_leader"] = np.nan

    er_pct_cols = [f"{c.lower()}_eventrisk_pct_train" for c in COMMODITIES]
    for commodity, col in zip(COMMODITIES, er_pct_cols):
        if col in dates.columns:
            dates[f"{commodity.lower()}_shock95"] = dates[col].ge(
                MAIN_SHOCK_PERCENTILE
            )

    if rolling_history.empty:
        dates["transmission_status"] = "WARMUP_NO_ROLLING_ESTIMATE"
        return dates

    keep = [
        "as_of_date", "window_start", "window_end", "period_type", "status",
        "selected_lag", "effective_obs", "oil_net_sender_pct",
        "gas_net_sender_pct", "power_net_sender_pct", "transmission_leader",
        "leader_net_sender_pct", "leader_net_ci_low", "leader_net_ci_high",
        "transmission_confidence", "TCI_pct", "TCI_ci_low", "TCI_ci_high",
    ]
    available = [c for c in keep if c in rolling_history.columns]
    r = rolling_history[available].copy().sort_values("as_of_date")
    r = r.rename(columns={
        "as_of_date": "transmission_asof_date",
        "window_start": "transmission_window_start",
        "window_end": "transmission_window_end",
        "period_type": "transmission_window_period",
        "status": "transmission_status",
    })

    # A daily rolling estimate exists for every date after the 30-day warm-up.
    # Use an exact-date merge so a failed date is never silently filled with the
    # previous day's network.
    dates = dates.merge(
        r,
        left_on="date",
        right_on="transmission_asof_date",
        how="left",
    )
    dates["transmission_status"] = dates["transmission_status"].fillna(
        "WARMUP_NO_30D_WINDOW"
    )
    dates["days_since_transmission_update"] = np.where(
        dates["transmission_asof_date"].notna(), 0.0, np.nan
    )
    return dates


def rolling_train_test_comparison(
    rolling_history: pd.DataFrame,
    stability: pd.DataFrame,
    train_connectedness: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize daily 30-day networks by Train/Transition/Test period."""
    rows = []
    ok = rolling_history[
        rolling_history.get("status", pd.Series(index=rolling_history.index)).eq("OK")
    ].copy() if not rolling_history.empty else pd.DataFrame()

    if not train_connectedness.empty:
        net, _, _ = _connectedness_maps(train_connectedness)
        leader = max(net, key=net.get)
        rows.append({
            "period_type": "TRAIN_FULL_BENCHMARK",
            "n_updates": 1,
            "mean_TCI_pct": float(train_connectedness["TCI_pct"].iloc[0]),
            "oil_mean_net_pct": net.get("Oil", np.nan),
            "gas_mean_net_pct": net.get("Gas", np.nan),
            "power_mean_net_pct": net.get("Power", np.nan),
            "oil_leader_share": float(leader == "Oil"),
            "gas_leader_share": float(leader == "Gas"),
            "power_leader_share": float(leader == "Power"),
            "train_leader_match_share": 1.0,
            "mean_gfevd_distance_from_train": 0.0,
        })

    for period_type in ["TRAIN", "TRANSITION", "TEST"]:
        g = ok[ok["period_type"].eq(period_type)].copy()
        if g.empty:
            continue
        s = stability[stability["period_type"].eq(period_type)] if not stability.empty else pd.DataFrame()
        rows.append({
            "period_type": period_type,
            "n_updates": int(len(g)),
            "mean_TCI_pct": float(g["TCI_pct"].mean()),
            "oil_mean_net_pct": float(g["oil_net_sender_pct"].mean()),
            "gas_mean_net_pct": float(g["gas_net_sender_pct"].mean()),
            "power_mean_net_pct": float(g["power_net_sender_pct"].mean()),
            "oil_leader_share": float(g["transmission_leader"].eq("Oil").mean()),
            "gas_leader_share": float(g["transmission_leader"].eq("Gas").mean()),
            "power_leader_share": float(g["transmission_leader"].eq("Power").mean()),
            "train_leader_match_share": (
                float(s["leader_match_to_train"].mean()) if not s.empty else np.nan
            ),
            "mean_gfevd_distance_from_train": (
                float(s["gfevd_frobenius_distance"].mean()) if not s.empty else np.nan
            ),
        })
    return pd.DataFrame(rows)


def save_rolling_graphs(
    rolling_history: pd.DataFrame,
    daily_state: pd.DataFrame,
    stability: pd.DataFrame,
):
    ok = rolling_history[
        rolling_history.get("status", pd.Series(index=rolling_history.index)).eq("OK")
    ].copy() if not rolling_history.empty else pd.DataFrame()

    if not ok.empty:
        fig, ax = plt.subplots(figsize=(11, 4.8))
        for commodity in COMMODITIES:
            ax.plot(
                ok["as_of_date"],
                ok[f"{commodity.lower()}_net_sender_pct"],
                linewidth=1.3,
                label=commodity,
            )
        ax.axhline(0, linewidth=1)
        ax.axvline(TEST_START, linestyle="--", linewidth=1)
        ax.set_title(
            f"Daily Rolling NET Transmission ({ROLLING_WINDOW_DAYS}D window)"
        )
        ax.set_ylabel("NET = TO - FROM (percentage points)")
        ax.set_xlabel("As-of date")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(GRAPH_DIR / "rolling_30d_daily_net_transmission.png", dpi=170)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(ok["as_of_date"], ok["TCI_pct"], linewidth=1.4)
        ax.axvline(TEST_START, linestyle="--", linewidth=1)
        ax.set_title(
            f"Daily Rolling Total Connectedness ({ROLLING_WINDOW_DAYS}D window)"
        )
        ax.set_ylabel("TCI (%)")
        ax.set_xlabel("As-of date")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(GRAPH_DIR / "rolling_30d_daily_tci.png", dpi=170)
        plt.close(fig)

        leader_code = {"Oil": 0, "Gas": 1, "Power": 2}
        fig, ax = plt.subplots(figsize=(11, 4.2))
        ax.step(
            ok["as_of_date"],
            ok["transmission_leader"].map(leader_code),
            where="post",
            linewidth=1.2,
            label="Transmission leader",
        )
        ax.axvline(TEST_START, linestyle="--", linewidth=1)
        ax.set_yticks([0, 1, 2], ["Oil", "Gas", "Power"])
        ax.set_title("Daily 30-Day Relative Transmission Leader")
        ax.set_xlabel("As-of date")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(GRAPH_DIR / "rolling_30d_daily_transmission_leader.png", dpi=170)
        plt.close(fig)

    if not stability.empty:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(
            stability["as_of_date"],
            stability["gfevd_frobenius_distance"],
            linewidth=1.4,
        )
        ax.axvline(TEST_START, linestyle="--", linewidth=1)
        ax.set_title("Distance from Full-Train GFEVD Structure")
        ax.set_ylabel("Frobenius distance")
        ax.set_xlabel("As-of date")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(GRAPH_DIR / "rolling_network_distance_from_train.png", dpi=170)
        plt.close(fig)

    if not daily_state.empty:
        er_cols = [f"{c.lower()}_eventrisk_pct_train" for c in COMMODITIES]
        if all(c in daily_state.columns for c in er_cols):
            fig, ax = plt.subplots(figsize=(12, 5))
            for commodity, col in zip(COMMODITIES, er_cols):
                ax.plot(daily_state["date"], daily_state[col], label=commodity)
            ax.axvline(TEST_START, linestyle="--", linewidth=1)
            ax.axhline(MAIN_SHOCK_PERCENTILE, linestyle=":", linewidth=1)
            ax.set_ylim(0, 100)
            ax.set_title("Daily EventRisk Percentile: Full 8-Month Path")
            ax.set_ylabel("Train-calibrated percentile")
            ax.set_xlabel("Date")
            ax.legend()
            ax.grid(alpha=0.2)
            fig.tight_layout()
            fig.savefig(GRAPH_DIR / "daily_eventrisk_full_8m.png", dpi=170)
            plt.close(fig)

        dgri_cols = [f"{c.lower()}_dgri_pct_train" for c in COMMODITIES]
        if all(c in daily_state.columns for c in dgri_cols):
            fig, ax = plt.subplots(figsize=(12, 5))
            for commodity, col in zip(COMMODITIES, dgri_cols):
                ax.plot(daily_state["date"], daily_state[col], label=commodity)
            ax.axvline(TEST_START, linestyle="--", linewidth=1)
            ax.set_ylim(0, 100)
            ax.set_title("Daily DGRI Percentile: Full 8-Month Path")
            ax.set_ylabel("Train-calibrated percentile")
            ax.set_xlabel("Date")
            ax.legend()
            ax.grid(alpha=0.2)
            fig.tight_layout()
            fig.savefig(GRAPH_DIR / "daily_dgri_full_8m.png", dpi=170)
            plt.close(fig)

        if {"risk_leader", "transmission_leader"}.issubset(daily_state.columns):
            code = {"Oil": 0, "Gas": 1, "Power": 2}
            fig, ax = plt.subplots(figsize=(12, 4.5))
            ax.scatter(
                daily_state["date"],
                daily_state["risk_leader"].map(code),
                s=10,
                label="Daily DGRI risk leader",
            )
            ax.step(
                daily_state["date"],
                daily_state["transmission_leader"].map(code),
                where="post",
                label="Same-day 30D transmission leader",
            )
            ax.axvline(TEST_START, linestyle="--", linewidth=1)
            ax.set_yticks([0, 1, 2], ["Oil", "Gas", "Power"])
            ax.set_title("Daily Risk Leader vs Daily 30-Day Transmission Leader")
            ax.set_xlabel("Date")
            ax.legend()
            ax.grid(alpha=0.2)
            fig.tight_layout()
            fig.savefig(GRAPH_DIR / "daily_risk_vs_transmission_leader.png", dpi=170)
            plt.close(fig)


def specification_sensitivity(daily: pd.DataFrame):
    """RAW/SOFT70/STRICT70에서 핵심 결론이 유지되는지 비교한다."""
    rows = []
    mode_map = {
        "raw": {
            "er_pct": "eventrisk_percentile_train_raw",
            "er_z": "eventrisk_log1p_zscore_train_raw",
            "dgri_pct": "dgri_percentile_train_raw",
        },
        "soft70": {
            "er_pct": "eventrisk_percentile_train_soft70",
            "er_z": "eventrisk_log1p_zscore_train_soft70",
            "dgri_pct": "dgri_percentile_train_soft70",
        },
        "strict70": {
            "er_pct": "eventrisk_percentile_train_strict70",
            "er_z": "eventrisk_log1p_zscore_train_strict70",
            "dgri_pct": "dgri_percentile_train_strict70",
        },
    }

    for mode, cols in mode_map.items():
        row = {"mode": mode}
        test = daily[daily["date"].ge(TEST_START)].copy()

        # Shock occurrence at the main 95th-percentile threshold.
        for commodity in COMMODITIES:
            count = int(
                (
                    test["commodity"].eq(commodity)
                    & pd.to_numeric(test[cols["er_pct"]], errors="coerce")
                    .ge(MAIN_SHOCK_PERCENTILE)
                ).sum()
            )
            row[f"shock_days_{commodity.lower()}"] = count

        # Latest common DGRI state.
        p = daily.pivot(
            index="date", columns="commodity", values=cols["dgri_pct"]
        ).reindex(columns=COMMODITIES)
        common = p.dropna(how="any")
        if common.empty:
            row.update({
                "latest_common_date": "",
                "current_risk_leader": "UNAVAILABLE",
                "current_max_dgri_pct": np.nan,
            })
        else:
            d = common.index.max()
            vals = common.loc[d]
            row.update({
                "latest_common_date": str(pd.Timestamp(d).date()),
                "current_risk_leader": str(vals.idxmax()),
                "current_max_dgri_pct": float(vals.max()),
            })

        # Current-window connectedness without bootstrap (sensitivity only).
        try:
            panel = make_eventrisk_panel(daily, cols["er_z"])
            fit, _, start, end = fit_current_window(panel, TEST_START)
            theta = generalized_fevd(fit, FEVD_HORIZON)
            conn = directional_connectedness(theta, f"{mode}_current")
            top = conn.sort_values("NET_sender_pct", ascending=False).iloc[0]
            row.update({
                "transmission_status": "OK",
                "net_sender": str(top["commodity"]),
                "net_sender_pct": float(top["NET_sender_pct"]),
                "TCI_pct": float(top["TCI_pct"]),
                "var_lag": int(fit["p"]),
                "var_nobs": int(fit["nobs"]),
                "var_window_start": str(pd.Timestamp(start).date()),
                "var_window_end": str(pd.Timestamp(end).date()),
            })
        except (RuntimeError, ValueError):
            row.update({
                "transmission_status": "INSUFFICIENT_VAR_OBS",
                "net_sender": "UNAVAILABLE",
                "net_sender_pct": np.nan,
                "TCI_pct": np.nan,
                "var_lag": np.nan,
                "var_nobs": np.nan,
                "var_window_start": "",
                "var_window_end": "",
            })
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    event_path, daily_path = resolve_inputs()
    print("=" * 78)
    print("RISK LEADERSHIP FINAL V3 + DAILY ROLLING 30D")
    print(f"DAILY INPUT : {daily_path}")
    print(f"EVENT INPUT : {event_path}")
    print(f"OUTPUT      : {OUTPUT_DIR}")
    print(f"INPUT VER.  : {METHODOLOGY_VERSION}")
    print(f"ANALYSIS VER: {ANALYSIS_VERSION}")
    print(f"ROLLING     : strict {ROLLING_WINDOW_DAYS}D window / DAILY update / fixed VAR({ROLLING_VAR_LAG})")
    print("=" * 78)

    daily = load_daily_from_02(daily_path)
    phi_df = build_decay_parameter_summary(daily)
    _save_csv(daily, "01_daily_eventrisk_soft70.csv")
    _save_csv(phi_df, "decay_parameters_soft70.csv")

    # OOS validation
    ar1_oos = oos_ar1_validation(daily)
    dgri_oos = oos_dgri_predictive_validation(daily)
    leadership_detail, leadership_summary = oos_leadership_validation(daily)
    _save_csv(ar1_oos, "13_oos_ar1_validation.csv")
    _save_csv(dgri_oos, "14_oos_dgri_predictive_validation.csv")
    _save_csv(leadership_detail, "15_oos_leadership_validation_detail.csv")
    _save_csv(leadership_summary, "16_oos_leadership_validation_summary.csv")

    # Shock sensitivity and main 95th-percentile shock
    shock_sensitivity, shock_map = shock_threshold_sensitivity(daily)
    main_shocks = shock_map[MAIN_SHOCK_PERCENTILE]
    main_shock_summary = summarize_shock_occurrence(
        main_shocks, MAIN_SHOCK_PERCENTILE
    )
    _save_csv(main_shocks, "02_shock_days.csv")
    _save_csv(main_shock_summary, "05_shock_occurrence_summary.csv")
    _save_csv(shock_sensitivity, "05b_shock_threshold_sensitivity.csv")

    # Optional episode matching
    if event_path is None or main_shocks.empty:
        clustered_nodes = pd.DataFrame()
        episodes = pd.DataFrame()
        diagnostics = pd.DataFrame()
        onset_summary = pd.DataFrame(
            columns=["origin_leader", "episode_count", "share"]
        )
    else:
        nodes = extract_shock_event_nodes(event_path, main_shocks)
        if nodes.empty:
            clustered_nodes = pd.DataFrame()
            episodes = pd.DataFrame()
            diagnostics = pd.DataFrame()
            onset_summary = pd.DataFrame(
                columns=["origin_leader", "episode_count", "share"]
            )
        else:
            clustered_nodes, episodes, diagnostics = cluster_episodes(nodes)
            onset_summary = summarize_onset(episodes)
    _save_csv(clustered_nodes, "03_shock_event_nodes.csv")
    _save_csv(episodes, "04_shock_episodes.csv")
    _save_csv(diagnostics, "episode_link_diagnostics.csv")
    _save_csv(onset_summary, "05_onset_leader_summary.csv")

    # Main z-score VAR and percentile robustness VAR
    main_var = _run_var_spec(daily, MAIN_VAR_COLUMN, "main_zscore")
    robust_var = _run_var_spec(daily, ROBUST_VAR_COLUMN, "robust_percentile")

    # Main outputs + legacy-compatible filenames
    _save_csv(main_var["lag_sel"], "06_var_lag_selection.csv")
    _save_csv(main_var["granger"], "07_granger_predictability.csv")
    _save_csv(main_var["full_long"], "08_gfevd_full_sample.csv")
    _save_csv(main_var["cur_lag_sel"], "current_window_var_lag_selection.csv")
    _save_csv(main_var["cur_long"], "09_gfevd_current_window.csv")
    main_transmission = pd.concat(
        [x for x in [main_var["full_conn"], main_var["cur_conn"]] if not x.empty],
        ignore_index=True,
    ) if (not main_var["full_conn"].empty or not main_var["cur_conn"].empty) else pd.DataFrame()
    _save_csv(main_transmission, "10_transmission_summary.csv")
    _save_csv(
        pd.concat([main_var["full_boot"], main_var["cur_boot"]], ignore_index=True),
        "10b_connectedness_bootstrap_ci.csv",
    )
    if not main_var["full_matrix"].empty:
        _save_csv(main_var["full_matrix"], "gfevd_full_matrix.csv", index=True)
    if not main_var["cur_matrix"].empty:
        _save_csv(main_var["cur_matrix"], "gfevd_current_matrix.csv", index=True)

    # Explicit robustness outputs
    _save_csv(robust_var["lag_sel"], "R01_var_lag_selection_percentile.csv")
    _save_csv(robust_var["granger"], "R02_granger_percentile.csv")
    _save_csv(robust_var["full_long"], "R03_gfevd_full_percentile.csv")
    _save_csv(robust_var["cur_long"], "R04_gfevd_current_percentile.csv")
    robust_transmission = pd.concat(
        [x for x in [robust_var["full_conn"], robust_var["cur_conn"]] if not x.empty],
        ignore_index=True,
    ) if (not robust_var["full_conn"].empty or not robust_var["cur_conn"].empty) else pd.DataFrame()
    _save_csv(robust_transmission, "R05_transmission_percentile.csv")

    # Strict 30-day rolling network, re-estimated every calendar day.
    rolling = rolling_transmission_analysis(daily, MAIN_VAR_COLUMN)
    rolling_history = rolling["history"]
    rolling_matrices = rolling["matrices"]
    rolling_lag_diagnostics = rolling["lag_diagnostics"]
    rolling_bootstrap = rolling["bootstrap"]
    rolling_stability = rolling["stability"]
    train_connectedness = rolling["train_connectedness"]
    train_matrix = rolling["train_matrix"]

    daily_state = build_daily_risk_and_transmission_state(
        daily, rolling_history
    )
    rolling_comparison = rolling_train_test_comparison(
        rolling_history,
        rolling_stability,
        train_connectedness,
    )

    _save_csv(rolling_history, "17_rolling_30d_daily_transmission_history.csv")
    _save_csv(
        rolling_lag_diagnostics,
        "17b_rolling_30d_daily_var_diagnostics.csv",
    )
    _save_csv(rolling_matrices, "18_rolling_30d_daily_gfevd_matrix.csv")
    _save_csv(
        rolling_bootstrap,
        "18b_rolling_30d_daily_checkpoint_bootstrap_ci.csv",
    )
    _save_csv(
        daily_state,
        "19_daily_risk_and_transmission_state.csv",
    )
    _save_csv(
        rolling_comparison,
        "20_train_test_daily_rolling_comparison.csv",
    )
    _save_csv(
        train_connectedness,
        "21_train_connectedness_summary.csv",
    )
    if not train_matrix.empty:
        _save_csv(train_matrix, "22_train_gfevd_matrix.csv", index=True)
    else:
        _save_csv(pd.DataFrame(), "22_train_gfevd_matrix.csv")
    _save_csv(
        rolling_stability,
        "23_daily_rolling_vs_train_network_stability.csv",
    )

    spec_sensitivity = specification_sensitivity(daily)
    _save_csv(
        spec_sensitivity,
        "S01_raw_soft_strict_sensitivity.csv",
    )

    current_state = current_risk_state(daily)
    _save_csv(current_state, "11_current_risk_state.csv")

    # Final answers
    shock_top = main_shock_summary.sort_values(
        ["shock_days", "max_shock_percentile"], ascending=False
    ).iloc[0]
    counts = ", ".join(
        f"{r.commodity} {int(r.shock_days)}회"
        for r in main_shock_summary.itertuples(index=False)
    )
    shock_answer = (
        shock_top["commodity"] if int(main_shock_summary["shock_days"].sum()) > 0
        else "NO_EXTREME_SHOCKS"
    )
    shock_detail = (
        f"95th-percentile shock-day count: {counts}; "
        f"총 {int(main_shock_summary['shock_days'].sum())}일"
    )

    latest_daily_roll = rolling_history[
        rolling_history["status"].eq("OK")
    ].sort_values("as_of_date").tail(1)
    if latest_daily_roll.empty:
        transmission_answer = "INSUFFICIENT_VAR_OBS"
        transmission_detail = "Daily 30D rolling VAR/GFEVD 유효관측치 부족"
        conn_for_answer = main_var["cur_conn"]
    else:
        latest_row = latest_daily_roll.iloc[0]
        transmission_answer = str(latest_row["transmission_leader"])
        transmission_detail = (
            f"30D daily rolling NET={latest_row['leader_net_sender_pct']:.2f}pp; "
            f"TCI={latest_row['TCI_pct']:.2f}%; "
            f"window={pd.Timestamp(latest_row['window_start']).date()}~"
            f"{pd.Timestamp(latest_row['window_end']).date()}; "
            f"fixed VAR({ROLLING_VAR_LAG}); n={int(latest_row['effective_obs'])}"
        )
        if np.isfinite(latest_row.get("leader_net_ci_low", np.nan)):
            transmission_detail += (
                f"; NET 95% bootstrap CI=[{latest_row['leader_net_ci_low']:.2f}, "
                f"{latest_row['leader_net_ci_high']:.2f}]"
            )

        latest_conn_rows = []
        for commodity in COMMODITIES:
            latest_conn_rows.append({
                "sample": "latest_daily_30d",
                "commodity": commodity,
                "TO_others_pct": latest_row[f"{commodity.lower()}_to_others_pct"],
                "FROM_others_pct": latest_row[f"{commodity.lower()}_from_others_pct"],
                "NET_sender_pct": latest_row[f"{commodity.lower()}_net_sender_pct"],
                "TCI_pct": latest_row["TCI_pct"],
            })
        conn_for_answer = pd.DataFrame(latest_conn_rows)

    risk_top = current_state.iloc[0]
    final = pd.DataFrame([
        {
            "question": "A. TEST에서 극단 EventRisk shock이 어느 시장에서 더 자주 발생했는가?",
            "answer": shock_answer,
            "metric": "95th-percentile shock-day count",
            "detail": shock_detail,
        },
        {
            "question": "B. TEST 종료시점 근방에서 상대적 위험 순전달자는 누구인가?",
            "answer": transmission_answer,
            "metric": "daily 30D log-EventRisk z-score GFEVD NET / TCI",
            "detail": transmission_detail,
        },
        {
            "question": "C. 최신 누적 위험 leader와 전체 위험수준은?",
            "answer": risk_top["commodity"],
            "metric": "Train-calibrated SOFT-70 DGRI percentile",
            "detail": (
                f"leader percentile={risk_top['dgri_percentile']:.2f}; "
                f"system={risk_top['system_risk_state']}; "
                f"update={risk_top['dgri_update_type']}; "
                f"date={pd.Timestamp(risk_top['date']).date()}"
            ),
        },
    ])
    _save_csv(final, "12_final_three_answers.csv")

    save_graphs(
        main_shock_summary,
        conn_for_answer,
        current_state,
    )
    save_rolling_graphs(
        rolling_history,
        daily_state,
        rolling_stability,
    )

    # Human-readable summary
    lines = [
        "=== DESIGN ===",
        f"Version: {METHODOLOGY_VERSION}",
        f"Train: {STUDY_START.date()} ~ {pd.Timestamp(CALIBRATION_END).date()}",
        f"Test : {TEST_START.date()} ~ {STUDY_END.date()}",
        "Main: SOFT-70; VAR input=log(1+EventRisk) Train z-score",
        "",
        "=== FINAL ANSWERS ===",
    ]
    for row in final.itertuples(index=False):
        lines.append(f"{row.question}\n  -> {row.answer}\n  -> {row.detail}")
    lines += [
        "", "=== SHOCK THRESHOLD SENSITIVITY ===",
        shock_sensitivity.to_string(index=False),
        "", "=== CURRENT RISK ===", current_state.to_string(index=False),
        "", "=== OOS AR(1) ===", ar1_oos.to_string(index=False),
        "", "=== OOS DGRI (all thresholds) ===", dgri_oos.to_string(index=False),
        "", "=== MAIN CONNECTEDNESS ===", main_transmission.to_string(index=False),
        "", "=== PERCENTILE VAR ROBUSTNESS ===", robust_transmission.to_string(index=False),
        "", "=== RAW / SOFT70 / STRICT70 SENSITIVITY ===", spec_sensitivity.to_string(index=False),
        "", "=== DAILY ROLLING TRANSMISSION: STRICT 30D WINDOW / DAILY UPDATE / FIXED VAR(1) ===", rolling_history.to_string(index=False),
        "", "=== TRAIN FULL NETWORK BENCHMARK ===", train_connectedness.to_string(index=False),
        "", "=== TRAIN / TRANSITION / TEST DAILY-ROLLING COMPARISON ===", rolling_comparison.to_string(index=False),
        "", "=== DAILY-ROLLING NETWORK STABILITY VS TRAIN ===", rolling_stability.to_string(index=False),
        "", "주의: Episode는 선택적 보조분석이다.",
        "주의: GFEVD는 predictive connectedness이며 구조적 인과가 아니다.",
    ]
    (OUTPUT_DIR / "analysis_summary.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    # Output manifest
    files = sorted([p for p in OUTPUT_DIR.rglob("*") if p.is_file()])
    manifest_df = pd.DataFrame([
        {
            "relative_path": str(p.relative_to(OUTPUT_DIR)),
            "size_bytes": p.stat().st_size,
            "modified_time": pd.Timestamp(p.stat().st_mtime, unit="s").isoformat(),
        }
        for p in files
    ])
    _save_csv(manifest_df, "00_output_manifest.csv")
    manifest = {
        "methodology_version": METHODOLOGY_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "daily_input": str(daily_path.resolve()),
        "event_input": str(event_path.resolve()) if event_path else None,
        "output_dir": str(OUTPUT_DIR.resolve()),
        "main_var_column": MAIN_VAR_COLUMN,
        "robust_var_column": ROBUST_VAR_COLUMN,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "rolling_window_days": ROLLING_WINDOW_DAYS,
        "rolling_update_step_days": ROLLING_STEP_DAYS,
        "rolling_fixed_var_lag": ROLLING_VAR_LAG,
        "rolling_min_effective_obs": ROLLING_MIN_EFFECTIVE_OBS,
        "rolling_bootstrap_reps_at_checkpoints": ROLLING_BOOTSTRAP_REPS,
        "rolling_bootstrap_month_end_only": ROLLING_BOOTSTRAP_MONTH_END_ONLY,
        "rolling_final_snapshot_included": ROLLING_INCLUDE_FINAL_DATE,
        "run_timestamp_utc": pd.Timestamp.utcnow().isoformat(),
    }
    (OUTPUT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n04 completed successfully.")
    print(f"Output: {OUTPUT_DIR.resolve()}")
    print(final.to_string(index=False))



if __name__ == "__main__":
    main()
