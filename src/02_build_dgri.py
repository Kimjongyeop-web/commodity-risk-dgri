#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
02_build_dgri.py
============================

고정 연구설계
------------
- 전체: 2024-10-01 ~ 2025-05-31
- Train: 2024-10-01 ~ 2025-03-31
- Test : 2025-04-01 ~ 2025-05-31

역할
----
01이 만든 날짜×시장별 EventRisk/QC 자료를 읽어 RAW, SOFT-70, STRICT-70
세 specification을 동시에 계산한다. 메인 결과는 SOFT-70이다.

V3 핵심 수정
------------
1) 입력 경로를 01 V3 output 하나로 고정한다.
2) output을 methodology version 폴더에 저장한다.
3) RAW/SOFT-70/STRICT-70을 함께 계산해 민감도 비교가 가능하다.
4) phi는 Train에서 specification별로 한 번만 추정하고 Test에서 고정한다.
5) DGRI percentile reference는 Train 중 EventRisk가 실제 관측된 날만 사용한다.
6) percentile 열 이름에 2y/rolling 같은 잘못된 표현을 사용하지 않는다.
7) VAR용 log(1+EventRisk) Train z-score를 함께 만든다.
8) missing EventRisk는 신규 shock 없이 기존 DGRI state만 감쇠시킨다.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


COMMODITIES = ["Oil", "Gas", "Power"]
MODES = ["raw", "soft70", "strict70"]
MAIN_MODE = "soft70"

STUDY_START = pd.Timestamp("2024-10-01")
TRAIN_END = pd.Timestamp("2025-03-31")
TEST_START = pd.Timestamp("2025-04-01")
STUDY_END = pd.Timestamp("2025-05-31")

COVERAGE_TARGET = 0.70
MIN_PHI_VALID_PAIRS = 10
PHI_MIN = 0.0
PHI_MAX = 0.999

METHODOLOGY_VERSION = "2024OCT_2025MAY_TRAIN6M_TEST2M_SOFT70_V3"
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = (
    SCRIPT_DIR
    / "eventrisk_research_final_output"
    / "outputs"
    / METHODOLOGY_VERSION
    / "dgri_input.csv"
)
OUTPUT_DIR = SCRIPT_DIR / "dgri_output" / METHODOLOGY_VERSION


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


def _parse_date(df: pd.DataFrame) -> pd.Series:
    if "date" in df.columns:
        return pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    if "event_date" in df.columns:
        raw = (
            df["event_date"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.strip()
        )
        return pd.to_datetime(raw, format="%Y%m%d", errors="coerce").dt.normalize()
    raise ValueError("입력에는 date 또는 event_date 열이 필요합니다.")


def load_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            "01 V3 output을 찾지 못했습니다. 먼저 01_collect_event_risk.py를 "
            f"실행하세요.\nExpected: {path}"
        )

    df = pd.read_csv(path, low_memory=False)
    required = {
        "commodity",
        "daily_event_risk_observed",
        "article_coverage_rate",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"02 필수 입력 열이 없습니다: {sorted(missing)}")

    out = df.copy()
    out["date"] = _parse_date(out)
    out["commodity"] = (
        out["commodity"].astype(str).str.strip().str.capitalize()
    )
    out = out[
        out["commodity"].isin(COMMODITIES)
        & out["date"].between(STUDY_START, STUDY_END)
    ].copy()

    numeric_cols = [
        "daily_event_risk_observed",
        "daily_event_risk_raw",
        "daily_event_risk_strict70",
        "article_coverage_rate",
        "candidate_event_count",
        "article_success_count",
        "event_count",
        "max_event_risk",
        "mean_event_risk",
        "raw_event_count",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "source_status" not in out.columns:
        out["source_status"] = np.where(
            out["daily_event_risk_observed"].notna(),
            "UNKNOWN_OBSERVED",
            "GDELT2_SOURCE_MISSING",
        )

    if "source_complete" in out.columns:
        out["source_complete"] = _as_bool(out["source_complete"])
    else:
        out["source_complete"] = out["source_status"].eq("GDELT2_HTTP_OK")

    if "source_partial" in out.columns:
        out["source_partial"] = _as_bool(out["source_partial"])
    else:
        out["source_partial"] = out["source_status"].eq("GDELT2_HTTP_PARTIAL")

    if "source_available" in out.columns:
        out["source_available"] = _as_bool(out["source_available"])
    else:
        out["source_available"] = (
            out["source_complete"] | out["source_partial"]
        )

    if "genuine_zero" in out.columns:
        out["genuine_zero"] = _as_bool(out["genuine_zero"])
    else:
        candidate = pd.to_numeric(
            out.get("candidate_event_count", 0), errors="coerce"
        ).fillna(0)
        out["genuine_zero"] = out["source_complete"] & candidate.eq(0)

    if "methodology_version" in out.columns:
        versions = set(out["methodology_version"].dropna().astype(str).unique())
        if versions and versions != {METHODOLOGY_VERSION}:
            raise RuntimeError(
                "01 output version이 02 V3와 다릅니다.\n"
                f"Found: {sorted(versions)}\nExpected: {METHODOLOGY_VERSION}"
            )

    dup = out.duplicated(["date", "commodity"], keep=False)
    if dup.any():
        bad = out.loc[dup, ["date", "commodity"]].drop_duplicates()
        raise ValueError(
            "date/commodity 중복행이 있습니다:\n" + bad.to_string(index=False)
        )

    return out.sort_values(["commodity", "date"]).reset_index(drop=True)


def build_eventrisk_specifications(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    observed = out["daily_event_risk_observed"].notna()
    coverage = out["article_coverage_rate"].clip(0.0, 1.0)

    # RAW: 실제로 관측된 값은 가중하지 않고 사용한다.
    if "daily_event_risk_raw" in out.columns:
        raw = pd.to_numeric(out["daily_event_risk_raw"], errors="coerce")
    else:
        raw = out["daily_event_risk_observed"].where(
            out["source_available"] | out["genuine_zero"]
        )
    out["daily_event_risk_raw"] = raw

    # SOFT-70 메인: coverage를 아는 관측치만 사용한다.
    out["soft70_weight"] = np.nan
    known = observed & coverage.notna()
    out.loc[known, "soft70_weight"] = np.minimum(
        1.0, coverage[known] / COVERAGE_TARGET
    )
    out.loc[out["genuine_zero"], "soft70_weight"] = 1.0
    out["daily_event_risk_soft70"] = (
        out["daily_event_risk_observed"] * out["soft70_weight"]
    )

    # STRICT-70: 완전한 source + coverage 70% 이상만 사용한다.
    if "daily_event_risk_strict70" in out.columns:
        strict = pd.to_numeric(
            out["daily_event_risk_strict70"], errors="coerce"
        )
    else:
        strict = out["daily_event_risk_observed"].where(
            out["source_complete"] & coverage.ge(COVERAGE_TARGET)
        )
    out["daily_event_risk_strict70"] = strict

    out["soft70_status"] = np.select(
        [
            out["genuine_zero"],
            ~observed & out["source_status"].eq("GDELT2_SOURCE_MISSING"),
            observed & coverage.isna(),
            observed & out["source_partial"],
            observed & coverage.lt(COVERAGE_TARGET),
        ],
        [
            "GENUINE_ZERO",
            "SOURCE_MISSING",
            "COVERAGE_UNKNOWN_EXCLUDED",
            "PARTIAL_SOURCE_OBSERVED",
            "LOW_COVERAGE_SOFT_WEIGHTED",
        ],
        default="FULL_WEIGHT",
    )

    out["period"] = np.where(
        out["date"].le(TRAIN_END),
        "TRAIN_CALIBRATION",
        "TEST_EVALUATION",
    )
    return out


def estimate_phi(g: pd.DataFrame, eventrisk_col: str, mode: str) -> dict:
    train = g[g["date"].between(STUDY_START, TRAIN_END)].sort_values("date")
    er = pd.to_numeric(train[eventrisk_col], errors="coerce").to_numpy(float)
    dates = train["date"].to_numpy(dtype="datetime64[D]")

    lag = er[:-1]
    cur = er[1:]
    gap = (dates[1:] - dates[:-1]).astype(int)
    valid = np.isfinite(lag) & np.isfinite(cur) & gap.__eq__(1)
    x = lag[valid]
    y = cur[valid]

    if len(x) < MIN_PHI_VALID_PAIRS:
        raise RuntimeError(
            f"{g['commodity'].iloc[0]} / {mode}: AR(1) 유효 pair가 "
            f"{len(x)}개뿐입니다."
        )

    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    intercept, phi_raw = float(beta[0]), float(beta[1])
    phi = float(np.clip(phi_raw, PHI_MIN, PHI_MAX))

    fitted = intercept + phi_raw * x
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = np.nan if ss_tot <= 0 else 1.0 - ss_res / ss_tot

    if 0 < phi < 1:
        half_life = float(math.log(0.5) / math.log(phi))
    elif phi == 0:
        half_life = 0.0
    else:
        half_life = np.inf

    return {
        "commodity": g["commodity"].iloc[0],
        "mode": mode,
        "eventrisk_column": eventrisk_col,
        "train_start": STUDY_START.date(),
        "train_end": TRAIN_END.date(),
        "intercept": intercept,
        "phi_raw": phi_raw,
        "phi_used": phi,
        "half_life_days": half_life,
        "r_squared_train": r2,
        "valid_train_pairs": int(len(x)),
        "success": True,
        "failure_reason": "",
    }


def build_dgri(
    dates: pd.Series,
    eventrisk: pd.Series,
    phi: float,
) -> tuple[np.ndarray, np.ndarray]:
    dates = pd.to_datetime(dates).reset_index(drop=True)
    er = pd.to_numeric(eventrisk, errors="coerce").reset_index(drop=True)
    dgri = np.full(len(er), np.nan, dtype=float)
    status = np.empty(len(er), dtype=object)

    prev_value = np.nan
    prev_date = None
    started = False

    for i, (day, shock) in enumerate(zip(dates, er)):
        if prev_date is not None and started:
            gap_days = int((day - prev_date).days)
            if gap_days > 1:
                prev_value *= phi ** (gap_days - 1)

        if np.isfinite(shock):
            base = prev_value if started and np.isfinite(prev_value) else 0.0
            value = float(shock) + phi * float(base)
            status[i] = "OBSERVED_EVENTRISK_UPDATE"
            started = True
        elif started and np.isfinite(prev_value):
            value = phi * float(prev_value)
            status[i] = "EVENTRISK_MISSING_DECAY_ONLY"
        else:
            value = np.nan
            status[i] = "EVENTRISK_MISSING_BEFORE_STATE"

        dgri[i] = value
        if np.isfinite(value):
            prev_value = value
        prev_date = day

    return dgri, status


def ecdf_midpoint(values: pd.Series, reference: np.ndarray) -> pd.Series:
    ref = np.sort(np.asarray(reference, dtype=float))
    ref = ref[np.isfinite(ref)]
    out = pd.Series(np.nan, index=values.index, dtype=float)
    if len(ref) == 0:
        return out

    vals = pd.to_numeric(values, errors="coerce")
    for idx, value in vals.items():
        if not np.isfinite(value):
            continue
        left = np.searchsorted(ref, value, side="left")
        right = np.searchsorted(ref, value, side="right")
        out.loc[idx] = 100.0 * (left + 0.5 * (right - left)) / len(ref)
    return out


def fixed_zscore(values: pd.Series, reference: np.ndarray, ddof: int = 1) -> pd.Series:
    ref = np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    if len(ref) < max(2, ddof + 1):
        return pd.Series(np.nan, index=values.index, dtype=float)
    mu = float(np.mean(ref))
    sd = float(np.std(ref, ddof=ddof))
    if sd <= 0:
        return pd.Series(0.0, index=values.index, dtype=float)
    return (pd.to_numeric(values, errors="coerce") - mu) / sd


def calculate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_parts = []
    param_rows = []

    mode_cols = {
        "raw": "daily_event_risk_raw",
        "soft70": "daily_event_risk_soft70",
        "strict70": "daily_event_risk_strict70",
    }

    for commodity in COMMODITIES:
        g = (
            df[df["commodity"].eq(commodity)]
            .sort_values("date")
            .copy()
            .reset_index(drop=True)
        )
        if g.empty:
            continue

        train_mask = g["date"].between(STUDY_START, TRAIN_END)

        for mode, er_col in mode_cols.items():
            try:
                params = estimate_phi(g, er_col, mode)
            except RuntimeError as exc:
                if mode == MAIN_MODE:
                    raise
                params = {
                    "commodity": commodity,
                    "mode": mode,
                    "eventrisk_column": er_col,
                    "train_start": STUDY_START.date(),
                    "train_end": TRAIN_END.date(),
                    "intercept": np.nan,
                    "phi_raw": np.nan,
                    "phi_used": np.nan,
                    "half_life_days": np.nan,
                    "r_squared_train": np.nan,
                    "valid_train_pairs": int(
                        g.loc[train_mask, er_col].notna().sum()
                    ),
                    "success": False,
                    "failure_reason": str(exc),
                }
                param_rows.append(params)
                g[f"decay_intercept_{mode}"] = np.nan
                g[f"decay_phi_raw_{mode}"] = np.nan
                g[f"decay_phi_{mode}"] = np.nan
                g[f"dgri_{mode}"] = np.nan
                g[f"dgri_status_{mode}"] = "SPECIFICATION_UNAVAILABLE"
                g[f"eventrisk_percentile_train_{mode}"] = np.nan
                g[f"eventrisk_log1p_zscore_train_{mode}"] = np.nan
                g[f"dgri_percentile_train_{mode}"] = np.nan
                g[f"dgri_zscore_train_{mode}"] = np.nan
                continue

            param_rows.append(params)
            phi = params["phi_used"]

            dgri, dgri_status = build_dgri(g["date"], g[er_col], phi)
            g[f"decay_intercept_{mode}"] = params["intercept"]
            g[f"decay_phi_raw_{mode}"] = params["phi_raw"]
            g[f"decay_phi_{mode}"] = phi
            g[f"dgri_{mode}"] = dgri
            g[f"dgri_status_{mode}"] = dgri_status

            er_ref = g.loc[
                train_mask & g[er_col].notna(), er_col
            ].to_numpy(float)

            # 핵심 V3 수정: DGRI reference는 실제 EventRisk 관측일만 사용한다.
            dgri_ref = g.loc[
                train_mask & g[er_col].notna() & g[f"dgri_{mode}"].notna(),
                f"dgri_{mode}",
            ].to_numpy(float)

            if len(er_ref) < 3 or len(dgri_ref) < 3:
                raise RuntimeError(
                    f"{commodity}/{mode}: Train reference가 너무 적습니다."
                )

            g[f"eventrisk_percentile_train_{mode}"] = ecdf_midpoint(
                g[er_col], er_ref
            )
            log_values = np.log1p(pd.to_numeric(g[er_col], errors="coerce").clip(lower=0))
            log_ref = np.log1p(np.clip(er_ref, 0, None))
            g[f"eventrisk_log1p_zscore_train_{mode}"] = fixed_zscore(
                log_values, log_ref
            )
            g[f"dgri_percentile_train_{mode}"] = ecdf_midpoint(
                g[f"dgri_{mode}"], dgri_ref
            )
            g[f"dgri_zscore_train_{mode}"] = fixed_zscore(
                g[f"dgri_{mode}"], dgri_ref
            )

        result_parts.append(g)

    if not result_parts:
        raise RuntimeError("계산 가능한 Oil/Gas/Power 데이터가 없습니다.")

    results = (
        pd.concat(result_parts, ignore_index=True)
        .sort_values(["date", "commodity"])
        .reset_index(drop=True)
    )
    params = pd.DataFrame(param_rows)

    # 메인 specification은 명시적 SOFT-70 열을 사용한다.
    results["daily_risk_rank"] = (
        results.groupby("date")["dgri_percentile_train_soft70"]
        .rank(method="min", ascending=False)
    )
    results["daily_risk_leader"] = results["daily_risk_rank"].eq(1)
    results["methodology_version"] = METHODOLOGY_VERSION

    return results, params


def save_outputs(results: pd.DataFrame, params: pd.DataFrame) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    long_path = OUTPUT_DIR / "shock_decay_dgri_long_final.csv"
    results.to_csv(long_path, index=False, encoding="utf-8-sig")
    params.to_csv(
        OUTPUT_DIR / "decay_parameters_all_specs.csv",
        index=False,
        encoding="utf-8-sig",
    )
    params[params["mode"].eq(MAIN_MODE)].to_csv(
        OUTPUT_DIR / "decay_parameters_soft70.csv",
        index=False,
        encoding="utf-8-sig",
    )

    main_metrics = [
        "daily_event_risk_soft70",
        "eventrisk_percentile_train_soft70",
        "eventrisk_log1p_zscore_train_soft70",
        "dgri_soft70",
        "dgri_percentile_train_soft70",
        "dgri_zscore_train_soft70",
        "decay_phi_soft70",
    ]
    frames = []
    for metric in main_metrics:
        p = results.pivot(index="date", columns="commodity", values=metric)
        p.columns = [f"{str(c).lower()}_{metric}" for c in p.columns]
        frames.append(p)
    pd.concat(frames, axis=1).reset_index().to_csv(
        OUTPUT_DIR / "shock_decay_dgri_wide_final.csv",
        index=False,
        encoding="utf-8-sig",
    )

    leaders = results.loc[
        results["daily_risk_leader"]
        & results["dgri_percentile_train_soft70"].notna(),
        [
            "date", "commodity", "dgri_soft70",
            "dgri_percentile_train_soft70", "daily_risk_rank", "dgri_status_soft70",
        ],
    ].rename(columns={"commodity": "risk_leader"})
    leaders.to_csv(
        OUTPUT_DIR / "daily_risk_leader.csv",
        index=False,
        encoding="utf-8-sig",
    )

    latest_date = results["date"].max()
    latest_cols = [
        "date", "commodity", "daily_event_risk_soft70",
        "dgri_soft70", "dgri_percentile_train_soft70", "dgri_status_soft70",
    ]
    latest = results.loc[results["date"].eq(latest_date), latest_cols].copy()
    for mode in MODES:
        latest[f"dgri_pct_{mode}"] = results.loc[
            results["date"].eq(latest_date),
            f"dgri_percentile_train_{mode}",
        ].to_numpy()
    latest.to_csv(
        OUTPUT_DIR / "specification_comparison_latest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = pd.DataFrame({
        "item": [
            "methodology_version", "study_start", "train_end", "test_start",
            "study_end", "main_specification", "coverage_target",
            "phi_estimation", "percentile_reference",
            "n_rows", "n_low_coverage_soft_weighted",
            "n_coverage_unknown_excluded", "n_source_missing",
        ],
        "value": [
            METHODOLOGY_VERSION, str(STUDY_START.date()), str(TRAIN_END.date()),
            str(TEST_START.date()), str(STUDY_END.date()), "SOFT70",
            COVERAGE_TARGET, "one-time AR(1) from TRAIN",
            "TRAIN observed-EventRisk days only", len(results),
            int(results["soft70_status"].eq("LOW_COVERAGE_SOFT_WEIGHTED").sum()),
            int(results["soft70_status"].eq("COVERAGE_UNKNOWN_EXCLUDED").sum()),
            int(results["soft70_status"].eq("SOURCE_MISSING").sum()),
        ],
    })
    summary.to_csv(
        OUTPUT_DIR / "dgri_run_qc_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "methodology_version": METHODOLOGY_VERSION,
        "input_path": str(INPUT_PATH.resolve()),
        "output_dir": str(OUTPUT_DIR.resolve()),
        "study_start": str(STUDY_START.date()),
        "train_end": str(TRAIN_END.date()),
        "test_start": str(TEST_START.date()),
        "study_end": str(STUDY_END.date()),
        "main_specification": MAIN_MODE,
        "modes": MODES,
        "run_timestamp_utc": pd.Timestamp.utcnow().isoformat(),
    }
    (OUTPUT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return long_path


def main() -> None:
    print("=" * 78)
    print("DGRI FINAL V3")
    print(f"INPUT  : {INPUT_PATH}")
    print(f"OUTPUT : {OUTPUT_DIR}")
    print(f"TRAIN  : {STUDY_START.date()} ~ {TRAIN_END.date()}")
    print(f"TEST   : {TEST_START.date()} ~ {STUDY_END.date()}")
    print("MAIN   : SOFT-70 | RAW/STRICT-70 sensitivity included")
    print("=" * 78)

    df = build_eventrisk_specifications(load_input(INPUT_PATH))
    results, params = calculate(df)
    long_path = save_outputs(results, params)

    print("\n=== SOFT-70 DECAY PARAMETERS ===")
    print(
        params[params["mode"].eq("soft70")][[
            "commodity", "phi_raw", "phi_used", "half_life_days",
            "valid_train_pairs",
        ]].to_string(index=False)
    )
    print(f"\n02 completed successfully.\nMain output: {long_path}")
    print("Next: 04_risk_leadership_analysis_FINAL_V3.py")


if __name__ == "__main__":
    main()
