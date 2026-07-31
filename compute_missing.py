"""
Master computation script for the paper data package.

This script focuses on the tasks called out in the project notes:
1. Wilcoxon signed-rank tests for random vs spatial RMSE.
2. Feature-importance stability with genuinely different seeded spatial folds.
3. Zambia 2012/2014 anomaly diagnostics with corrected direction.
4. Wheat 2017 anomaly diagnostics, including country breakdown.
5. Consolidated markdown package summarizing generated outputs.

It does not invent missing legacy artifacts. If a required deliverable cannot be
verified from the local project state, the markdown package flags that plainly.
"""

from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent

IGNORE_COLS = [
    "harvest_year",
    "latitude",
    "longitude",
    "harvest_area",
    "production",
    "country_code",
    "adm_id",
    "crop_name_x",
    "crop_name_y",
    "year",
    "key",
    "region1",
    "region2",
]

SEEDS = [42, 43, 44]
MODELS = {
    "Null Baseline": "null",
    "Ridge": Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
    "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(n_estimators=50, random_state=42, n_jobs=-1),
}

DATASETS = [
    {
        "name": "CY-Bench Maize (Europe)",
        "csv": "cybench_full.csv",
        "yield_col": "yield",
        "year_col": "harvest_year",
        "group_col": "adm_id",
        "country_col": "country_code",
        "family": "cybench",
    },
    {
        "name": "CY-Bench Wheat (Europe)",
        "csv": "cybench_wheat_full.csv",
        "yield_col": "yield",
        "year_col": "harvest_year",
        "group_col": "adm_id",
        "country_col": "country_code",
        "family": "cybench",
    },
    {
        "name": "CY-Bench Maize (Zambia)",
        "csv": "cybench_zambia_maize.csv",
        "yield_col": "yield",
        "year_col": "harvest_year",
        "group_col": "adm_id",
        "country_col": "country_code",
        "family": "cybench",
    },
    {
        "name": "SustainBench Soybean",
        "csv": "sustainbench_full.csv",
        "yield_col": "yield_t_ha",
        "year_col": "year",
        "group_col": "region1",
        "country_col": None,
        "family": "sustainbench",
    },
]

FEATURE_IMPORTANCE_DATASETS = [
    "CY-Bench Maize (Europe)",
    "CY-Bench Wheat (Europe)",
    "CY-Bench Maize (Zambia)",
]


def sanitize(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace("-", "-")
    )


def load_dataset(cfg: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(ROOT / cfg["csv"])
    yield_col = cfg["yield_col"]
    year_col = cfg["year_col"]
    group_col = cfg["group_col"]

    df = df[df[yield_col] > 0].copy()
    if cfg["family"] == "sustainbench":
        df["harvest_year"] = df[year_col].astype(int)
        df["adm_id"] = df[group_col].astype(str)
    else:
        df["harvest_year"] = df[year_col].astype(int)
        df["adm_id"] = df[group_col].astype(str)

    numeric_cols = df.select_dtypes(include="number").columns
    predictor_cols = [
        col
        for col in numeric_cols
        if col not in IGNORE_COLS
        and col != yield_col
        and not (cfg["family"] == "sustainbench" and col.endswith("_min"))
    ]
    df = df.dropna(subset=predictor_cols).copy()
    df["_yield"] = df[yield_col].astype(float)
    df["_year"] = df["harvest_year"].astype(int)
    df["_group"] = df["adm_id"].astype(str)
    X = df[predictor_cols].to_numpy()
    y = df["_yield"].to_numpy()
    return df, X, y, predictor_cols


def clone_model(model_name: str):
    base = MODELS[model_name]
    if model_name == "Null Baseline":
        return base
    return sklearn.base.clone(base)


def fit_predict(model_name: str, X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
    if model_name == "Null Baseline":
        return np.full(len(test_idx), float(np.mean(y[train_idx])))
    model = clone_model(model_name)
    model.fit(X[train_idx], y[train_idx])
    return model.predict(X[test_idx])


def evaluate_main_results(df: pd.DataFrame, X: np.ndarray, y: np.ndarray, dataset_name: str) -> pd.DataFrame:
    groups = df["_group"].to_numpy()
    years = df["_year"].to_numpy()
    unique_years = np.sort(np.unique(years))
    rows: list[dict] = []

    for model_name in MODELS:
        random_rmse, random_r2 = [], []
        for seed in SEEDS:
            train_idx, test_idx = train_test_split(
                np.arange(len(y)),
                test_size=0.2,
                random_state=seed,
            )
            preds = fit_predict(model_name, X, y, train_idx, test_idx)
            random_rmse.append(root_mean_squared_error(y[test_idx], preds))
            random_r2.append(r2_score(y[test_idx], preds))
        rows.append(
            summary_row(dataset_name, model_name, "Random", random_rmse, random_r2)
        )

        spatial_rmse, spatial_r2 = [], []
        for train_idx, test_idx in GroupKFold(n_splits=5).split(X, y, groups=groups):
            preds = fit_predict(model_name, X, y, train_idx, test_idx)
            spatial_rmse.append(root_mean_squared_error(y[test_idx], preds))
            spatial_r2.append(r2_score(y[test_idx], preds))
        rows.append(
            summary_row(dataset_name, model_name, "Spatial", spatial_rmse, spatial_r2)
        )

        temporal_rmse, temporal_r2 = [], []
        for i in range(4, len(unique_years) - 1):
            train_years = unique_years[: i + 1]
            test_year = unique_years[i + 1]
            train_idx = np.where(np.isin(years, train_years))[0]
            test_idx = np.where(years == test_year)[0]
            if len(train_idx) == 0 or len(test_idx) < 5:
                continue
            preds = fit_predict(model_name, X, y, train_idx, test_idx)
            temporal_rmse.append(root_mean_squared_error(y[test_idx], preds))
            temporal_r2.append(r2_score(y[test_idx], preds))
        if temporal_rmse:
            rows.append(
                summary_row(dataset_name, model_name, "Temporal", temporal_rmse, temporal_r2)
            )

        st_rmse, st_r2 = [], []
        unique_regions = np.unique(groups)
        tail_years = unique_years[-max(1, int(len(unique_years) * 0.2)) :]
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            n_regions = max(1, int(len(unique_regions) * 0.2))
            test_regions = rng.choice(unique_regions, size=n_regions, replace=False)
            train_mask = (~np.isin(groups, test_regions)) & (~np.isin(years, tail_years))
            test_mask = np.isin(groups, test_regions) & np.isin(years, tail_years)
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            preds = fit_predict(model_name, X, y, train_idx, test_idx)
            st_rmse.append(root_mean_squared_error(y[test_idx], preds))
            st_r2.append(r2_score(y[test_idx], preds))
        if st_rmse:
            rows.append(
                summary_row(dataset_name, model_name, "Spatiotemporal", st_rmse, st_r2)
            )

    return pd.DataFrame(rows)


def summary_row(dataset_name: str, model_name: str, split_name: str, rmses: list[float], r2s: list[float]) -> dict:
    return {
        "Dataset": dataset_name,
        "Model": model_name,
        "Split": split_name,
        "RMSE": f"{np.mean(rmses):.3f} +/- {np.std(rmses):.3f}",
        "R2": f"{np.mean(r2s):.3f} +/- {np.std(r2s):.3f}",
    }


def rank_biserial(diffs: np.ndarray) -> tuple[float, float, float]:
    abs_diffs = np.abs(diffs)
    ranks = pd.Series(abs_diffs).rank(method="average").to_numpy()
    w_plus = float(ranks[diffs > 0].sum())
    w_minus = float(ranks[diffs < 0].sum())
    denom = w_plus + w_minus
    effect = (w_plus - w_minus) / denom if denom else 0.0
    return w_plus, w_minus, effect


def wilcoxon_random_vs_spatial(
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    model_name: str = "Random Forest",
    seed: int = 42,
) -> dict:
    groups = df["_group"].to_numpy()
    spatial_rmse = []
    random_rmse = []

    spatial_splitter = GroupKFold(n_splits=n_splits)
    random_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for (sp_train, sp_test), (rd_train, rd_test) in zip(
        spatial_splitter.split(X, y, groups=groups),
        random_splitter.split(X, y),
    ):
        spatial_preds = fit_predict(model_name, X, y, sp_train, sp_test)
        random_preds = fit_predict(model_name, X, y, rd_train, rd_test)
        spatial_rmse.append(root_mean_squared_error(y[sp_test], spatial_preds))
        random_rmse.append(root_mean_squared_error(y[rd_test], random_preds))

    random_rmse_arr = np.array(random_rmse)
    spatial_rmse_arr = np.array(spatial_rmse)
    diffs = spatial_rmse_arr - random_rmse_arr
    stat, p_value = wilcoxon(diffs, alternative="greater", zero_method="wilcox")
    w_plus, w_minus, effect = rank_biserial(diffs)

    return {
        "Folds": n_splits,
        "Random_RMSE": random_rmse_arr,
        "Spatial_RMSE": spatial_rmse_arr,
        "Diffs": diffs,
        "Statistic": float(stat),
        "PValue": float(p_value),
        "WPlus": w_plus,
        "WMinus": w_minus,
        "RankBiserialR": effect,
        "SpatialWorseFolds": int((diffs > 0).sum()),
    }


def make_seeded_group_folds(groups: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled_groups = unique_groups.copy()
    rng.shuffle(shuffled_groups)
    group_folds = np.array_split(shuffled_groups, n_splits)

    splits = []
    for fold_groups in group_folds:
        test_mask = np.isin(groups, fold_groups)
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(~test_mask)[0]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        splits.append((train_idx, test_idx))
    return splits


def compute_feature_importance_table(
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    predictor_cols: list[str],
    dataset_name: str,
) -> tuple[pd.DataFrame, list[dict]]:
    groups = df["_group"].to_numpy()
    random_seed_importance = []
    spatial_seed_importance = []
    fold_logs = []

    for seed in SEEDS:
        xgb_random = XGBRegressor(n_estimators=50, random_state=seed, n_jobs=-1)
        train_idx, _ = train_test_split(
            np.arange(len(y)),
            test_size=0.2,
            random_state=seed,
        )
        xgb_random.fit(X[train_idx], y[train_idx])
        random_seed_importance.append(xgb_random.feature_importances_)

        fold_importances = []
        for fold_id, (sp_train, sp_test) in enumerate(
            make_seeded_group_folds(groups, n_splits=5, seed=seed),
            start=1,
        ):
            xgb_spatial = XGBRegressor(n_estimators=50, random_state=seed, n_jobs=-1)
            xgb_spatial.fit(X[sp_train], y[sp_train])
            fold_importances.append(xgb_spatial.feature_importances_)
            held_out = sorted(df.iloc[sp_test]["_group"].unique().tolist())
            fold_logs.append(
                {
                    "Dataset": dataset_name,
                    "Seed": seed,
                    "Fold": fold_id,
                    "HeldOutRegions": held_out,
                    "HeldOutRegionCount": len(held_out),
                    "TestSamples": int(len(sp_test)),
                }
            )
        spatial_seed_importance.append(np.mean(np.vstack(fold_importances), axis=0))

    random_arr = np.vstack(random_seed_importance)
    spatial_arr = np.vstack(spatial_seed_importance)
    table = pd.DataFrame(
        {
            "feature": predictor_cols,
            "rand_mean": random_arr.mean(axis=0),
            "rand_sd": random_arr.std(axis=0),
            "spat_mean": spatial_arr.mean(axis=0),
            "spat_sd": spatial_arr.std(axis=0),
        }
    ).sort_values("rand_mean", ascending=False)
    return table, fold_logs


def zambia_anomaly_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    anomaly_years = [2012, 2014]
    baseline = df.loc[~df["_year"].isin(anomaly_years), "_yield"]
    hist_mean = baseline.mean()
    hist_std = baseline.std()

    summary_rows = []
    for year in anomaly_years:
        year_df = df[df["_year"] == year]
        year_mean = year_df["_yield"].mean()
        z_score = (year_mean - hist_mean) / hist_std
        summary_rows.append(
            {
                "year": year,
                "historical_mean_excl_2012_2014": hist_mean,
                "historical_std_excl_2012_2014": hist_std,
                "year_mean": year_mean,
                "regions": int(year_df["_group"].nunique()),
                "records": int(len(year_df)),
                "z_score": z_score,
                "direction": "positive anomaly" if z_score > 0 else "negative anomaly",
            }
        )

    per_year = (
        df.groupby("_year")["_yield"]
        .mean()
        .reset_index()
        .rename(columns={"_year": "year", "_yield": "mean_yield"})
    )
    return pd.DataFrame(summary_rows), per_year


def wheat_yearly_temporal_breakdown(df: pd.DataFrame, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    years = df["_year"].to_numpy()
    unique_years = np.sort(np.unique(years))
    model_name = "Random Forest"
    rows = []

    for i in range(4, len(unique_years) - 1):
        train_years = unique_years[: i + 1]
        test_year = unique_years[i + 1]
        train_idx = np.where(np.isin(years, train_years))[0]
        test_idx = np.where(years == test_year)[0]
        if len(test_idx) < 5:
            continue
        preds = fit_predict(model_name, X, y, train_idx, test_idx)
        rows.append(
            {
                "year": int(test_year),
                "rmse": root_mean_squared_error(y[test_idx], preds),
                "r2": r2_score(y[test_idx], preds),
                "n": int(len(test_idx)),
            }
        )
    return pd.DataFrame(rows)


def wheat_2017_anomaly_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_year = 2017
    regions_2017 = sorted(df.loc[df["_year"] == target_year, "_group"].unique().tolist())

    region_rows = []
    for region in regions_2017:
        current = df[(df["_group"] == region) & (df["_year"] == target_year)]
        history = df[(df["_group"] == region) & (df["_year"] != target_year)]
        if len(current) != 1 or len(history) < 5:
            continue

        hist_mean = history["_yield"].mean()
        hist_std = history["_yield"].std()
        actual = float(current["_yield"].iloc[0])
        z_score = np.nan if hist_std == 0 else (actual - hist_mean) / hist_std
        country = current["country_code"].iloc[0] if "country_code" in current.columns else "NA"
        region_rows.append(
            {
                "country_code": country,
                "adm_id": region,
                "historical_mean_excl_2017": hist_mean,
                "historical_std_excl_2017": hist_std,
                "yield_2017": actual,
                "difference": actual - hist_mean,
                "z_score": z_score,
                "direction": "higher than history" if actual > hist_mean else "lower than history",
            }
        )

    region_df = pd.DataFrame(region_rows).sort_values(["country_code", "adm_id"])

    overall = pd.DataFrame(
        [
            {
                "regions_evaluated": int(len(region_df)),
                "historical_mean_excl_2017": region_df["historical_mean_excl_2017"].mean(),
                "historical_std_excl_2017": region_df["historical_std_excl_2017"].mean(),
                "yield_2017_mean": region_df["yield_2017"].mean(),
                "difference_mean": region_df["difference"].mean(),
                "mean_region_z_score": region_df["z_score"].mean(),
                "direction": "positive anomaly"
                if region_df["z_score"].mean() > 0
                else "negative anomaly",
            }
        ]
    )

    country_breakdown = (
        region_df.groupby("country_code")
        .agg(
            regions=("adm_id", "count"),
            historical_mean_excl_2017=("historical_mean_excl_2017", "mean"),
            yield_2017_mean=("yield_2017", "mean"),
            difference_mean=("difference", "mean"),
            mean_region_z_score=("z_score", "mean"),
        )
        .reset_index()
        .sort_values("country_code")
    )
    country_breakdown["direction"] = np.where(
        country_breakdown["mean_region_z_score"] > 0,
        "positive anomaly",
        "negative anomaly",
    )

    return overall, country_breakdown, region_df


def dataset_descriptives(cfg: dict, df: pd.DataFrame, predictor_cols: list[str]) -> dict:
    years = df["_year"]
    target = df["_yield"]
    predictors = ", ".join(predictor_cols)
    if len(predictors) > 140:
        predictors = predictors[:137] + "..."

    return {
        "Dataset": cfg["name"],
        "Regions": int(df["_group"].nunique()),
        "YearRange": f"{int(years.min())}-{int(years.max())}",
        "ValidRecords": int(len(df)),
        "PredictorCount": len(predictor_cols),
        "Predictors": predictors,
        "TargetMean": round(float(target.mean()), 3),
        "TargetSD": round(float(target.std()), 3),
        "TargetMin": round(float(target.min()), 3),
        "TargetMax": round(float(target.max()), 3),
    }


def dataframe_to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    if df.empty:
        return "_No rows generated._"
    table_df = df.copy()
    if index:
        table_df = table_df.reset_index()

    headers = [str(col) for col in table_df.columns]
    rows = [[str(value) for value in row] for row in table_df.to_numpy()]
    widths = []
    for col_idx, header in enumerate(headers):
        cell_width = max([len(header)] + [len(row[col_idx]) for row in rows])
        widths.append(cell_width)

    def format_row(values: list[str]) -> str:
        return "| " + " | ".join(
            value.ljust(widths[idx]) for idx, value in enumerate(values)
        ) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    lines = [format_row(headers), separator]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)


def write_markdown_package(
    main_results: pd.DataFrame,
    wilcoxon_df: pd.DataFrame,
    feature_tables: dict[str, pd.DataFrame],
    zambia_summary: pd.DataFrame,
    zambia_years: pd.DataFrame,
    wheat_temporal: pd.DataFrame,
    wheat_overall: pd.DataFrame,
    wheat_country: pd.DataFrame,
    descriptives: pd.DataFrame,
) -> None:
    outstanding = [
        "Could not find a separate local script or artifact that unambiguously defines the single-holdout pilot table. Not regenerated here.",
        "The rolling spatiotemporal figure/data were already present locally and are referenced as existing artifacts.",
        "Outstanding citations still needed: (a) external confirmation of 2018 Romania/Hungary above-average maize yield, (b) Kapoor & Narayanan 2023 Patterns full citation, (c) the 2026 '2,047 Benchmark Datasets' leakage landscape paper full citation.",
    ]

    summary_lines = [
        "# Paper Data Package",
        "",
        "## Flagged Summary",
        "",
        "- Genuine bug fixed in code path: spatial feature-importance seeds previously reused the same deterministic GroupKFold split, which can force spatial SD to 0.0000.",
        "- Genuine bug fixed in scope: Wheat 2017 anomaly diagnostic was missing and is now computed explicitly.",
        "- Coverage fix: Wilcoxon outputs are now generated for both 5-fold and 10-fold comparisons across all four datasets.",
        "- Uncertain legacy item: a distinct single-holdout pilot table could not be verified from local scripts/files, so it is flagged rather than invented.",
        "",
        "## 1. Main Results Tables",
        "",
        dataframe_to_markdown(main_results),
        "",
        "## 2. Wilcoxon Significance Tests",
        "",
        dataframe_to_markdown(wilcoxon_df),
        "",
        "## 3. Feature Importance Stability (Top 8 by random-mean importance)",
        "",
    ]

    for dataset_name, table in feature_tables.items():
        summary_lines.append(f"### {dataset_name}")
        summary_lines.append("")
        summary_lines.append(dataframe_to_markdown(table.head(8)))
        summary_lines.append("")

    summary_lines.extend(
        [
            "## 4. Zambia 2012 and 2014 Anomaly Diagnostics",
            "",
            dataframe_to_markdown(zambia_summary),
            "",
            dataframe_to_markdown(zambia_years),
            "",
            "## 5. Wheat 2017 Diagnostics",
            "",
            "### Overall",
            "",
            dataframe_to_markdown(wheat_overall),
            "",
            "### By Country",
            "",
            dataframe_to_markdown(wheat_country),
            "",
            "## 6. Wheat Temporal Walk-Forward Yearly Breakdown",
            "",
            dataframe_to_markdown(wheat_temporal),
            "",
            "## 7. Dataset Descriptive Statistics",
            "",
            dataframe_to_markdown(descriptives),
            "",
            "## 8. Existing Rolling Spatiotemporal Artifacts",
            "",
            "- `rolling_st_CY-Bench_Maize_Europe.csv`",
            "- `rolling_st_CY-Bench_Wheat_Europe.csv`",
            "- `rolling_st_CY-Bench_Maize_Zambia.csv`",
            "- `rolling_st_SustainBench_Soybean.csv`",
            "- `figures/figure_rolling_spatiotemporal.png`",
            "",
            "## 9. Outstanding / Not Verified",
            "",
        ]
    )
    summary_lines.extend([f"- {line}" for line in outstanding])
    summary_lines.append("")

    (ROOT / "paper_data_package.md").write_text("\n".join(summary_lines), encoding="utf-8")


def main() -> None:
    print("=" * 72)
    print("Loading datasets and regenerating package artifacts")
    print("=" * 72)

    loaded = {}
    main_result_frames = []
    descriptive_rows = []

    for cfg in DATASETS:
        df, X, y, predictor_cols = load_dataset(cfg)
        loaded[cfg["name"]] = {
            "cfg": cfg,
            "df": df,
            "X": X,
            "y": y,
            "predictor_cols": predictor_cols,
        }
        print(
            f"{cfg['name']}: rows={len(df)}, regions={df['_group'].nunique()}, "
            f"years={df['_year'].min()}-{df['_year'].max()}, predictors={len(predictor_cols)}"
        )
        main_result_frames.append(evaluate_main_results(df, X, y, cfg["name"]))
        descriptive_rows.append(dataset_descriptives(cfg, df, predictor_cols))

    main_results = pd.concat(main_result_frames, ignore_index=True)
    main_results.to_csv(ROOT / "all_dataset_results.csv", index=False)
    pd.DataFrame(descriptive_rows).to_csv(ROOT / "dataset_descriptives.csv", index=False)

    print("\n" + "=" * 72)
    print("Section A: Wilcoxon tests")
    print("=" * 72)
    wilcoxon_rows = []
    for dataset_name, bundle in loaded.items():
        for folds in (5, 10):
            result = wilcoxon_random_vs_spatial(
                bundle["df"], bundle["X"], bundle["y"], n_splits=folds
            )
            print(
                f"{dataset_name} ({folds}-fold): stat={result['Statistic']:.3f}, "
                f"p={result['PValue']:.4f}, worse={result['SpatialWorseFolds']}/{folds}"
            )
            wilcoxon_rows.append(
                {
                    "Dataset": dataset_name,
                    "Folds": folds,
                    "SpatialWorseFolds": result["SpatialWorseFolds"],
                    "Statistic": result["Statistic"],
                    "PValue": result["PValue"],
                    "WPlus": result["WPlus"],
                    "WMinus": result["WMinus"],
                    "RankBiserialR": result["RankBiserialR"],
                    "RandomRMSEPerFold": json.dumps(np.round(result["Random_RMSE"], 6).tolist()),
                    "SpatialRMSEPerFold": json.dumps(np.round(result["Spatial_RMSE"], 6).tolist()),
                    "Diffs": json.dumps(np.round(result["Diffs"], 6).tolist()),
                }
            )
    wilcoxon_df = pd.DataFrame(wilcoxon_rows)
    wilcoxon_df.to_csv(ROOT / "wilcoxon_all_datasets.csv", index=False)

    print("\n" + "=" * 72)
    print("Section B: Feature importance stability")
    print("=" * 72)
    feature_tables = {}
    fold_logs = []
    for dataset_name in FEATURE_IMPORTANCE_DATASETS:
        bundle = loaded[dataset_name]
        table, logs = compute_feature_importance_table(
            bundle["df"],
            bundle["X"],
            bundle["y"],
            bundle["predictor_cols"],
            dataset_name,
        )
        feature_tables[dataset_name] = table
        fold_logs.extend(logs)
        out_name = f"feat_importance_{sanitize(dataset_name)}.csv"
        table.to_csv(ROOT / out_name, index=False)
        print(
            f"{dataset_name}: top spatial SD={table['spat_sd'].max():.6f}, "
            f"nonzero spatial SD count={(table['spat_sd'] > 0).sum()}"
        )
    pd.DataFrame(fold_logs).to_csv(ROOT / "feature_importance_spatial_fold_logs.csv", index=False)

    print("\n" + "=" * 72)
    print("Section C: Zambia anomaly diagnostics")
    print("=" * 72)
    zambia_bundle = loaded["CY-Bench Maize (Zambia)"]
    zambia_summary, zambia_years = zambia_anomaly_analysis(zambia_bundle["df"])
    zambia_summary.to_csv(ROOT / "zambia_anomaly_summary.csv", index=False)
    zambia_years.to_csv(ROOT / "zambia_yearly_means.csv", index=False)
    print(zambia_summary.to_string(index=False))

    print("\n" + "=" * 72)
    print("Section D: Wheat temporal breakdown and 2017 anomaly")
    print("=" * 72)
    wheat_bundle = loaded["CY-Bench Wheat (Europe)"]
    wheat_temporal = wheat_yearly_temporal_breakdown(
        wheat_bundle["df"], wheat_bundle["X"], wheat_bundle["y"]
    )
    wheat_temporal.to_csv(ROOT / "wheat_temporal_yearly.csv", index=False)
    wheat_overall, wheat_country, wheat_regions = wheat_2017_anomaly_analysis(wheat_bundle["df"])
    wheat_overall.to_csv(ROOT / "wheat_2017_anomaly_overall.csv", index=False)
    wheat_country.to_csv(ROOT / "wheat_2017_country_breakdown.csv", index=False)
    wheat_regions.to_csv(ROOT / "wheat_2017_region_breakdown.csv", index=False)
    print(wheat_overall.to_string(index=False))

    print("\n" + "=" * 72)
    print("Section E: Writing consolidated markdown package")
    print("=" * 72)
    write_markdown_package(
        main_results=main_results,
        wilcoxon_df=wilcoxon_df[
            [
                "Dataset",
                "Folds",
                "SpatialWorseFolds",
                "Statistic",
                "PValue",
                "WPlus",
                "WMinus",
                "RankBiserialR",
            ]
        ],
        feature_tables=feature_tables,
        zambia_summary=zambia_summary,
        zambia_years=zambia_years,
        wheat_temporal=wheat_temporal,
        wheat_overall=wheat_overall,
        wheat_country=wheat_country,
        descriptives=pd.DataFrame(descriptive_rows),
    )

    print("\nGenerated files:")
    for path in [
        "all_dataset_results.csv",
        "dataset_descriptives.csv",
        "wilcoxon_all_datasets.csv",
        "feature_importance_spatial_fold_logs.csv",
        "zambia_anomaly_summary.csv",
        "zambia_yearly_means.csv",
        "wheat_temporal_yearly.csv",
        "wheat_2017_anomaly_overall.csv",
        "wheat_2017_country_breakdown.csv",
        "wheat_2017_region_breakdown.csv",
        "paper_data_package.md",
    ]:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
