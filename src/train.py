"""
src/train.py
-------------
data/processed/features_engineered.parquet dosyasını kullanarak CS2 harita-kazanan
tahmin modelini eğitir.

Adımlar:
    1. KRONOLOJİK split (asla rastgele split değil — geçmişle geleceği eğit/test
       arasında karıştırmak, zaman serisi verisinde en klasik leakage kaynağıdır)
    2. XGBoost ve LightGBM'i aynı split üzerinde karşılaştırır (+ Logistic Regression
       baseline, "modelim gerçekten bir şey öğreniyor mu" sağlaması için)
    3. Kazanan modeli isotonic regression ile KALİBRE eder (ham ağaç modeli olasılıkları
       genelde aşırı-güvenli/az-güvenli olur; API'de "win probability" göstereceğimiz
       için kalibrasyon şart)
    4. Feature importance çıkarır ve görselleştirir
    5. Modeli + metadata'yı models/ klasörüne kaydeder (predict.py ve api/main.py
       bunları doğrudan yükleyecek)

Kullanım:
    python -m src.train --features-path data/processed/features_engineered.parquet --output-dir models
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb

matplotlib.use("Agg")  # PyCharm/headless ortamda güvenli backend
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


# --------------------------------------------------------------------------- #
# Konfigürasyon
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    features_path: Path
    output_dir: Path
    train_frac: float = 0.70
    val_frac: float = 0.15
    # test_frac = 1 - train_frac - val_frac

    target_col: str = "team1_map_win"
    tune_hyperparams: bool = True
    optuna_n_trials: int = 40
    optuna_n_splits: int = 4

    numeric_features: tuple = (
        "team1_form_last5", "team1_form_last10", "team1_form_career",
        "team2_form_last5", "team2_form_last10", "team2_form_career",
        "team1_h2h_win_rate", "team1_h2h_matches_played",
        "team1_map_win_rate", "team1_map_experience",
        "team2_map_win_rate", "team2_map_experience",
        "team1_team_adr_form", "team1_team_kast_form", "team1_team_kddiff_form",
        "team2_team_adr_form", "team2_team_kast_form", "team2_team_kddiff_form",
        "team1_roster_avg_experience", "team2_roster_avg_experience",
        "team1_elo", "team2_elo", "team1_rest_days", "team2_rest_days",
        "team1_streak", "team2_streak",
        "diff_form_last5", "diff_form_last10", "diff_form_career",
        "diff_map_win_rate", "diff_h2h_win_rate",
        "diff_team_adr_form", "diff_team_kast_form", "diff_team_kddiff_form",
        "diff_elo", "diff_rest_days", "diff_streak",
    )
    categorical_features: tuple = ("map_name", "tier", "bestOf")


# --------------------------------------------------------------------------- #
# Veri hazırlama
# --------------------------------------------------------------------------- #
def load_and_prepare(cfg: TrainConfig) -> pd.DataFrame:
    df = pd.read_parquet(cfg.features_path)
    df = df.sort_values("datetime").reset_index(drop=True)

    # Sadece pipeline.py'de gerçekten üretilen kolonları kullan (savunmacı kontrol)
    missing_num = [c for c in cfg.numeric_features if c not in df.columns]
    if missing_num:
        logger.warning("Beklenen numeric feature'lar eksik, atlanacak: %s", missing_num)

    df["bestOf"] = df["bestOf"].fillna(-1).astype(int).astype(str)
    df["tier"] = df["tier"].astype(str)
    df["map_name"] = df["map_name"].astype(str)

    return df


def chronological_split(df: pd.DataFrame, cfg: TrainConfig):
    n = len(df)
    train_end = int(n * cfg.train_frac)
    val_end = int(n * (cfg.train_frac + cfg.val_frac))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    logger.info(
        "Split -> train: %d satır (%s -> %s) | val: %d satır (%s -> %s) | test: %d satır (%s -> %s)",
        len(train_df), train_df["datetime"].min().date(), train_df["datetime"].max().date(),
        len(val_df), val_df["datetime"].min().date(), val_df["datetime"].max().date(),
        len(test_df), test_df["datetime"].min().date(), test_df["datetime"].max().date(),
    )
    return train_df, val_df, test_df


def get_xy(df: pd.DataFrame, cfg: TrainConfig, feature_cols: list):
    X = df[feature_cols].copy()
    for c in cfg.categorical_features:
        if c in X.columns:
            X[c] = X[c].astype("category")
    y = df[cfg.target_col].astype(int)
    return X, y


# --------------------------------------------------------------------------- #
# Hiperparametre optimizasyonu (Optuna, ZAMAN-SERİSİ uyumlu CV)
# --------------------------------------------------------------------------- #
def tune_xgboost_optuna(X_train: pd.DataFrame, y_train: pd.Series, n_trials: int, n_splits: int) -> dict:
    """XGBoost hiperparametrelerini Optuna ile arar.

    KRİTİK: sklearn'ün varsayılan KFold'u RASTGELE böler -- zaman serisi verisinde bu,
    geleceğin geçmişi tahmin etmek için kullanılmasına (leakage) yol açar. Bunun yerine
    TimeSeriesSplit kullanıyoruz: her fold'da eğitim seti SADECE validasyon setinden
    KRONOLOJİK OLARAK ÖNCEKİ verilerden oluşur (expanding window).
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        }
        fold_losses = []
        for fold_train_idx, fold_val_idx in tscv.split(X_train):
            X_tr, X_va = X_train.iloc[fold_train_idx], X_train.iloc[fold_val_idx]
            y_tr, y_va = y_train.iloc[fold_train_idx], y_train.iloc[fold_val_idx]
            model = xgb.XGBClassifier(
                **params, eval_metric="logloss", enable_categorical=True,
                tree_method="hist", random_state=42,
            )
            model.fit(X_tr, y_tr)
            proba = model.predict_proba(X_va)[:, 1]
            fold_losses.append(log_loss(y_va, proba))
        return float(np.mean(fold_losses))

    study = optuna.create_study(direction="minimize", study_name="xgboost_cs2")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info("Optuna tamamlandı -> %d deneme, en iyi ortalama fold LogLoss: %.4f",
                n_trials, study.best_value)
    logger.info("En iyi hiperparametreler: %s", study.best_params)
    return study.best_params


# --------------------------------------------------------------------------- #
# Model eğitimi
# --------------------------------------------------------------------------- #
def train_xgboost(X_train, y_train, X_val, y_val, params: dict | None = None) -> xgb.XGBClassifier:
    default_params = dict(
        n_estimators=500, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    )
    final_params = {**default_params, **(params or {})}
    model = xgb.XGBClassifier(
        **final_params,
        eval_metric="logloss",
        enable_categorical=True,
        tree_method="hist",
        early_stopping_rounds=30,
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    logger.info("XGBoost eğitildi -> best_iteration: %s", model.best_iteration)
    return model


def train_lightgbm(X_train, y_train, X_val, y_val, cat_cols) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        verbosity=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        categorical_feature=[c for c in cat_cols if c in X_train.columns],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    logger.info("LightGBM eğitildi -> best_iteration: %s", model.best_iteration_)
    return model


def train_logistic_baseline(X_train, y_train, numeric_cols):
    """Sadece numerik feature'larla basit bir baseline -> ağaç modellerinin gerçekten
    ekstra sinyal yakalayıp yakalamadığını anlamak için sağlama (sanity check)."""
    X_num = X_train[numeric_cols].fillna(0.5)
    scaler = StandardScaler().fit(X_num)
    model = LogisticRegression(max_iter=1000)
    model.fit(scaler.transform(X_num), y_train)
    return model, scaler


# --------------------------------------------------------------------------- #
# Değerlendirme
# --------------------------------------------------------------------------- #
def evaluate(name: str, y_true, y_proba) -> dict:
    metrics = {
        "model": name,
        "roc_auc": roc_auc_score(y_true, y_proba),
        "log_loss": log_loss(y_true, y_proba),
        "brier_score": brier_score_loss(y_true, y_proba),
        "accuracy_at_0.5": float(((y_proba >= 0.5).astype(int) == y_true).mean()),
    }
    logger.info(
        "[%s] ROC-AUC: %.4f | LogLoss: %.4f | Brier: %.4f | Acc@0.5: %.4f",
        name, metrics["roc_auc"], metrics["log_loss"], metrics["brier_score"], metrics["accuracy_at_0.5"],
    )
    return metrics


# --------------------------------------------------------------------------- #
# Kalibrasyon
# --------------------------------------------------------------------------- #
def calibrate_model(model, method: str, X_val, y_val):
    """sklearn >=1.6 'cv=prefit' parametresini kaldırıp yerine FrozenEstimator sarmalayıcısını
    getirdi. Eski sürümlerle de çalışması için iki yolu da deniyoruz."""
    try:
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=method)
        calibrated.fit(X_val, y_val)
    except ImportError:
        calibrated = CalibratedClassifierCV(model, method=method, cv="prefit")
        calibrated.fit(X_val, y_val)
    return calibrated


def select_best_calibration(model, X_val, y_val, min_isotonic_samples: int = 5000):
    """Isotonic ve sigmoid (Platt) kalibrasyonunu VALİDASYON setinde karşılaştırıp
    en iyisini seçer -- test setine bu karar aşamasında hiç dokunulmaz.

    ÖNEMLİ: isotonic regression, küçük validation setlerinde (kabaca <5000 satır)
    "basamaklı" (piecewise-constant/step) bir eğriye yakınsama eğilimindedir. Bu,
    validation log-loss'ta iyi görünse bile, üretimde çok farklı maçların aynı
    olasılık değerine yuvarlanmasına (çözünürlük kaybı) yol açar -- bu da API
    kullanıcısına anlamsız/güvensiz görünür. Bu yüzden validation seti yeterince
    büyük değilse isotonic aday listesinden tamamen çıkarılır.
    """
    raw_val_proba = model.predict_proba(X_val)[:, 1]
    candidates = {"raw": (model, log_loss(y_val, raw_val_proba))}

    methods = ["sigmoid"]
    if len(y_val) >= min_isotonic_samples:
        methods.append("isotonic")
    else:
        logger.warning(
            "Validation seti çok küçük (%d < %d) -> isotonic kalibrasyon degenerate "
            "(basamaklı) sonuç riski taşıdığı için aday listesinden çıkarıldı.",
            len(y_val), min_isotonic_samples,
        )

    for method in methods:
        cal_model = calibrate_model(model, method, X_val, y_val)
        val_proba = cal_model.predict_proba(X_val)[:, 1]
        candidates[method] = (cal_model, log_loss(y_val, val_proba))

    for name, (_, ll) in candidates.items():
        logger.info("  [Validation] %-10s -> LogLoss: %.4f", name, ll)

    best_name = min(candidates, key=lambda k: candidates[k][1])
    logger.info("Validation setinde en iyi kalibrasyon stratejisi: '%s'", best_name)
    return best_name, candidates[best_name][0]


def plot_calibration_curve(y_true, proba_raw, proba_calibrated, output_path: Path):
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, proba in (("Kalibrasyon öncesi", proba_raw), ("Kalibrasyon sonrası", proba_calibrated)):
        frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Mükemmel kalibrasyon")
    ax.set_xlabel("Tahmin edilen olasılık")
    ax.set_ylabel("Gerçekleşen galibiyet oranı")
    ax.set_title("Kalibrasyon Eğrisi (Test Seti)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info("Kalibrasyon grafiği kaydedildi -> %s", output_path)


def plot_feature_importance(model, feature_names, output_path: Path, top_n: int = 20):
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh([feature_names[i] for i in order][::-1], importances[order][::-1])
    ax.set_xlabel("Önem (gain)")
    ax.set_title("Feature Importance (İlk %d)" % top_n)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info("Feature importance grafiği kaydedildi -> %s", output_path)


# --------------------------------------------------------------------------- #
# Ana pipeline
# --------------------------------------------------------------------------- #
def run(cfg: TrainConfig) -> None:
    df = load_and_prepare(cfg)
    train_df, val_df, test_df = chronological_split(df, cfg)

    feature_cols = [c for c in cfg.numeric_features if c in df.columns] + list(cfg.categorical_features)

    X_train, y_train = get_xy(train_df, cfg, feature_cols)
    X_val, y_val = get_xy(val_df, cfg, feature_cols)
    X_test, y_test = get_xy(test_df, cfg, feature_cols)

    logger.info("Sınıf dengesi -> train: %.3f | val: %.3f | test: %.3f",
                y_train.mean(), y_val.mean(), y_test.mean())

    # ---- Baseline: Logistic Regression (sadece numerik feature'larla) ----
    numeric_cols = [c for c in cfg.numeric_features if c in df.columns]
    log_model, scaler = train_logistic_baseline(X_train, y_train, numeric_cols)
    logreg_proba = log_model.predict_proba(scaler.transform(X_test[numeric_cols].fillna(0.5)))[:, 1]
    results = [evaluate("Logistic Regression (baseline)", y_test, logreg_proba)]

    # ---- XGBoost (varsayılan hiperparametreler) ----
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    results.append(evaluate("XGBoost (varsayılan)", y_test, xgb_proba))

    # ---- XGBoost (Optuna ile ayarlanmış hiperparametreler) ----
    if cfg.tune_hyperparams:
        logger.info("--- Optuna hiperparametre araması başlıyor (%d deneme, %d fold, zaman-serisi CV) ---",
                    cfg.optuna_n_trials, cfg.optuna_n_splits)
        best_params = tune_xgboost_optuna(X_train, y_train, cfg.optuna_n_trials, cfg.optuna_n_splits)
        xgb_tuned_model = train_xgboost(X_train, y_train, X_val, y_val, params=best_params)
        xgb_tuned_proba = xgb_tuned_model.predict_proba(X_test)[:, 1]
        results.append(evaluate("XGBoost (Optuna-tuned)", y_test, xgb_tuned_proba))
    else:
        xgb_tuned_model = None

    # ---- LightGBM ----
    lgb_model = train_lightgbm(X_train, y_train, X_val, y_val, list(cfg.categorical_features))
    lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
    results.append(evaluate("LightGBM", y_test, lgb_proba))

    results_df = pd.DataFrame(results).set_index("model")
    logger.info("\n--- Model karşılaştırma tablosu (test seti) ---\n%s", results_df.round(4).to_string())

    # ---- En iyi modeli seç (log_loss'a göre) ----
    best_name = results_df["log_loss"].idxmin()
    candidate_models = {
        "XGBoost (varsayılan)": xgb_model, "LightGBM": lgb_model,
    }
    if xgb_tuned_model is not None:
        candidate_models["XGBoost (Optuna-tuned)"] = xgb_tuned_model
    best_raw_model = candidate_models.get(best_name, xgb_model)
    logger.info("Kazanan model: %s", best_name)

    # ---- Kalibrasyon (seçim validation'da yapılır, test setine dokunulmaz) ----
    logger.info("--- Kalibrasyon stratejisi seçiliyor (isotonic vs sigmoid vs ham) ---")
    calib_strategy, calibrated_model = select_best_calibration(best_raw_model, X_val, y_val)
    calibrated_proba = calibrated_model.predict_proba(X_test)[:, 1]
    raw_best_proba = best_raw_model.predict_proba(X_test)[:, 1]

    logger.info("--- Final karşılaştırma (%s, TEST seti, kalibrasyon kararı görmeden) ---", best_name)
    evaluate(f"{best_name} (ham)", y_test, raw_best_proba)
    evaluate(f"{best_name} (final, strateji={calib_strategy})", y_test, calibrated_proba)

    # ---- Kayıt ----
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(calibrated_model, cfg.output_dir / "champion_model.pkl")
    joblib.dump(best_raw_model, cfg.output_dir / "champion_model_raw.pkl")

    metadata = {
        "champion_model_name": best_name,
        "calibration_strategy": calib_strategy,
        "feature_cols": feature_cols,
        "numeric_features": numeric_cols,
        "categorical_features": list(cfg.categorical_features),
        "target_col": cfg.target_col,
        "train_date_range": [str(train_df["datetime"].min()), str(train_df["datetime"].max())],
        "test_date_range": [str(test_df["datetime"].min()), str(test_df["datetime"].max())],
        "test_metrics_calibrated": evaluate(f"{best_name} (final)", y_test, calibrated_proba),
    }
    with open(cfg.output_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info("Metadata kaydedildi -> %s", cfg.output_dir / "model_metadata.json")

    plot_calibration_curve(y_test, raw_best_proba, calibrated_proba, cfg.output_dir / "calibration_curve.png")

    if best_name in candidate_models:
        plot_feature_importance(best_raw_model, feature_cols, cfg.output_dir / "feature_importance.png")

    logger.info("Eğitim tamamlandı. Tüm çıktılar -> %s", cfg.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CS2 harita-kazanan tahmin modeli eğitimi.")
    parser.add_argument("--features-path", type=Path, default=Path("data/processed/features_engineered.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(features_path=args.features_path, output_dir=args.output_dir)
    run(cfg)


if __name__ == "__main__":
    main()