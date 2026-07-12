"""
src/pipeline.py
-----------------
Harita-seviyesi (maps_clean.parquet) ve maç-seviyesi (matches_summary.parquet) temiz
verilerden, model eğitimi için 4 grup feature üretir:

    1. Team Form       -> son 5/10 maçtaki galibiyet oranı
    2. Map Advantage    -> takımın o spesifik haritadaki tarihsel galibiyet oranı
    3. Head-to-Head     -> iki takımın birbirine karşı tarihsel üstünlüğü
    4. Player Firepower -> başlangıç kadrosunun son maçlardaki ortalama ADR/KAST/KDDIFF'i

KRİTİK TASARIM KARARI (DATA LEAKAGE ÖNLEME):
    EDA'da doğrulandığı üzere, aynı match_id'ye ait tüm harita satırları BİREBİR AYNI
    datetime damgasını taşıyor (bir Bo3/Bo5'in tüm haritaları aynı gün/saat gösteriliyor).
    Bu yüzden feature'lar ham datetime sıralamasıyla hesaplanırsa, örneğin bir serinin
    1. haritasının sonucu 2. haritayı tahmin ederken sızabilir.

    Çözüm: Tüm rolling/expanding istatistikler ÖNCE MAÇ SEVİYESİNDE (match_id bazında,
    haritalar birleştirilmiş halde) hesaplanır, ve her maç için hesaplama SADECE o maçtan
    KRONOLOJİK OLARAK ÖNCEKİ maçları kullanır (mevcut match_id'nin kendisi ve aynı ana ait
    hiçbir harita dahil edilmez). Elde edilen "maç öncesi" feature değeri, o maça ait TÜM
    harita satırlarına aynen kopyalanır. Böylece bir Bo3'ün 2. haritası, 1. haritanın
    sonucunu "bilerek" tahmin edilmiş olmaz.

Çıktı:
    data/processed/features_engineered.parquet
    -> harita-seviyesi tablo + team1_*/team2_* feature'ları + fark (diff) kolonları
       + hedef değişken team1_map_win
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


@dataclass
class FeatureConfig:
    processed_dir: Path
    output_dir: Path
    form_windows: tuple = (5, 10)          # Team Form için rolling pencere boyutları
    player_form_window: int = 5            # Oyuncu firepower için rolling pencere boyutu
    map_advantage_min_history: int = 1     # Map advantage için min. geçmiş maç sayısı
    global_prior_win_rate: float = 0.5     # Hiç geçmişi olmayan takım/eşleşme için varsayılan


# --------------------------------------------------------------------------- #
# Yardımcı: match_id seviyesinde uzun (long) format inşası
# --------------------------------------------------------------------------- #
def build_team_match_long(matches: pd.DataFrame) -> pd.DataFrame:
    """Her maç için iki perspektif satırı üretir (team1 açısından, team2 açısından).

    NOT: Grup/eşleştirme anahtarı team_id DEĞİL team_name'dir; team_id bu veri
    setinde maç bazında değişen güvenilmez bir kolondur (bkz. eda_cleaning.py).
    team_id yine de referans için tutulur ama hiçbir groupby/join anahtarında
    kullanılmaz.
    """
    a = matches.rename(columns={
        "team1_id": "team_id", "team2_id": "opponent_id",
        "team1": "team_name", "team2": "opponent_name",
        "team1_match_win": "match_win",
    })[["match_id", "datetime", "tier", "team_id", "opponent_id", "team_name", "opponent_name", "match_win"]]

    b = matches.rename(columns={
        "team2_id": "team_id", "team1_id": "opponent_id",
        "team2": "team_name", "team1": "opponent_name",
    })
    b["match_win"] = 1 - matches["team1_match_win"]
    b = b[["match_id", "datetime", "tier", "team_id", "opponent_id", "team_name", "opponent_name", "match_win"]]

    long_df = pd.concat([a, b], ignore_index=True)
    return long_df.sort_values(["team_name", "datetime"]).reset_index(drop=True)


def build_team_map_match_agg(maps: pd.DataFrame) -> pd.DataFrame:
    """Harita satırlarını match_id + takım bazında özetler (map advantage feature'ı için
    her (team_name, map_name) çiftinin maç-seviyesi galibiyet geçmişini oluşturur).

    Grup anahtarı team_name'dir (bkz. build_team_match_long'daki not: team_id güvenilmez).
    """
    rows = []
    for side in ("team1", "team2"):
        sub = maps[["match_id", "datetime", "map_name", side]].copy()
        sub = sub.rename(columns={side: "team_name"})
        sub["map_win"] = maps["team1_map_win"] if side == "team1" else (1 - maps["team1_map_win"])
        rows.append(sub)
    long_df = pd.concat(rows, ignore_index=True)

    # Nadiren aynı maçta aynı harita teknik nedenlerle (restart) iki kez oynanmış olabilir.
    # Map-advantage feature'ı (match_id, team_name, map_name) bazında BENZERSİZ anahtar
    # gerektirir; bu durumda o maçtaki performansı ortalayarak tek satıra indirgiyoruz.
    before = len(long_df)
    long_df = long_df.groupby(["match_id", "team_name", "map_name", "datetime"], as_index=False)["map_win"].mean()
    if before != len(long_df):
        logger.warning("Aynı maç içinde tekrar oynanan harita(lar) nedeniyle %d satır birleştirildi.", before - len(long_df))

    return long_df.sort_values(["team_name", "map_name", "datetime"]).reset_index(drop=True)


def build_player_match_agg(maps: pd.DataFrame) -> pd.DataFrame:
    """Her (match_id, team_id, player_id) için o maçtaki ortalama performansı çıkarır
    (bir maçta birden fazla harita oynanmışsa performans o maç içinde ortalanır)."""
    frames = []
    for side in ("team1", "team2"):
        for p in range(1, 6):
            cols = {
                "match_id": "match_id",
                "datetime": "datetime",
                f"{side}_id": "team_id",
                f"{side}_player{p}_id": "player_id",
                f"{side}_player{p}_kills": "kills",
                f"{side}_player{p}_deaths": "deaths",
                f"{side}_player{p}_assists": "assists",
                f"{side}_player{p}_adr": "adr",
                f"{side}_player{p}_kast": "kast",
                f"{side}_player{p}_kddiff": "kddiff",
            }
            sub = maps[list(cols.keys())].rename(columns=cols)
            frames.append(sub)

    long_df = pd.concat(frames, ignore_index=True)
    long_df = long_df.dropna(subset=["player_id"])
    long_df["player_id"] = long_df["player_id"].astype(int)

    # Aynı maçta birden fazla harita oynandıysa -> o maç için oyuncu performansını ortala
    match_level = (
        long_df.groupby(["match_id", "datetime", "team_id", "player_id"], as_index=False)
        .agg(adr=("adr", "mean"), kast=("kast", "mean"), kddiff=("kddiff", "mean"))
    )
    return match_level.sort_values(["player_id", "datetime"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Feature 1: Team Form (rolling win rate, shifted -> leakage-free)
# --------------------------------------------------------------------------- #
def add_team_form(team_long: pd.DataFrame, windows: tuple) -> pd.DataFrame:
    team_long = team_long.sort_values(["team_name", "datetime"]).copy()
    grp = team_long.groupby("team_name")["match_win"]
    for w in windows:
        # shift(1): mevcut maçın SONUCUNU dahil etmeden, ondan önceki w maça bakar
        team_long[f"form_last{w}"] = grp.transform(
            lambda s: s.shift(1).rolling(window=w, min_periods=1).mean()
        )
    # Kariyerin tamamı boyunca genel galibiyet oranı (expanding, shifted)
    team_long["form_career"] = grp.transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    return team_long


# --------------------------------------------------------------------------- #
# Feature 2: Map Advantage (expanding win rate on that map, shifted)
# --------------------------------------------------------------------------- #
def add_map_advantage(team_map_long: pd.DataFrame) -> pd.DataFrame:
    team_map_long = team_map_long.sort_values(["team_name", "map_name", "datetime"]).copy()
    grp = team_map_long.groupby(["team_name", "map_name"])["map_win"]
    team_map_long["map_win_rate"] = grp.transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    team_map_long["map_experience"] = grp.transform(
        lambda s: s.shift(1).expanding(min_periods=1).count()
    )
    return team_map_long


# --------------------------------------------------------------------------- #
# Feature 3: Head-to-Head (expanding win rate against THIS specific opponent, shifted)
# --------------------------------------------------------------------------- #
def add_h2h(team_long: pd.DataFrame) -> pd.DataFrame:
    team_long = team_long.sort_values(["team_name", "opponent_name", "datetime"]).copy()
    grp = team_long.groupby(["team_name", "opponent_name"])["match_win"]
    team_long["h2h_win_rate"] = grp.transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    team_long["h2h_matches_played"] = grp.transform(lambda s: s.shift(1).expanding(min_periods=1).count())
    return team_long


# --------------------------------------------------------------------------- #
# Feature 4: Player Firepower (rolling avg ADR/KAST/KDDIFF per player, shifted)
# --------------------------------------------------------------------------- #
def add_player_rolling_form(player_match: pd.DataFrame, window: int) -> pd.DataFrame:
    player_match = player_match.sort_values(["player_id", "datetime"]).copy()
    grp = player_match.groupby("player_id")
    for col in ("adr", "kast", "kddiff"):
        player_match[f"{col}_roll"] = grp[col].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
        )
    return player_match


def aggregate_team_firepower(player_match_with_roll: pd.DataFrame) -> pd.DataFrame:
    """Her (match_id, team_id) için kadronun ortalama rolling ADR/KAST/KDDIFF'ini üretir."""
    team_fp = (
        player_match_with_roll.groupby(["match_id", "team_id"], as_index=False)
        .agg(
            team_adr_form=("adr_roll", "mean"),
            team_kast_form=("kast_roll", "mean"),
            team_kddiff_form=("kddiff_roll", "mean"),
            roster_avg_experience=("adr_roll", "count"),  # kaç oyuncunun geçmiş verisi vardı
        )
    )
    return team_fp


# --------------------------------------------------------------------------- #
# Her şeyi maç seviyesinde birleştirip harita satırlarına yayma
# --------------------------------------------------------------------------- #
def assemble_match_level_features(
    matches: pd.DataFrame,
    team_long: pd.DataFrame,
    team_map_long: pd.DataFrame,
    team_fp: pd.DataFrame,
    cfg: FeatureConfig,
) -> pd.DataFrame:
    """team1_* / team2_* olarak maç başına tek satırlık feature tablosu üretir."""

    form_cols = [c for c in team_long.columns if c.startswith("form_")] + ["h2h_win_rate", "h2h_matches_played"]
    team_side = team_long[["match_id", "team_name"] + form_cols]

    out = matches[["match_id", "datetime", "tier", "team1_id", "team2_id", "team1", "team2", "team1_match_win"]].copy()

    for side, id_col, name_col in (("team1", "team1_id", "team1"), ("team2", "team2_id", "team2")):
        # Rolling form / H2H: team_name bazlı join (team_id güvenilmez, bkz. yukarıdaki not)
        merged = out[["match_id", name_col]].merge(
            team_side, left_on=["match_id", name_col], right_on=["match_id", "team_name"], how="left"
        )
        for c in form_cols:
            out[f"{side}_{c}"] = merged[c].values

        # Player firepower: team_id burada sadece AYNI MAÇ içinde team1/team2 tarafını
        # ayırt etmek için kullanılıyor (bu kapsamda güvenli, çünkü çapraz-maç
        # karşılaştırma yapmıyoruz).
        fp = out[["match_id", id_col]].merge(
            team_fp, left_on=["match_id", id_col], right_on=["match_id", "team_id"], how="left"
        )
        for c in ("team_adr_form", "team_kast_form", "team_kddiff_form", "roster_avg_experience"):
            out[f"{side}_{c}"] = fp[c].values

    # Map advantage, harita bazlı olduğu için haritalar-seviyesinde birleştirilecek (aşağıda)
    return out


def attach_map_features(maps: pd.DataFrame, match_features: pd.DataFrame, team_map_long: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    drop_cols = [c for c in ("tier", "team1_match_win", "team1", "team2") if c in match_features.columns]
    df = maps.merge(match_features.drop(columns=drop_cols), on=["match_id", "datetime"], how="left", suffixes=("", "_mf"))

    for side in ("team1", "team2"):
        key = maps[["match_id", "map_name", side]].rename(columns={side: "team_name"})
        merged = key.merge(team_map_long, on=["match_id", "team_name", "map_name"], how="left")
        df[f"{side}_map_win_rate"] = merged["map_win_rate"].values
        df[f"{side}_map_experience"] = merged["map_experience"].values

    # Eksik geçmiş -> global prior ile doldur (soğuk başlangıç / yeni takım problemi)
    fill_cols = [c for c in df.columns if c.endswith(("_win_rate", "form_career",
                                                        "form_last5", "form_last10", "h2h_win_rate"))]
    for c in fill_cols:
        df[c] = df[c].fillna(cfg.global_prior_win_rate)

    count_cols = [c for c in df.columns if c.endswith(("_experience", "_matches_played"))]
    for c in count_cols:
        df[c] = df[c].fillna(0)

    # Oyuncu performans feature'ları için lig ortalaması ile doldurma (soğuk başlangıç)
    perf_cols = [c for c in df.columns if c.endswith(("_adr_form", "_kast_form", "_kddiff_form"))]
    for c in perf_cols:
        df[c] = df[c].fillna(df[c].median())

    # Fark (diff) kolonları -> modelin doğrudan "kim daha iyi" sinyalini görmesi için
    diff_pairs = [
        ("form_last5", "form_last5"), ("form_last10", "form_last10"), ("form_career", "form_career"),
        ("map_win_rate", "map_win_rate"), ("team_adr_form", "team_adr_form"),
        ("team_kast_form", "team_kast_form"), ("team_kddiff_form", "team_kddiff_form"),
    ]
    for base, _ in diff_pairs:
        df[f"diff_{base}"] = df[f"team1_{base}"] - df[f"team2_{base}"]

    df["diff_h2h_win_rate"] = df["team1_h2h_win_rate"] - 0.5  # zaten team1 perspektifinden simetrik

    return df


# --------------------------------------------------------------------------- #
# Ana pipeline
# --------------------------------------------------------------------------- #
def run(cfg: FeatureConfig) -> pd.DataFrame:
    logger.info("Temiz veri okunuyor: %s", cfg.processed_dir)
    maps = pd.read_parquet(cfg.processed_dir / "maps_clean.parquet")
    matches = pd.read_parquet(cfg.processed_dir / "matches_summary.parquet")

    logger.info("Team-match long format inşa ediliyor...")
    team_long = build_team_match_long(matches)
    team_long = add_team_form(team_long, cfg.form_windows)
    team_long = add_h2h(team_long)

    logger.info("Team-map long format inşa ediliyor (Map Advantage)...")
    team_map_long = build_team_map_match_agg(maps)
    team_map_long = add_map_advantage(team_map_long)

    logger.info("Oyuncu performans geçmişi inşa ediliyor (Player Firepower)...")
    player_match = build_player_match_agg(maps)
    player_match = add_player_rolling_form(player_match, cfg.player_form_window)
    team_fp = aggregate_team_firepower(player_match)

    logger.info("Maç-seviyesi feature tablosu birleştiriliyor...")
    match_features = assemble_match_level_features(matches, team_long, team_map_long, team_fp, cfg)

    logger.info("Feature'lar harita-seviyesi tabloya yayılıyor...")
    final_df = attach_map_features(maps, match_features, team_map_long, cfg)

    logger.info("Final feature tablosu: %d satır, %d kolon", *final_df.shape)

    engineered_cols = [c for c in final_df.columns if c.startswith(("team1_form", "team2_form", "diff_", "team1_h2h", "team2_h2h", "team1_map_win_rate", "team2_map_win_rate", "team1_team_", "team2_team_"))]
    logger.info("Üretilen feature kolonları (%d adet):\n%s", len(engineered_cols), "\n".join(engineered_cols))

    nan_check = final_df[engineered_cols].isna().sum()
    remaining_nans = nan_check[nan_check > 0]
    if len(remaining_nans):
        logger.warning("Hâlâ NaN içeren feature kolonları:\n%s", remaining_nans.to_string())
    else:
        logger.info("Tüm feature kolonlarında NaN kalmadı.")

    return final_df


def save(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "features_engineered.parquet"
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info("Kaydedildi -> %s (%.1f MB)", path, path.stat().st_size / 1e6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CS2 feature engineering pipeline.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = FeatureConfig(processed_dir=args.processed_dir, output_dir=args.output_dir)
    df = run(cfg)
    save(df, cfg.output_dir)


if __name__ == "__main__":
    main()