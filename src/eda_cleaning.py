"""
src/eda_cleaning.py
--------------------
CS2 Pro Matches veri seti (cs2_all_tiers_games.csv + tier dosyaları + teams/players/
tournaments referans tabloları) için EDA ve veri temizleme pipeline'ının ilk aşaması.

VERİ SETİ HAKKINDA KRİTİK NOTLAR (EDA sırasında keşfedildi):
    1. cs2_all_tiers_games.csv = tier1 + tier2 + tier3 dosyalarının birleşimidir.
       Bu üç dosya match_id bazında birbirini dışlar (kesişim yok). Bu yüzden
       tier bilgisini ayrı dosyalardan çıkarıp ana tabloya bir kolon olarak
       geri ekliyoruz -> modelin "rakip seviyesi" sinyalini öğrenebilmesi için.

    2. `is_total` kolonu satır granülaritesini belirler:
         - is_total == False -> GERÇEKTEN OYNANMIŞ harita satırı (skor + oyuncu stat'ları dolu)
         - is_total == True  -> maç özeti / oynanmamış harita placeholder'ı (skor 0-0,
           oyuncu stat'ı yok). Eğitim için KULLANILAMAZ, filtrelenir.

    3. `team1_win` kolonu maç bazlı DEĞİL, harita bazlıdır (Bo3/Bo5 maçlarda satırdan
       satıra değişir). Ancak ~%1-2 oranında score1_game/score2_game ile tutarsızdır
       (scraping kaynaklı anomali). Bu yüzden nihai hedef değişkeni ham kolondan değil,
       skorlardan yeniden türetiyoruz: team1_map_win = score1_game > score2_game.

    4. *** ÖNEMLİ DÜZELTME (ilk EDA'da yanlış varsayılmıştı) ***
       team_id bu veri setinde İSTİKRARLI bir takım kimliği DEĞİLDİR — aynı takım
       (ör. Natus Vincere) yüzlerce maçında onlarca farklı team_id ile görünüyor
       (muhtemelen HLTV'nin etkinlik/kadro bazlı iç kaydı). Buna karşın player_id
       oyuncu bazında tamamen stabildir. Bu yüzden takım bazlı rolling feature'larda
       (form, map advantage, H2H) grup anahtarı team_id DEĞİL, takım İSMİ olmalı;
       oyuncu bazlı feature'larda (firepower) ise player_id güvenle kullanılabilir.

Çıktılar:
    - data/processed/maps_clean.parquet     -> harita bazlı (bir satır = bir oynanan harita)
                                                 Bu, feature engineering ve model eğitimi
                                                 için ANA tablo olacak.
    - data/processed/matches_summary.parquet -> maç bazlı özet (bir satır = bir maç)
                                                 H2H ve maç-seviyesi rolling feature'lar için.

Kullanım:
    python -m src.eda_cleaning --raw-dir data/raw --output-dir data/processed
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eda_cleaning")


# --------------------------------------------------------------------------- #
# Konfigürasyon
# --------------------------------------------------------------------------- #
@dataclass
class CleaningConfig:
    raw_dir: Path
    output_dir: Path

    all_tiers_file: str = "cs2_all_tiers_games.csv"
    tier_files: dict = None  # __post_init__'te doldurulacak
    teams_file: str = "teams.csv"
    players_file: str = "players.csv"
    tournaments_file: str = "tournaments.csv"

    min_valid_date: str = "2018-01-01"

    def __post_init__(self):
        if self.tier_files is None:
            self.tier_files = {
                "tier1": "cs2_tier1_games.csv",
                "tier2": "cs2_tier2_games.csv",
                "tier3": "cs2_tier3_games.csv",
            }


# --------------------------------------------------------------------------- #
# Yükleme
# --------------------------------------------------------------------------- #
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.error("Dosya bulunamadı: %s", path)
        sys.exit(1)
    return pd.read_csv(path, low_memory=False)


def load_all_sources(cfg: CleaningConfig) -> dict[str, pd.DataFrame]:
    logger.info("Ham dosyalar okunuyor: %s", cfg.raw_dir)
    data = {
        "games": load_csv(cfg.raw_dir / cfg.all_tiers_file),
        "teams": load_csv(cfg.raw_dir / cfg.teams_file),
        "players": load_csv(cfg.raw_dir / cfg.players_file),
        "tournaments": load_csv(cfg.raw_dir / cfg.tournaments_file),
    }
    for tier_name, fname in cfg.tier_files.items():
        data[tier_name] = load_csv(cfg.raw_dir / fname)

    logger.info("games: %d satır, %d kolon", *data["games"].shape)
    logger.info("teams: %d satır | players: %d satır | tournaments: %d satır",
                len(data["teams"]), len(data["players"]), len(data["tournaments"]))
    return data


# --------------------------------------------------------------------------- #
# Tier etiketleme
# --------------------------------------------------------------------------- #
def attach_tier_label(games: pd.DataFrame, cfg: CleaningConfig, data: dict) -> pd.DataFrame:
    """match_id -> tier eşlemesini tier1/tier2/tier3 dosyalarından çıkarır."""
    tier_map: dict[int, str] = {}
    for tier_name in cfg.tier_files:
        ids = data[tier_name]["match_id"].unique()
        overlap = set(ids) & set(tier_map.keys())
        if overlap:
            logger.warning("%d match_id birden fazla tier dosyasında bulundu.", len(overlap))
        tier_map.update({mid: tier_name for mid in ids})

    games["tier"] = games["match_id"].map(tier_map)
    unmatched = games["tier"].isna().sum()
    if unmatched:
        logger.warning("%d satır hiçbir tier dosyasıyla eşleşmedi -> 'unknown' etiketlenecek.", unmatched)
        games["tier"] = games["tier"].fillna("unknown")

    logger.info("Tier dağılımı:\n%s", games["tier"].value_counts().to_string())
    return games


# --------------------------------------------------------------------------- #
# Harita (map) seviyesi temizlik — ANA eğitim tablosu
# --------------------------------------------------------------------------- #
def build_map_level_table(games: pd.DataFrame, cfg: CleaningConfig) -> pd.DataFrame:
    logger.info("--- Harita (map) seviyesi tablo inşa ediliyor ---")

    df = games[games["is_total"] == False].copy()  # noqa: E712 (pandas bool mask netliği için)
    logger.info("is_total==False filtresi sonrası: %d satır", len(df))

    before = len(df)
    df = df.dropna(subset=["map_name", "score1_game", "score2_game"])
    logger.info("Eksik map_name/skor nedeniyle %d satır atıldı.", before - len(df))

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    before = len(df)
    df = df[df["datetime"].notna() & (df["datetime"] >= cfg.min_valid_date)]
    logger.info("Geçersiz/çok eski tarih nedeniyle %d satır atıldı.", before - len(df))

    df["score1_game"] = df["score1_game"].astype(int)
    df["score2_game"] = df["score2_game"].astype(int)

    # Hedef değişken: HAM team1_win yerine skordan yeniden türetilmiş, güvenilir versiyon
    df["team1_map_win"] = (df["score1_game"] > df["score2_game"]).astype(int)

    inconsistent = (df["team1_map_win"] != df["team1_win"]).sum()
    logger.info(
        "Ham 'team1_win' ile skordan türetilen hedef arasında %d satırda (%0.2f%%) tutarsızlık "
        "bulundu -> güvenilir olan skor-türevi hedef kullanılacak.",
        inconsistent, 100 * inconsistent / len(df),
    )

    df["map_name"] = df["map_name"].astype(str).str.strip().str.title()

    before = len(df)
    df = df.drop_duplicates(subset=["match_id", "game_id"], keep="first")
    logger.info("Duplicate (match_id, game_id) nedeniyle %d satır atıldı.", before - len(df))

    df = df.sort_values("datetime").reset_index(drop=True)

    # Oyuncu stat kolonlarındaki eksiklik oranını raporla (sub/eksik veri senaryoları)
    player_stat_cols = [c for c in df.columns if any(
        c.endswith(suffix) for suffix in ("_kills", "_deaths", "_assists", "_adr", "_kast", "_kddiff")
    )]
    missing_pct = df[player_stat_cols].isna().mean().sort_values(ascending=False) * 100
    logger.info("Oyuncu istatistik kolonlarında en yüksek eksiklik (ilk 6):\n%s",
                missing_pct.head(6).to_string())
    df["team1_missing_p5"] = df["team1_player5"].isna().astype(int)
    df["team2_missing_p5"] = df["team2_player5"].isna().astype(int)

    logger.info("Harita seviyesi final tablo: %d satır, tarih aralığı %s -> %s",
                len(df), df["datetime"].min().date(), df["datetime"].max().date())
    logger.info("Sınıf dengesi (team1_map_win):\n%s",
                df["team1_map_win"].value_counts(normalize=True).round(3).to_string())

    return df


# --------------------------------------------------------------------------- #
# Maç seviyesi özet tablo — H2H ve seri-bazlı feature'lar için
# --------------------------------------------------------------------------- #
def build_match_summary_table(games: pd.DataFrame) -> pd.DataFrame:
    logger.info("--- Maç seviyesi özet tablo inşa ediliyor ---")

    const_cols = [
        "match_id", "tournament", "team1_id", "team1", "team2_id", "team2",
        "score1_match", "score2_match", "bestOf", "datetime", "tier",
    ]
    summary = (
        games.sort_values("is_total")  # False önce gelsin, ilk kayıt daha güvenilir olsun
        .groupby("match_id", as_index=False)[const_cols[1:]]
        .first()
    )
    summary["datetime"] = pd.to_datetime(summary["datetime"], errors="coerce")
    summary = summary.dropna(subset=["datetime", "score1_match", "score2_match"])
    summary["team1_match_win"] = (summary["score1_match"] > summary["score2_match"]).astype(int)
    summary = summary.sort_values("datetime").reset_index(drop=True)

    logger.info("Maç seviyesi tablo: %d benzersiz maç", len(summary))
    return summary


# --------------------------------------------------------------------------- #
# Kaydetme
# --------------------------------------------------------------------------- #
def save_outputs(map_df: pd.DataFrame, match_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    map_path = output_dir / "maps_clean.parquet"
    match_path = output_dir / "matches_summary.parquet"

    map_df.to_parquet(map_path, index=False, engine="pyarrow")
    match_df.to_parquet(match_path, index=False, engine="pyarrow")

    logger.info("Kaydedildi -> %s (%.1f MB)", map_path, map_path.stat().st_size / 1e6)
    logger.info("Kaydedildi -> %s (%.1f MB)", match_path, match_path.stat().st_size / 1e6)


# --------------------------------------------------------------------------- #
# Ana pipeline
# --------------------------------------------------------------------------- #
def run(cfg: CleaningConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_all_sources(cfg)
    games = attach_tier_label(data["games"], cfg, data)

    map_df = build_map_level_table(games, cfg)
    match_df = build_match_summary_table(games)

    save_outputs(map_df, match_df, cfg.output_dir)
    return map_df, match_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CS2 maç verisi EDA ve temizleme.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = CleaningConfig(raw_dir=args.raw_dir, output_dir=args.output_dir)
    run(cfg)


if __name__ == "__main__":
    main()