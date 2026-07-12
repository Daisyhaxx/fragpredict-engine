"""
src/predict.py
----------------
Eğitilmiş modeli (models/champion_model.pkl) ve metadata'yı yükleyip, HENÜZ
OYNANMAMIŞ bir maç için ("Team A vs Team B, X haritasında") tahmin üretir.

Zorluk: train.py'deki feature'lar geçmiş, tamamlanmış maçlardan hesaplanmıştı.
Burada ise elimizde henüz oynanmamış bir maç var — bu yüzden pipeline.py'deki
AYNI rolling/expanding mantığını, "bugüne kadarki TÜM geçmiş" üzerinden (shift
YAPMADAN, çünkü artık gerçekten geçmişte kalan veriden bahsediyoruz) yeniden
kullanıyoruz. Bu iki modülün feature tanımlarının birbirinden sapmaması hayati
önem taşıdığı için pipeline.py'deki üretim fonksiyonlarını DOĞRUDAN import
ediyoruz (kopyala-yapıştır yapmıyoruz).

Kullanım (CLI):
    python -m src.predict --team1 "Natus Vincere" --team2 "Vitality" --map Mirage --tier tier1 --bestof 3

Kullanım (Python içinden / API'den):
    from src.predict import Predictor
    predictor = Predictor()
    result = predictor.predict("Natus Vincere", "Vitality", map_name="Mirage", tier="tier1", best_of=3)
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from src.pipeline import (
    add_h2h,
    add_map_advantage,
    add_player_rolling_form,
    add_team_form,
    build_player_match_agg,
    build_team_map_match_agg,
    build_team_match_long,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("predict")


@dataclass
class PredictConfig:
    processed_dir: Path = Path("data/processed")
    models_dir: Path = Path("models")
    global_prior_win_rate: float = 0.5
    player_form_window: int = 5
    form_windows: tuple = (5, 10)


class Predictor:
    """Modeli ve geçmiş veriyi bir kez yükler, sonrasında birden çok tahmin için
    tekrar tekrar kullanılabilir (API içinde tek bir instance ayakta tutulmalı —
    her istekte diskten yeniden yüklemek maliyetlidir)."""

    def __init__(self, cfg: PredictConfig | None = None):
        self.cfg = cfg or PredictConfig()
        self._load_model_and_metadata()
        self._load_and_build_snapshots()

    # ----------------------------------------------------------------- #
    def _load_model_and_metadata(self) -> None:
        model_path = self.cfg.models_dir / "champion_model.pkl"
        meta_path = self.cfg.models_dir / "model_metadata.json"

        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Model veya metadata bulunamadı ({model_path}, {meta_path}). "
                "Önce 'python -m src.train' çalıştırıldığından emin ol."
            )

        logger.info("Model yükleniyor: %s", model_path)
        self.model = joblib.load(model_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.feature_cols = self.metadata["feature_cols"]
        self.categorical_features = self.metadata["categorical_features"]
        logger.info("Model: %s | Kalibrasyon: %s", self.metadata["champion_model_name"],
                    self.metadata.get("calibration_strategy", "raw"))

    # ----------------------------------------------------------------- #
    def _load_and_build_snapshots(self) -> None:
        """pipeline.py'deki üretim fonksiyonlarını yeniden kullanarak takım/oyuncu
        bazlı 'bugüne kadarki durum' tablolarını bir kez inşa eder ve bellekte tutar."""
        logger.info("Geçmiş veri okunuyor ve snapshot tabloları inşa ediliyor...")
        maps = pd.read_parquet(self.cfg.processed_dir / "maps_clean.parquet")
        matches = pd.read_parquet(self.cfg.processed_dir / "matches_summary.parquet")

        self.team_long = add_h2h(add_team_form(build_team_match_long(matches), self.cfg.form_windows))
        self.team_map_long = add_map_advantage(build_team_map_match_agg(maps))

        player_match = build_player_match_agg(maps)
        self.player_match = add_player_rolling_form(player_match, self.cfg.player_form_window)

        # Her takımın en güncel kadrosunu tahmin etmek için: o takımın SON oynadığı
        # maçtaki oyuncu id'lerini referans alıyoruz.
        self.latest_roster = self._build_latest_roster(maps)

        self.known_teams = set(self.team_long["team_name"].unique())
        self.known_maps = set(self.team_map_long["map_name"].unique())
        logger.info("%d benzersiz takım, %d benzersiz harita geçmişte bulundu.",
                    len(self.known_teams), len(self.known_maps))

    @staticmethod
    def _build_latest_roster(maps: pd.DataFrame) -> dict:
        roster = {}
        for side in ("team1", "team2"):
            player_cols = [f"{side}_player{p}_id" for p in range(1, 6)]
            sub = maps[["datetime", side] + player_cols].rename(columns={side: "team_name"})
            sub = sub.sort_values("datetime")
            for team_name, group in sub.groupby("team_name"):
                last_row = group.iloc[-1]
                roster[team_name] = [int(pid) for pid in last_row[player_cols].dropna().tolist()]
        return roster

    # ----------------------------------------------------------------- #
    # Snapshot fonksiyonları -- "bugün itibarıyla bu takımın durumu ne?"
    # ----------------------------------------------------------------- #
    def _team_form_snapshot(self, team_name: str) -> dict:
        rows = self.team_long[self.team_long["team_name"] == team_name].sort_values("datetime")
        if rows.empty:
            logger.warning("'%s' geçmiş veride bulunamadı -> varsayılan (lig ortalaması) değerler kullanılacak.", team_name)
            return {f"form_last{w}": self.cfg.global_prior_win_rate for w in self.cfg.form_windows} | {
                "form_career": self.cfg.global_prior_win_rate
            }
        win = rows["match_win"]
        out = {f"form_last{w}": win.tail(w).mean() for w in self.cfg.form_windows}
        out["form_career"] = win.mean()
        return out

    def _h2h_snapshot(self, team_name: str, opponent_name: str) -> dict:
        rows = self.team_long[
            (self.team_long["team_name"] == team_name) & (self.team_long["opponent_name"] == opponent_name)
        ]
        if rows.empty:
            return {"h2h_win_rate": self.cfg.global_prior_win_rate, "h2h_matches_played": 0}
        return {"h2h_win_rate": rows["match_win"].mean(), "h2h_matches_played": float(len(rows))}

    def _map_snapshot(self, team_name: str, map_name: str) -> dict:
        rows = self.team_map_long[
            (self.team_map_long["team_name"] == team_name) & (self.team_map_long["map_name"] == map_name)
        ]
        if rows.empty:
            return {"map_win_rate": self.cfg.global_prior_win_rate, "map_experience": 0.0}
        return {"map_win_rate": rows["map_win"].mean(), "map_experience": float(len(rows))}

    def _firepower_snapshot(self, team_name: str) -> dict:
        player_ids = self.latest_roster.get(team_name, [])
        if not player_ids:
            logger.warning("'%s' için kadro bilgisi bulunamadı -> lig medyanı kullanılacak.", team_name)
            med = self.player_match[["adr_roll", "kast_roll", "kddiff_roll"]].median()
            return {
                "team_adr_form": med["adr_roll"], "team_kast_form": med["kast_roll"],
                "team_kddiff_form": med["kddiff_roll"], "roster_avg_experience": 0.0,
            }

        latest_stats = (
            self.player_match[self.player_match["player_id"].isin(player_ids)]
            .sort_values("datetime")
            .groupby("player_id")
            .tail(1)
        )
        if latest_stats.empty:
            med = self.player_match[["adr_roll", "kast_roll", "kddiff_roll"]].median()
            return {
                "team_adr_form": med["adr_roll"], "team_kast_form": med["kast_roll"],
                "team_kddiff_form": med["kddiff_roll"], "roster_avg_experience": 0.0,
            }

        return {
            "team_adr_form": latest_stats["adr_roll"].mean(),
            "team_kast_form": latest_stats["kast_roll"].mean(),
            "team_kddiff_form": latest_stats["kddiff_roll"].mean(),
            "roster_avg_experience": float(len(latest_stats)),
        }

    # ----------------------------------------------------------------- #
    def _build_feature_row(self, team1: str, team2: str, map_name: str, tier: str, best_of: int) -> pd.DataFrame:
        t1_form = self._team_form_snapshot(team1)
        t2_form = self._team_form_snapshot(team2)
        h2h = self._h2h_snapshot(team1, team2)
        t1_map = self._map_snapshot(team1, map_name)
        t2_map = self._map_snapshot(team2, map_name)
        t1_fp = self._firepower_snapshot(team1)
        t2_fp = self._firepower_snapshot(team2)

        row = {
            "team1_form_last5": t1_form["form_last5"], "team1_form_last10": t1_form["form_last10"],
            "team1_form_career": t1_form["form_career"],
            "team2_form_last5": t2_form["form_last5"], "team2_form_last10": t2_form["form_last10"],
            "team2_form_career": t2_form["form_career"],
            "team1_h2h_win_rate": h2h["h2h_win_rate"], "team1_h2h_matches_played": h2h["h2h_matches_played"],
            "team1_map_win_rate": t1_map["map_win_rate"], "team1_map_experience": t1_map["map_experience"],
            "team2_map_win_rate": t2_map["map_win_rate"], "team2_map_experience": t2_map["map_experience"],
            "team1_team_adr_form": t1_fp["team_adr_form"], "team1_team_kast_form": t1_fp["team_kast_form"],
            "team1_team_kddiff_form": t1_fp["team_kddiff_form"],
            "team2_team_adr_form": t2_fp["team_adr_form"], "team2_team_kast_form": t2_fp["team_kast_form"],
            "team2_team_kddiff_form": t2_fp["team_kddiff_form"],
            "team1_roster_avg_experience": t1_fp["roster_avg_experience"],
            "team2_roster_avg_experience": t2_fp["roster_avg_experience"],
            "diff_form_last5": t1_form["form_last5"] - t2_form["form_last5"],
            "diff_form_last10": t1_form["form_last10"] - t2_form["form_last10"],
            "diff_form_career": t1_form["form_career"] - t2_form["form_career"],
            "diff_map_win_rate": t1_map["map_win_rate"] - t2_map["map_win_rate"],
            "diff_h2h_win_rate": h2h["h2h_win_rate"] - 0.5,
            "diff_team_adr_form": t1_fp["team_adr_form"] - t2_fp["team_adr_form"],
            "diff_team_kast_form": t1_fp["team_kast_form"] - t2_fp["team_kast_form"],
            "diff_team_kddiff_form": t1_fp["team_kddiff_form"] - t2_fp["team_kddiff_form"],
            "map_name": map_name, "tier": tier, "bestOf": str(best_of),
        }

        df = pd.DataFrame([row])[self.feature_cols]
        for c in self.categorical_features:
            df[c] = df[c].astype("category")
        return df

    # ----------------------------------------------------------------- #
    def predict(self, team1: str, team2: str, map_name: str, tier: str = "tier1", best_of: int = 3) -> dict:
        for name, label in ((team1, "team1"), (team2, "team2")):
            if name not in self.known_teams:
                logger.warning("'%s' (%s) geçmiş veride yok -> soğuk başlangıç (lig ortalaması) varsayımıyla tahmin edilecek.", name, label)
        if map_name not in self.known_maps:
            logger.warning("'%s' haritası geçmiş veride yok -> map advantage için lig ortalaması kullanılacak.", map_name)

        X = self._build_feature_row(team1, team2, map_name, tier, best_of)
        proba_team1 = float(self.model.predict_proba(X)[:, 1][0])

        return {
            "team1": team1,
            "team2": team2,
            "map": map_name,
            "tier": tier,
            "best_of": best_of,
            "team1_win_probability": round(proba_team1, 4),
            "team2_win_probability": round(1 - proba_team1, 4),
            "predicted_winner": team1 if proba_team1 >= 0.5 else team2,
            "model": self.metadata["champion_model_name"],
        }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CS2 maç/harita tahmini üretir.")
    parser.add_argument("--team1", required=True)
    parser.add_argument("--team2", required=True)
    parser.add_argument("--map", dest="map_name", required=True)
    parser.add_argument("--tier", default="tier1")
    parser.add_argument("--bestof", type=int, default=3)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PredictConfig(processed_dir=args.processed_dir, models_dir=args.models_dir)
    predictor = Predictor(cfg)
    result = predictor.predict(args.team1, args.team2, args.map_name, args.tier, args.bestof)

    print("\n" + "=" * 50)
    print(f"  {result['team1']}  vs  {result['team2']}   ({result['map']})")
    print("=" * 50)
    print(f"  {result['team1']}: %{result['team1_win_probability'] * 100:.1f}")
    print(f"  {result['team2']}: %{result['team2_win_probability'] * 100:.1f}")
    print(f"  Tahmin edilen kazanan: {result['predicted_winner']}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()