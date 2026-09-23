"""
src/ingest_new.py
------------------
Kaggle veri setinin bittiği tarihten sonraki maçları (data/raw_new/*.json veya *.csv) Kaggle
şemasına dönüştürüp data/raw/cs2_new_games.csv olarak yazar. eda_cleaning.py bu
dosyayı otomatik olarak Kaggle verisiyle birleştirir; her maçın tier'ı turnuvasına göre belirlenir.

Akış:
    data/raw_new/*.json veya *.csv      (ham veri -- bu script ona ASLA dokunmaz)
      -> şema kontrolü                  (eksik kolon varsa durur)
      -> tekrar eden haritaları atma     (aynı maç iki kez toplandıysa son hali kalır)
      -> tarih filtresi                  (sadece Kaggle'ın son tarihinden SONRASI)
      -> tutarlılık kontrolleri          (seri skoru, kddiff, ADR/KAST aralıkları...)
      -> takım isimlerini eşleme         (otomatik + data/mappings/team_names.csv)
      -> turnuva tier'ını belirleme      (Kaggle serileri + data/mappings/tournament_tiers.csv)
      -> oyuncu ID'lerini eşleme         (otomatik + data/mappings/player_ids.csv)
      -> match_id / game_id dönüşümü     (Kaggle ID'leriyle çakışmayan sayılar)
    -> data/raw/cs2_new_games.csv

Takımlarda KATI davranır: eşleşmeyen tek bir takım bile varsa çıktı yazılmaz.
Sebep: yanlış eşleşen bir takım (ör. "Vitality" != "Team Vitality") modelin gözünde
yepyeni bir takım olur ve Elo/form zinciri SESSİZCE sıfırlanır. Eşleşmeyen isimler
team_names.csv'ye boş olarak eklenir; sen doldurursun (gerçekten yeni takımsa NEW yaz).

Oyuncularda ESNEK davranır: eşleşmeyen oyuncu yeni oyuncu kabul edilir ama raporda
listelenir. Yanlış eşleşme görürsen player_ids.csv ile düzeltirsin.

Kullanım (proje kök dizininden):
    python -m src.ingest_new
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest_new")

# Yeni match_id = kaynak match_id + bu ofset. Kaggle ID'leri bu sayının çok altında
# olduğu için çakışma imkansız (script yine de ayrıca kontrol ediyor).
MATCH_ID_OFFSET = 1_000_000_000
# Kaggle'da karşılığı olmayan oyuncuya verilecek ID = bu ofset + kaynak player_id.
NEW_PLAYER_ID_OFFSET = 10_000_000
NEW_MARKER = "NEW"

SIDES = ("team1", "team2")
PLAYER_SLOTS = range(1, 6)
PLAYER_STATS = ("kills", "deaths", "assists", "adr", "kast", "kddiff")

MATCH_COLUMNS = [
    "match_id", "game_id", "datetime", "tournament", "team1", "team2",
    "team1_id", "team2_id", "bestOf", "score1_match", "score2_match",
    "map_name", "score1_game", "score2_game", "is_total",
]

# Takım isimlerinden atılan "gürültü" kelimeler: "Team Vitality" -> "vitality",
# "FURIA Esports" -> "furia", "FaZe Clan" -> "faze".
TEAM_NOISE_TOKENS = {"team", "gaming", "esports", "esport", "clan", "club"}


# --------------------------------------------------------------------------- #
# Yardımcılar
# --------------------------------------------------------------------------- #
def player_cols(side: str, slot: int) -> tuple[str, str, list[str]]:
    base = f"{side}_player{slot}"
    return base, f"{base}_id", [f"{base}_{s}" for s in PLAYER_STATS]


def required_columns() -> list[str]:
    cols = list(MATCH_COLUMNS)
    for side in SIDES:
        for slot in PLAYER_SLOTS:
            name, pid, stats = player_cols(side, slot)
            cols += [name, pid, *stats]
    return cols


def parse_datetime(series: pd.Series) -> pd.Series:
    """Farklı formatları (T'li, T'siz, saat dilimli) tek tip, saat dilimsiz UTC'ye çevirir."""
    dt = pd.to_datetime(series, format="mixed", errors="coerce", utc=True)
    return dt.dt.tz_convert(None)


def kaggle_datetime_format(kaggle: pd.DataFrame) -> str:
    """Çıktıdaki tarih formatını Kaggle'daki ham formatla BİREBİR aynı yapar.
    Aksi halde eda_cleaning iki farklı formatı birlikte okurken tarihleri
    sessizce NaT'ye çevirip satır kaybettirebilir."""
    sample = kaggle["datetime"].dropna().astype(str).iloc[0]
    sep = "T" if "T" in sample else " "
    has_tz = bool(re.search(r"(Z|[+-]\d{2}:?\d{2})$", sample))
    return f"%Y-%m-%d{sep}%H:%M:%S" + ("+00:00" if has_tz else "")


def normalize_team(name: str) -> str:
    s = re.sub(r"[\W_]+", " ", str(name).lower())
    tokens = [t for t in s.split() if t not in TEAM_NOISE_TOKENS]
    return "".join(tokens) if tokens else s.replace(" ", "")


def kaggle_nickname(full_name: str) -> str:
    """Kaggle'da iki format var:
        'NiKo (Nikola Kovač), BA' -> 'niko'
        'Neityu, FR'              -> 'neityu'   (gerçek isim yok, sadece ülke kodu)
    İkinci format ilk sürümde gözden kaçmıştı; bu oyuncular yanlışlıkla 'yeni oyuncu'
    sayılıyordu."""
    s = str(full_name).split(" (")[0]
    s = re.sub(r",\s*[A-Za-z]{2,3}\s*$", "", s)
    return s.strip().lower()


def load_mapping(path: Path, key_col: str, value_col: str) -> dict[str, str]:
    if not path.exists():
        return {}
    m = pd.read_csv(path, dtype=str)
    m = m.dropna(subset=[key_col, value_col])
    m = m[m[value_col].str.strip() != ""]
    return dict(zip(m[key_col].str.strip(), m[value_col].str.strip()))


# --------------------------------------------------------------------------- #
# 1) Yükleme, şema, tekrar, tarih
# --------------------------------------------------------------------------- #
def _read_json_file(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    """Scraper çıktısı: {"status": "ok", "url": ..., "scraped_at": ..., "rows": [...]}.
    Doğrudan satır listesi ([...]) de kabul edilir. Sorun varsa (None, neden) döner."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return None, f"JSON okunamadı: {exc}"

    if isinstance(payload, dict):
        if payload.get("status", "ok") != "ok":
            return None, f"status={payload.get('status')}"
        rows, scraped_at = payload.get("rows") or [], payload.get("scraped_at")
    elif isinstance(payload, list):
        rows, scraped_at = payload, None
    else:
        return None, "beklenmeyen JSON yapısı"

    if not rows:
        return None, "rows boş"
    part = pd.DataFrame(rows)
    part["scraped_at"] = scraped_at
    return part, None


def load_raw_new(raw_new_dir: Path, reports_dir: Path) -> pd.DataFrame:
    files = sorted(list(raw_new_dir.glob("*.json")) + list(raw_new_dir.glob("*.csv")))
    if not files:
        logger.error("%s altında hiç JSON/CSV yok.", raw_new_dir)
        sys.exit(1)

    frames, skipped = [], []
    for f in files:
        if f.suffix == ".json":
            part, reason = _read_json_file(f)
            if part is None:
                skipped.append({"file": f.name, "reason": reason})
                continue
        else:
            part = pd.read_csv(f, dtype={"game_id": str})
        part["game_id"] = part["game_id"].astype(str)
        part["source_file"] = f.name
        frames.append(part)

    if skipped:
        pd.DataFrame(skipped).to_csv(reports_dir / "skipped_files.csv", index=False)
        logger.warning("%d dosya atlandı -> %s", len(skipped), reports_dir / "skipped_files.csv")
    if not frames:
        logger.error("Okunabilir dosya kalmadı.")
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True)
    logger.info("%d dosya okundu (%d atlandı), toplam %d harita satırı.", len(frames), len(skipped), len(df))
    return df


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in required_columns() if c not in df.columns]
    if missing:
        logger.error("Eksik kolonlar (%d adet): %s", len(missing), missing[:15])
        sys.exit(1)
    logger.info("Şema kontrolü geçti.")


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Aynı harita birden fazla kez çekildiyse EN YENİ çekimi tutar (scraped_at varsa
    ona göre, yoksa dosya sırasına göre)."""
    before = len(df)
    if "scraped_at" in df.columns:
        order = pd.to_datetime(df["scraped_at"], format="mixed", errors="coerce", utc=True)
        df = df.assign(_order=order).sort_values("_order", na_position="first").drop(columns="_order")
    df = df.drop_duplicates(subset=["match_id", "game_id"], keep="last")
    if before != len(df):
        logger.info("Tekrar eden %d harita satırı atıldı (en yeni çekim tutuldu).", before - len(df))
    return df


def filter_after_cutoff(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    df = df.copy()
    df["_dt"] = parse_datetime(df["datetime"])
    bad = df["_dt"].isna().sum()
    if bad:
        logger.warning("%d satırda tarih okunamadı, atıldı.", bad)
    before = len(df)
    df = df[df["_dt"] > cutoff]
    logger.info("Kaggle'ın son tarihi: %s -> bundan önceki %d satır atıldı (Kaggle'da zaten var).",
                cutoff, before - bad - len(df))
    return df


# --------------------------------------------------------------------------- #
# 2) Tutarlılık kontrolleri
# --------------------------------------------------------------------------- #
def find_invalid_matches(df: pd.DataFrame) -> dict:
    """Sorunlu maçları nedenleriyle döndürür. Bir haritası sorunlu olan maçın
    TÜMÜ atılır -- yarım bir seri, seri skoru/H2H hesaplarını bozar."""
    problems: dict = {}

    def flag(mask: pd.Series, reason: str) -> None:
        for mid in df.loc[mask, "match_id"].unique():
            problems.setdefault(mid, set()).add(reason)

    flag(df["team1"] == df["team2"], "team1 ve team2 aynı")
    flag(df["score1_game"] == df["score2_game"], "harita skoru berabere")
    flag(df[["score1_game", "score2_game"]].max(axis=1) < 13, "harita kazananı 13'e ulaşmamış")

    for side in SIDES:
        for slot in PLAYER_SLOTS:
            _, _, stats = player_cols(side, slot)
            kills, deaths, _, adr, kast, kddiff = (df[c] for c in stats)
            known = kills.notna() & deaths.notna() & kddiff.notna()
            flag(known & (kills - deaths != kddiff), "kddiff != kills - deaths")
            flag(adr.notna() & ((adr < 0) | (adr > 250)), "ADR mantık dışı")
            flag(kast.notna() & ((kast < 0) | (kast > 100)), "KAST 0-100 dışında")

    # Seri skoru, toplanan harita sonuçlarıyla tutmalı (tutmuyorsa büyük ihtimalle
    # bir harita eksik toplanmış).
    map_win1 = (df["score1_game"] > df["score2_game"]).groupby(df["match_id"]).sum()
    map_win2 = (df["score2_game"] > df["score1_game"]).groupby(df["match_id"]).sum()
    series = df.groupby("match_id")[["score1_match", "score2_match"]].first()
    bad_series = series.index[(series["score1_match"] != map_win1) | (series["score2_match"] != map_win2)]
    for mid in bad_series:
        problems.setdefault(mid, set()).add("seri skoru harita sonuçlarıyla tutmuyor (eksik harita?)")

    # Maç içinde sabit olması gerekenler
    per_match = df.groupby("match_id")[["team1", "team2", "datetime", "tournament"]].nunique()
    for mid in per_match.index[(per_match > 1).any(axis=1)]:
        problems.setdefault(mid, set()).add("aynı maçın haritalarında takım/tarih/turnuva farklı")

    return problems


# --------------------------------------------------------------------------- #
# 3) Takım eşleştirme (KATI)
# --------------------------------------------------------------------------- #
def resolve_teams(df: pd.DataFrame, kaggle: pd.DataFrame, manual: dict) -> tuple[dict, list, list]:
    counts = pd.concat([kaggle["team1"], kaggle["team2"]]).dropna().value_counts()
    exact = set(counts.index)
    by_key: dict[str, list[str]] = {}
    for name in counts.index:  # en sık geçenden en seyreğe
        by_key.setdefault(normalize_team(name), []).append(name)

    resolved, report, unresolved = {}, [], []
    for name in sorted(set(df["team1"]) | set(df["team2"])):
        if name in manual:
            target = manual[name]
            resolved[name] = name if target == NEW_MARKER else target
            method = "yeni takım (manuel)" if target == NEW_MARKER else "manuel"
            if target != NEW_MARKER and target not in exact:
                logger.warning("Manuel eşleştirme '%s' -> '%s': bu isim Kaggle'da yok, yazımı kontrol et.",
                               name, target)
        elif name in exact:
            resolved[name], method = name, "birebir aynı"
        elif normalize_team(name) in by_key:
            candidates = by_key[normalize_team(name)]
            resolved[name], method = candidates[0], "normalize"
            if len(candidates) > 1:
                method = f"normalize (adaylar: {' | '.join(candidates)}, en sık geçen seçildi)"
        else:
            close = difflib.get_close_matches(normalize_team(name), list(by_key), n=3, cutoff=0.6)
            unresolved.append({"source_name": name, "suggestions": " | ".join(by_key[k][0] for k in close)})
            continue
        report.append({"source_name": name, "kaggle_name": resolved[name], "method": method})
    return resolved, report, unresolved


def append_templates(path: Path, keys: list[str], key_col: str, value_col: str) -> None:
    """Eşleşmeyen anahtarları mapping dosyasına değeri BOŞ olarak ekler; sen
    doldurursun. Zaten dosyada olanları tekrar eklemez."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(path, dtype=str) if path.exists() else pd.DataFrame(columns=[key_col, value_col])
    known = set(existing[key_col].dropna())
    new_rows = pd.DataFrame([{key_col: k, value_col: ""} for k in keys if k not in known])
    if not new_rows.empty:
        pd.concat([existing, new_rows], ignore_index=True).to_csv(path, index=False)


# --------------------------------------------------------------------------- #
# 3b) Turnuva -> tier eşleştirme (KATI)
# --------------------------------------------------------------------------- #
VALID_TIERS = ("tier1", "tier2", "tier3")
SEASON_WORDS = {"spring", "summer", "fall", "autumn", "winter"}


def normalize_tournament(name: str) -> str:
    """Turnuva serisini yıl/sezon/numaradan arındırır:
    'ESL Pro League Season 23' -> 'esl pro league', 'CCT Online Finals #5' -> 'cct online finals'.
    'Qualifier', 'Closed' gibi kelimeler BİLEREK korunur -- bir etkinliğin elemesi
    ile ana turnuvası farklı tier'da olabilir."""
    s = str(name).lower()
    s = re.sub(r"\bseason\s*\d+\b|\bs\d+\b|#\s*\d+", " ", s)
    s = re.sub(r"[\W_]+", " ", s)
    return " ".join(t for t in s.split() if not t.isdigit() and t not in SEASON_WORDS)


def build_tournament_tier_index(raw_dir: Path) -> dict[str, str]:
    """Kaggle tier dosyalarından: normalize edilmiş seri adı -> en sık görüldüğü tier."""
    frames = []
    for tier in VALID_TIERS:
        path = raw_dir / f"cs2_{tier}_games.csv"
        if not path.exists():
            logger.error("Tier dosyası bulunamadı: %s", path)
            sys.exit(1)
        t = pd.read_csv(path, usecols=["match_id", "tournament"]).drop_duplicates("match_id")
        t["tier"] = tier
        frames.append(t)
    all_t = pd.concat(frames, ignore_index=True).dropna(subset=["tournament"])
    all_t["series"] = all_t["tournament"].map(normalize_tournament)
    counts = all_t.groupby(["series", "tier"]).size().unstack(fill_value=0)
    return counts.idxmax(axis=1).to_dict()


def resolve_tiers(df: pd.DataFrame, index: dict[str, str], manual: dict) -> tuple[dict, list, list]:
    resolved, report, unresolved = {}, [], []
    for tournament in sorted(df["tournament"].dropna().unique()):
        series = normalize_tournament(tournament)
        if tournament in manual:
            tier = manual[tournament].lower()
            if tier not in VALID_TIERS:
                logger.error("tournament_tiers.csv: '%s' için geçersiz tier '%s' (tier1/tier2/tier3 olmalı).",
                             tournament, manual[tournament])
                sys.exit(1)
            resolved[tournament], method = tier, "manuel"
        elif series in index:
            resolved[tournament], method = index[series], f"Kaggle serisi: '{series}'"
        else:
            close = difflib.get_close_matches(series, list(index), n=3, cutoff=0.6)
            unresolved.append({
                "tournament": tournament,
                "benzer_kaggle_serileri": " | ".join(f"{c} ({index[c]})" for c in close),
            })
            continue
        report.append({"tournament": tournament, "tier": resolved[tournament], "method": method})
    return resolved, report, unresolved


# --------------------------------------------------------------------------- #
# 4) Oyuncu eşleştirme (ESNEK)
# --------------------------------------------------------------------------- #
def resolve_players(df: pd.DataFrame, kaggle: pd.DataFrame, manual: dict) -> tuple[dict, list]:
    rows = []
    for side in SIDES:
        for slot in PLAYER_SLOTS:
            name_col, id_col, _ = player_cols(side, slot)
            sub = kaggle[[name_col, id_col, side]].dropna(subset=[name_col, id_col])
            rows.append(pd.DataFrame({
                "nick": sub[name_col].map(kaggle_nickname),
                "kaggle_id": sub[id_col].astype("int64"),
                "team": sub[side],
            }))
    k = pd.concat(rows, ignore_index=True)
    if k["kaggle_id"].max() >= NEW_PLAYER_ID_OFFSET:
        logger.error("Kaggle player_id'leri NEW_PLAYER_ID_OFFSET'i aşıyor, ofseti büyüt.")
        sys.exit(1)
    nick_to_ids = k.groupby("nick")["kaggle_id"].apply(lambda s: sorted(set(s))).to_dict()
    id_to_teams = k.groupby("kaggle_id")["team"].apply(set).to_dict()

    # Kaynaktaki oyuncular (takım isimleri BURADA zaten Kaggle isimlerine çevrilmiş)
    src_rows = []
    for side in SIDES:
        for slot in PLAYER_SLOTS:
            name_col, id_col, _ = player_cols(side, slot)
            sub = df[[name_col, id_col, side]].dropna(subset=[id_col])
            src_rows.append(pd.DataFrame({
                "source_id": pd.to_numeric(sub[id_col]).astype("int64"),
                "name": sub[name_col], "team": sub[side],
            }))
    src = pd.concat(src_rows, ignore_index=True)

    id_map, report = {}, []
    for source_id, grp in src.groupby("source_id"):
        name = grp["name"].mode().iat[0]
        teams = set(grp["team"])
        new_id = NEW_PLAYER_ID_OFFSET + int(source_id)
        candidates = nick_to_ids.get(str(name).strip().lower(), [])

        if str(source_id) in manual:
            target = manual[str(source_id)]
            id_map[source_id] = new_id if target == NEW_MARKER else int(target)
            status = "manuel"
        elif len(candidates) == 1:
            id_map[source_id], status = candidates[0], "eşleşti"
        elif len(candidates) > 1:
            narrowed = [c for c in candidates if id_to_teams.get(c, set()) & teams]
            if len(narrowed) == 1:
                id_map[source_id], status = narrowed[0], "eşleşti (takım bilgisiyle)"
            else:
                id_map[source_id], status = new_id, "BELİRSİZ -> yeni oyuncu sayıldı"
        else:
            id_map[source_id], status = new_id, "yeni oyuncu"

        report.append({
            "source_player_id": int(source_id), "name": name, "teams": " | ".join(sorted(teams)),
            "kaggle_player_id": id_map[source_id], "status": status,
            "candidates": " | ".join(map(str, candidates)),
        })
    return id_map, report


# --------------------------------------------------------------------------- #
# 5) ID dönüşümü ve son hazırlık
# --------------------------------------------------------------------------- #
def remap_ids(df: pd.DataFrame, kaggle: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    src_match = pd.to_numeric(df["match_id"], errors="raise").astype("int64")
    new_match = src_match + MATCH_ID_OFFSET

    kaggle_ids = set(pd.to_numeric(kaggle["match_id"], errors="coerce").dropna().astype("int64"))
    collisions = set(new_match) & kaggle_ids
    if collisions:
        logger.error("Yeni match_id'ler Kaggle'la çakışıyor: %s", sorted(collisions)[:10])
        sys.exit(1)

    # Harita sırası: game_id "2398108_2" ise 2; yoksa maç içindeki sıra
    map_no = pd.to_numeric(df["game_id"].astype(str).str.extract(r"_(\d+)$")[0], errors="coerce")
    fallback = df.groupby(src_match).cumcount() + 1
    map_no = map_no.fillna(fallback).astype("int64")

    df["source_match_id"] = src_match
    df["source_game_id"] = df["game_id"]
    df["match_id"] = new_match
    df["game_id"] = new_match * 10 + map_no
    return df


def finalize(df: pd.DataFrame, kaggle: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = df["_dt"].dt.strftime(kaggle_datetime_format(kaggle))
    df["is_total"] = False
    df["team1_win"] = (df["score1_game"] > df["score2_game"]).astype(int)
    df["games_played"] = df.groupby("match_id")["game_id"].transform("count")
    if "map_id" not in df.columns:
        df["map_id"] = pd.NA
    df = df.drop(columns=["_dt"])
    front = [c for c in kaggle.columns if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest].sort_values(["datetime", "match_id", "game_id"])


# --------------------------------------------------------------------------- #
# Ana akış
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> None:
    reports_dir = args.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    team_map_path = args.mappings_dir / "team_names.csv"
    player_map_path = args.mappings_dir / "player_ids.csv"
    tier_map_path = args.mappings_dir / "tournament_tiers.csv"

    raw = load_raw_new(args.raw_new_dir, reports_dir)
    validate_schema(raw)
    raw = deduplicate(raw)

    logger.info("Kaggle verisi okunuyor: %s", args.kaggle_games)
    kaggle = pd.read_csv(args.kaggle_games, low_memory=False)
    df = filter_after_cutoff(raw, parse_datetime(kaggle["datetime"]).max())
    if df.empty:
        logger.warning("Kaggle'ın son tarihinden sonra maç yok, yazılacak bir şey kalmadı.")
        return

    problems = find_invalid_matches(df)
    if problems:
        pd.DataFrame([{"match_id": m, "problems": "; ".join(sorted(p))} for m, p in problems.items()]) \
            .to_csv(reports_dir / "invalid_matches.csv", index=False)
        logger.warning("%d maç tutarsız bulundu ve atıldı -> %s", len(problems), reports_dir / "invalid_matches.csv")
        df = df[~df["match_id"].isin(problems)]
    if df.empty:
        logger.error("Geçerli maç kalmadı.")
        sys.exit(1)

    # --- Takımlar ve turnuva tier'ları (ikisi de katı; eksikler TEK SEFERDE raporlanır) ---
    team_map, team_report, unresolved_teams = resolve_teams(
        df, kaggle, load_mapping(team_map_path, "source_name", "kaggle_name"))
    tier_map, tier_report, unresolved_tiers = resolve_tiers(
        df, build_tournament_tier_index(args.kaggle_games.parent),
        load_mapping(tier_map_path, "tournament", "tier"))
    pd.DataFrame(team_report).to_csv(reports_dir / "team_matches.csv", index=False)
    pd.DataFrame(tier_report).to_csv(reports_dir / "tournament_matches.csv", index=False)

    if unresolved_teams or unresolved_tiers:
        steps = []
        if unresolved_teams:
            pd.DataFrame(unresolved_teams).to_csv(reports_dir / "unmatched_teams.csv", index=False)
            append_templates(team_map_path, [u["source_name"] for u in unresolved_teams], "source_name", "kaggle_name")
            steps.append(f"  - {len(unresolved_teams)} takım: {team_map_path} -> kaggle_name kolonuna Kaggle'daki "
                         f"ismi yaz, gerçekten yeniyse NEW yaz")
        if unresolved_tiers:
            pd.DataFrame(unresolved_tiers).to_csv(reports_dir / "unmatched_tournaments.csv", index=False)
            append_templates(tier_map_path, [u["tournament"] for u in unresolved_tiers], "tournament", "tier")
            steps.append(f"  - {len(unresolved_tiers)} turnuva: {tier_map_path} -> tier kolonuna tier1/tier2/tier3 yaz "
                         f"(benzer seriler için {reports_dir / 'unmatched_tournaments.csv'})")
        logger.error("Eşleşmeyenler var, çıktı YAZILMADI. Doldurup tekrar çalıştır:\n%s", "\n".join(steps))
        sys.exit(1)

    for side in SIDES:
        df[side] = df[side].map(team_map)
    df["tier"] = df["tournament"].map(tier_map)

    # --- Oyuncular (esnek) ---
    player_map, player_report = resolve_players(df, kaggle, load_mapping(player_map_path, "source_player_id", "kaggle_player_id"))
    pd.DataFrame(player_report).to_csv(reports_dir / "player_matches.csv", index=False)
    for side in SIDES:
        for slot in PLAYER_SLOTS:
            _, id_col, _ = player_cols(side, slot)
            src_ids = pd.to_numeric(df[id_col])
            df[id_col] = src_ids.map(lambda x: player_map.get(int(x)) if pd.notna(x) else pd.NA).astype("Int64")

    # --- ID'ler ve yazma ---
    df = finalize(remap_ids(df, kaggle), kaggle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    status = pd.DataFrame(player_report)["status"].value_counts()
    logger.info("=" * 60)
    logger.info("Yazıldı -> %s", args.output)
    logger.info("Maç: %d | Harita: %d | Tarih: %s -> %s",
                df["match_id"].nunique(), len(df), df["datetime"].min(), df["datetime"].max())
    logger.info("Tier dağılımı (maç): %s", df.drop_duplicates("match_id")["tier"].value_counts().to_dict())
    logger.info("Takım eşleşmeleri: %s", pd.DataFrame(team_report)["method"].value_counts().to_dict())
    logger.info("Oyuncu eşleşmeleri: %s", status.to_dict())
    logger.info("Kontrol için raporlar: %s (özellikle 'yeni oyuncu' ve 'BELİRSİZ' satırlarına bak)", reports_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Yeni maç verisini Kaggle şemasına dönüştürür.")
    p.add_argument("--raw-new-dir", type=Path, default=Path("data/raw_new"))
    p.add_argument("--kaggle-games", type=Path, default=Path("data/raw/cs2_all_tiers_games.csv"))
    p.add_argument("--mappings-dir", type=Path, default=Path("data/mappings"))
    p.add_argument("--reports-dir", type=Path, default=Path("data/ingest_reports"))
    p.add_argument("--output", type=Path, default=Path("data/raw/cs2_new_games.csv"))
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())