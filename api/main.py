"""
api/main.py
------------
CS2 Maç Tahmin Motoru -- FastAPI servis katmanı.

Sorumluluklar:
    - Uygulama ayağa kalkarken Predictor'ı BİR KEZ yükler (her istekte değil --
      geçmiş veriyi işleyip snapshot tabloları kurmak ~20-30 saniye sürüyor).
    - POST /api/v1/predict: bir maç için galibiyet olasılığı üretir.
    - Her tahmini Supabase'deki 'predictions' tablosuna loglar (React Native
      uygulamasında "geçmiş tahminlerim" gibi bir ekran için temel oluşturur).
    - Supabase kapalı/yanlış yapılandırılmışsa bile API ÇÖKMEZ -- sadece loglama
      atlanır ve uyarı basılır. Tahmin üretmek, loglamaya bağımlı olmamalı.

Ortam değişkenleri (.env dosyasından okunur, bkz. .env.example):
    SUPABASE_URL, SUPABASE_KEY

Çalıştırma:
    uvicorn api.main:app --reload --port 8000
    Sonra: http://localhost:8000/docs (Swagger UI, otomatik oluşur)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.predict import Predictor, PredictConfig

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("api")

# --------------------------------------------------------------------------- #
# Global durum (uygulama ayağa kalkarken bir kez doldurulur)
# --------------------------------------------------------------------------- #
state: dict = {"predictor": None, "supabase": None}

WEB_INDEX = Path(__file__).resolve().parent.parent / "web" / "index.html"


def _init_supabase():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        logger.warning(
            "SUPABASE_URL / SUPABASE_KEY tanımlı değil (.env dosyasını kontrol et) -> "
            "tahmin loglama DEVRE DIŞI, API yine de çalışmaya devam edecek."
        )
        return None
    try:
        from supabase import create_client
        client = create_client(url, key)
        logger.info("Supabase bağlantısı kuruldu.")
        return client
    except Exception:
        logger.exception("Supabase bağlantısı kurulamadı -> loglama devre dışı, API çalışmaya devam ediyor.")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Predictor yükleniyor (geçmiş veri işleniyor, ~20-30 saniye sürebilir)...")
    state["predictor"] = Predictor(PredictConfig())
    state["supabase"] = _init_supabase()
    logger.info("API hazır.")
    yield
    logger.info("API kapanıyor.")
    state.clear()


app = FastAPI(
    title="CS2 Match Prediction Engine",
    description="Profesyonel CS2 maçları için harita-bazlı galibiyet olasılığı tahmini.",
    version="1.0.0",
    lifespan=lifespan,
)

# React Native / web dashboard'un farklı origin'lerden erişebilmesi için.
# Üretimde allow_origins'i gerçek domain'lerinle sınırlandırmayı unutma.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Şemalar
# --------------------------------------------------------------------------- #
class PredictRequest(BaseModel):
    team1: str = Field(..., examples=["Team Vitality"])
    team2: str = Field(..., examples=["Natus Vincere"])
    map_name: str = Field(..., examples=["Mirage"], alias="map")
    tier: str = Field(default="tier1", examples=["tier1"])
    best_of: int = Field(default=3, examples=[3])

    model_config = {"populate_by_name": True}


class PredictResponse(BaseModel):
    team1: str
    team2: str
    map: str
    tier: str
    best_of: int
    team1_win_probability: float
    team2_win_probability: float
    predicted_winner: str
    model: str
    logged_to_db: bool


class TeamsResponse(BaseModel):
    count: int
    teams: list[str]


class MapsResponse(BaseModel):
    count: int
    maps: list[str]
    as_of: str  # aktif havuzun hesaplandığı tarih (verideki son maç)


class ModelInfoResponse(BaseModel):
    model: str
    data_from: str
    data_to: str
    test_from: str
    test_to: str
    test_metrics: dict[str, float]


# --------------------------------------------------------------------------- #
# Yardımcı: Supabase'e loglama (asla ana akışı bozmamalı)
# --------------------------------------------------------------------------- #
def _log_prediction_to_supabase(result: dict) -> bool:
    supabase = state.get("supabase")
    if supabase is None:
        return False
    try:
        row = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "team1": result["team1"],
            "team2": result["team2"],
            "map_name": result["map"],
            "tier": result["tier"],
            "best_of": result["best_of"],
            "team1_win_probability": result["team1_win_probability"],
            "team2_win_probability": result["team2_win_probability"],
            "predicted_winner": result["predicted_winner"],
            "model_name": result["model"],
        }
        supabase.table("predictions").insert(row).execute()
        return True
    except Exception:
        logger.exception("Tahmin Supabase'e loglanamadı -> tahmin sonucu yine de döndürülüyor.")
        return False


# --------------------------------------------------------------------------- #
# Endpoint'ler
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
def web_ui():
    """Test edenler için tek sayfalık arayüz (web/index.html) -- aşağıdaki API'yi kullanır."""
    return FileResponse(WEB_INDEX)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "predictor_loaded": state.get("predictor") is not None,
        "supabase_connected": state.get("supabase") is not None,
    }


@app.get("/api/v1/teams", response_model=TeamsResponse)
def list_teams():
    """Frontend'de (React Native / web) takım seçim dropdown'ı için kullanılabilir."""
    predictor: Predictor = state["predictor"]
    teams = sorted(predictor.known_teams)
    return TeamsResponse(count=len(teams), teams=teams)


@app.get("/api/v1/maps", response_model=MapsResponse)
def list_maps():
    """Sadece AKTİF harita havuzunu döner (son 60 günde oynanan haritalar, bkz.
    src/predict.py compute_active_map_pool). Havuzdan çıkan haritalar listelenmez."""
    predictor: Predictor = state["predictor"]
    maps = sorted(predictor.active_maps)
    return MapsResponse(count=len(maps), maps=maps, as_of=str(predictor.map_pool_as_of.date()))


@app.get("/api/v1/model", response_model=ModelInfoResponse)
def model_info():
    """Arayüzün altında gösterilen model/veri özeti (model_metadata.json'dan)."""
    meta = state["predictor"].metadata
    metrics = meta["test_metrics_calibrated"]
    return ModelInfoResponse(
        model=meta["champion_model_name"],
        data_from=meta["deployed_train_date_range"][0][:10],
        data_to=meta["deployed_train_date_range"][1][:10],
        test_from=meta["test_date_range"][0][:10],
        test_to=meta["test_date_range"][1][:10],
        test_metrics={k: v for k, v in metrics.items() if isinstance(v, (int, float))},
    )


@app.post("/api/v1/predict", response_model=PredictResponse)
def predict_match(request: PredictRequest):
    predictor: Optional[Predictor] = state.get("predictor")
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model henüz yüklenmedi, birazdan tekrar dene.")

    if request.team1.strip() == request.team2.strip():
        raise HTTPException(status_code=400, detail="team1 ve team2 aynı takım olamaz.")
    if request.best_of not in (1, 3, 5):
        raise HTTPException(status_code=400, detail="best_of yalnızca 1, 3 veya 5 olabilir.")
    # Eğitimde görülmemiş tier modele NaN kategori olarak gider -> sessizce saçma tahmin.
    valid_tiers = predictor.metadata.get("category_levels", {}).get("tier", ["tier1", "tier2", "tier3"])
    if request.tier not in valid_tiers:
        raise HTTPException(status_code=400, detail=f"tier yalnızca {', '.join(valid_tiers)} olabilir.")
    # CLI bilinmeyen takımı lig ortalamasıyla (soğuk başlangıç) tahmin eder; herkese açık
    # API'de yazım hatası sessizce anlamsız bir tahmine dönüşmesin diye reddediyoruz.
    unknown = [t for t in (request.team1, request.team2) if t not in predictor.known_teams]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Bilinmeyen takım: {', '.join(unknown)}. Listeden seç.")

    try:
        result = predictor.predict(
            team1=request.team1,
            team2=request.team2,
            map_name=request.map_name,
            tier=request.tier,
            best_of=request.best_of,
        )
    except ValueError as exc:
        # Kullanıcı kaynaklı hata (ör. havuz dışı harita) -> 400
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        # Ayrıntı sadece sunucu logunda -- herkese açık yanıtta iç hata metni dönmüyoruz.
        logger.exception("Tahmin üretilirken hata oluştu.")
        raise HTTPException(status_code=500, detail="Tahmin üretilemedi, sunucu hatası.") from exc

    logged = _log_prediction_to_supabase(result)

    return PredictResponse(**result, logged_to_db=logged)