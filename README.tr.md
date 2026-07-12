# CS2 Maç Tahmin Motoru

Bu proje, profesyonel bir Counter-Strike 2 maçında bir haritayı hangi takımın
kazanacağını tahmin ediyor. Tek bir Jupyter notebook'ta kalmayıp, gerçek bir ML
sisteminin nasıl kurulacağını denemek için başladı — tek uzun bir script yerine,
veri temizliği, feature engineering, model eğitimi ve API üzerinden tahmin
sunumu için ayrı ayrı aşamalar.

> 🇬🇧 For the English README, see [README.md](README.md).

## Genel Bakış

İki takım ve bir harita ver — mesela Team Vitality vs Natus Vincere, Mirage'da —
her iki taraf için kalibre edilmiş bir galibiyet olasılığı dönsün. Model, ~2.5
yıllık profesyonel maç geçmişiyle eğitilmiş bir gradient boosting sınıflandırıcısı,
bir FastAPI endpoint'i üzerinden sunuluyor, her tahmin de daha sonra incelenebilsin
diye bir Postgres veritabanına (Supabase) kaydediliyor.

```
ham CSV'ler (HLTV maç geçmişi)
   │
   ▼
src/eda_cleaning.py   →  temizlik, şema doğrulama, sızıntı-güvenli hedef türetme
   │
   ▼
src/pipeline.py       →  sızıntısız feature engineering (Elo, form, H2H, kadro
   │                      istikrarı, oyuncu performansı, harita-özel rating'ler...)
   ▼
src/train.py          →  kronolojik train/val/test split, XGBoost / LightGBM /
   │                      CatBoost karşılaştırması, Optuna tuning, olasılık kalibrasyonu
   ▼
src/predict.py         → inference katmanı (train/serve tutarsızlığını önlemek için
   │                      pipeline.py'nin aynı feature fonksiyonlarını yeniden kullanır)
   ▼
api/main.py             → FastAPI servisi, /api/v1/predict, Supabase loglama
```

## Sonuçlar

Mevcut şampiyon model (CatBoost, sigmoid ile kalibre edilmiş), **kesinlikle
kronolojik, hiç görülmemiş bir test döneminde** (Ocak–Haziran 2026) değerlendirildi:

| Metrik | Değer |
|---|---|
| ROC-AUC | 0.653 |
| Log Loss | 0.651 |
| Brier Score | 0.230 |
| Doğruluk (@0.5) | %61.7 |

Bağlam için: profesyonel e-spor bahis piyasaları bu tip tahminlerde genelde
0.60–0.68 ROC-AUC aralığında dolaşıyor — CS2 yüksek varyanslı bir oyun ve
neredeyse-mükemmel tahmin gerçekçi bir hedef değil. Bu modelin neleri yapıp
neleri yapamadığına dair dürüst bir tartışma için aşağıdaki
[Sınırlamalar ve Çıkarılan Dersler](#sınırlamalar-ve-çıkarılan-dersler) bölümüne bakın.

## Veri

Kaynak: HLTV profesyonel maç geçmişi (tier 1-3), **5.134 maç / 10.675 oynanan
harita**, **392 takım** ve **1.398 oyuncu**, **25 Ekim 2023 – 28 Haziran 2026**
aralığında.

Ham CSV'ler bu repoya commitlenmemiştir (bkz. `.gitignore`) — pipeline'ı
çalıştırmadan önce şunları `data/raw/` altına yerleştir:
- `cs2_all_tiers_games.csv`, `cs2_tier1_games.csv`, `cs2_tier2_games.csv`, `cs2_tier3_games.csv`
- `teams.csv`, `players.csv`, `tournaments.csv`

## Feature Engineering

Tüm feature'lar **kesinlikle sızıntısız (leakage-free)** hesaplanacak şekilde
tasarlandı: herhangi bir maç için hesaplanan rolling/expanding istatistik,
SADECE o maçtan *kesinlikle önceki* maçların verisini kullanır. Bu, özel bir
dikkat gerektirdi çünkü kaynak veride aynı seriye (best-of) ait tüm haritalar
birebir aynı zaman damgasını paylaşıyor — saf kronolojik sıralama tek başına,
aynı serinin haritaları arasındaki sızıntıyı önlemeye yetmiyor.

| Grup | Açıklama |
|---|---|
| Team Form | Son 5 / 10 / tüm maçlardaki galibiyet oranı |
| Map Advantage | Takımın o spesifik haritadaki tarihsel galibiyet oranı |
| Head-to-Head | İki takımın birbirine karşı tarihsel galibiyet oranı |
| Player Firepower | Başlangıç kadrosunun rolling ADR / KAST / rating'i |
| Elo Rating | Klasik, iteratif olarak güncellenen güç puanı |
| Harita-özel Elo | Her (takım, harita) çifti için ayrı Elo rating |
| Rest Days | Takımın önceki maçından bu yana geçen gün sayısı |
| Win/Loss Streak | Anlık, işaretli momentum serisi |
| Roster Stability | Bir önceki maça göre kadro örtüşmesi |

## Proje Yapısı

```
├── data/
│   ├── raw/                  # girdi CSV'leri (gitignore'da)
│   └── processed/            # temizlenmiş parquet dosyaları (gitignore'da)
├── src/
│   ├── eda_cleaning.py       # adım 1: temizlik ve doğrulama
│   ├── pipeline.py           # adım 2: feature engineering
│   ├── train.py              # adım 3: model eğitimi ve kalibrasyon
│   └── predict.py            # adım 4: inference
├── api/
│   └── main.py                # FastAPI servisi
├── supabase/
│   └── schema.sql             # predictions tablosu DDL
├── models/                    # eğitilmiş model dosyaları + metadata
├── requirements.txt
└── .env.example
```

## Kurulum

```bash
pip install -r requirements.txt
```

`.env.example` dosyasını `.env` olarak kopyala ve Supabase proje bilgilerini
gir (Project Settings → API).

## Pipeline'ı Çalıştırma

Her aşama sırayla çalıştırılmalı — her aşama bir öncekinin çıktısını okuyor:

```bash
# 1. Ham CSV'leri temizle -> data/processed/{maps_clean,matches_summary}.parquet
python -m src.eda_cleaning --raw-dir data/raw --output-dir data/processed

# 2. Feature engineering -> data/processed/features_engineered.parquet
python -m src.pipeline --processed-dir data/processed --output-dir data/processed

# 3. Eğit + kalibre et + ayarla -> models/champion_model.pkl
python -m src.train --features-path data/processed/features_engineered.parquet --output-dir models

# 4. Komut satırından tahmin al
python -m src.predict --team1 "Team Vitality" --team2 "Natus Vincere" --map Mirage --tier tier1 --bestof 3
```

## API'yi Çalıştırma

```bash
uvicorn api.main:app --reload --port 8000
```

İnteraktif dokümantasyon: `http://localhost:8000/docs`.

| Endpoint | Açıklama |
|---|---|
| `GET /health` | Servis sağlık kontrolü |
| `GET /api/v1/teams` | Bilinen takım isimleri listesi |
| `GET /api/v1/maps` | Bilinen harita isimleri listesi |
| `POST /api/v1/predict` | Maç tahmini üretir; yapılandırılmışsa Supabase'e loglar |

Supabase yapılandırılmamışsa API sorunsuz şekilde çalışmaya devam eder —
tahminler yine döner, sadece loglanmaz (`logged_to_db: false`).

## Sınırlamalar ve Çıkarılan Dersler

Modelin ne yapamadığını da açıkça söylemekte fayda var, sadece ne yaptığını değil.

- **CS2 yüksek varyanslı bir oyun.** ~0.65 ROC-AUC, bu problem için gerçekçi bir
  tavan — modelleme yaklaşımının bir eksikliği değil.
- **Kalibrasyon zamanla kayıyor (concept drift).** Sahne evrimleşiyor — kadrolar
  değişiyor, yeni takımlar yükseliyor. Bir dönemde öğrenilen kalibrasyon eğrisi,
  sonraki bir dönemde mükemmel diyagonal olmayacaktır. Bu varsayılmadı, açıkça
  ölçüldü ve olası bir çözüm olarak cross-fitted kalibrasyon denendi — işe
  yaramadı, bu da nedenin bir kalibrasyon yöntemi sorunu değil, gerçek bir
  dağılım kayması olduğunu doğruladı. Pratik çözüm: veri biriktikçe periyodik
  olarak yeniden eğitmek.
- **Sahne değişse de eski veri hâlâ işe yarıyor.** Farklı eğitim pencerelerini
  (6/12/18 ay vs. tüm geçmiş) karşılaştıran ampirik bir test, tüm ~2.5 yıllık
  geçmişin, yakınlığa göre kırpılmış herhangi bir alt kümeden daha iyi performans
  gösterdiğini ortaya çıkardı — bu veri seti büyüklüğünde örneklem büyüklüğü,
  eskimenin önüne geçti. Veri biriktikçe bu değişebilir.
- **Geliştirme sırasında 2 gerçek veri sızıntısı bug'ı yakalandı** (bunları
  isimlendirmeye değer, çünkü yakalanmaları da mühendislik sürecinin bir
  parçasıydı):
  1. Kaynak veride `team_id`, istikrarlı bir takım kimliği *değil* (aynı takım
     maçlar arasında düzinelerce farklı ID altında görünüyor) — takım
     eşleştirmesi isim üzerinden yapılmak zorunda kaldı.
  2. Harita-özel Elo implementasyonu başlangıçta aynı maçın iki perspektifini
     (team1/team2) tek bir döngüde işliyordu, bu da ikinci satırın birincinin
     *az önce güncellediği* rating'i maç öncesiymiş gibi okumasına ve o maçın
     kendi sonucunun sızmasına yol açtı. Test ROC-AUC'sindeki gerçekçi olmayan
     bir sıçrama (0.65 → 0.78) sayesinde yakalandı ve her maçın tam olarak bir
     kez işlenmesi sağlanarak düzeltildi.
- **Test edildi ama benimsenmedi** (negatif sonuçlar, şeffaflık için burada
  tutuluyor): yakınlık-ağırlıklı eğitim örnekleri, daha geniş Optuna araması
  (100 deneme) ve Optuna ile ayarlanmış CatBoost — hepsi, tutulan test setinde
  daha basit karşılıklarından *daha kötü* performans gösterdi. ~1.600 satırlık
  bir validation setiyle, agresif hiperparametre araması genelleme yapmak yerine
  validation gürültüsüne aşırı uyum sağlama eğiliminde.

## Veri Kaynağı

Veri seti [Kaggle'daki Counter-Strike Pro Matches](https://www.kaggle.com/datasets/ektarr/counter-strike-pro-matches) —
aslen HLTV.org profesyonel maç geçmişinden derlenmiş. Bu repoda yeniden
dağıtılmamaktadır — Kaggle'dan indirip CSV'leri yukarıda anlatıldığı gibi
`data/raw/` altına yerleştir.

## Lisans

Bu proje eğitim/portföy amaçlıdır.
