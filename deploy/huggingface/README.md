---
title: FragPredict
emoji: 🎯
colorFrom: yellow
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: CS2 harita bazlı galibiyet olasılığı tahmini
---

# FragPredict

Profesyonel CS2 maçlarında, seçilen haritada hangi takımın kazanacağını tahmin eden
bir makine öğrenmesi modeli. Kaynak kod ve model detayları: https://github.com/Daisyhaxx/fragpredict-engine

## Veri ve lisans

Model, ektarr'ın Kaggle'da yayımladığı
[Counter-Strike Pro Matches](https://www.kaggle.com/datasets/ektarr/counter-strike-pro-matches)
veri setiyle (aslen HLTV.org maç geçmişi) eğitildi. Veri seti
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) lisanslı.

Bu Space'teki `data/processed/` dosyaları (`maps_clean.parquet`,
`matches_summary.parquet`) o veri setinden türetildi: veri temizlendi, maç ve harita
düzeyinde yeniden düzenlendi. Bu dosyalar da aynı lisansla,
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) ile paylaşılıyor.
`models/` altındaki eğitilmiş model de aynı veriden türetildiği için aynı lisansla
işaretlendi.
