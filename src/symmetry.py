"""
src/symmetry.py
----------------
Takım sırasından bağımsız tahmin: "A vs B" ile "B vs A" aynı maçtır, aynı sonucu almalı.

Sorun: kaynak veride team1/team2 sırası rastgele değil (team1 haritaların ~%53'ünü
kazanıyor, maçların ~%60'ında Elo'su daha yüksek). Sadece bu sırayla eğitilen model
"1. sıradaki takıma" ortalama ~5 puanlık bir bonus öğreniyordu -- oysa özellikler
hesaba katıldıktan sonra bu bonusun veride karşılığı yok. Web/CLI'da sırayı kullanıcı
seçtiği için aynı maç, sıraya göre farklı (maçların ~%15'inde farklı favorili) tahmin
alıyordu.

Çözüm iki adımlı:
    1. Eğitimde her satırın AYNA kopyası eklenir (takımlar yer değiştirmiş, etiket
       ters) -> model 1. sıraya bonus vermeyi öğrenmez.
    2. Tahminde iki sıranın ortalaması alınır. Ağaç modelleri matematiksel olarak tam
       simetrik olamadığından kalan küçük farkı (~1-2 puan) bu adım tam sıfırlar:
           P(A kazanır) = (f(A,B) + 1 - f(B,A)) / 2   =>   P(A,B) + P(B,A) = 1

DİKKAT (leakage): ayna satırlar split'ten SONRA, her bölümün (train/val/CV fold'u)
kendi içinde üretilmeli -- bir test haritasının aynası eğitime girerse model o
haritanın sonucunu görmüş olur.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# team2_ karşılığı olmayan, team1 perspektifinden tek taraflı kolonlar
ONE_SIDED_MIRROR = {
    "team1_h2h_win_rate": lambda s: 1 - s,    # karşı tarafın aynı rakibe karşı kazanma oranı
    "team1_h2h_matches_played": lambda s: s,  # karşılaşma sayısı iki taraf için aynı
}


def mirror_features(X: pd.DataFrame) -> pd.DataFrame:
    """Takımların yerini değiştirir: team1_* <-> team2_*, diff_* -> -diff_*.
    Maç bağlamı (map_name, tier, bestOf) değişmez. Kuralı bilinmeyen tek taraflı bir
    kolon görürse sessizce yanlış ayna üretmek yerine hata verir."""
    M = X.copy()
    for col in X.columns:
        if col.startswith("team1_"):
            other = "team2_" + col.removeprefix("team1_")
            if other in X.columns:
                M[col], M[other] = X[other], X[col]
            elif col in ONE_SIDED_MIRROR:
                M[col] = ONE_SIDED_MIRROR[col](X[col])
            else:
                raise ValueError(f"'{col}' için ayna kuralı tanımlı değil (ONE_SIDED_MIRROR'a ekle).")
        elif col.startswith("team2_"):
            if "team1_" + col.removeprefix("team2_") not in X.columns:
                raise ValueError(f"'{col}' için ayna kuralı tanımlı değil.")
        elif col.startswith("diff_"):
            M[col] = -X[col]
    return M


def augment_with_mirror(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Her satırın aynasını (etiketi ters) ekler. Split'ten sonra, tek bir bölüme uygula."""
    X_aug = pd.concat([X, mirror_features(X)], ignore_index=True)
    y_aug = pd.concat([y, 1 - y], ignore_index=True)
    return X_aug, y_aug


def symmetric_proba(model, X: pd.DataFrame) -> np.ndarray:
    """team1'in kazanma olasılığı, iki sıranın ortalaması olarak (sıradan bağımsız)."""
    p = model.predict_proba(X)[:, 1]
    p_mirror = model.predict_proba(mirror_features(X))[:, 1]
    return (p + 1 - p_mirror) / 2


def order_asymmetry(model, X: pd.DataFrame) -> float:
    """Ortalamadan ÖNCE modelin kendi tutarsızlığı: ortalama |f(A,B) + f(B,A) - 1|.
    Teşhis içindir -- symmetric_proba'nın çıktısında bu değer tanım gereği 0."""
    p = model.predict_proba(X)[:, 1]
    p_mirror = model.predict_proba(mirror_features(X))[:, 1]
    return float(np.abs(p + p_mirror - 1).mean())
