# Prompt Kütüphanesi — Claude Code için

> Sprint boyunca kullanacağın hazır Türkçe promptlar. Köşeli parantez içini doldur.

## Hafta 1 — Veri ve baseline

```
RetailRocket veri setini [data/raw/retailrocket/] altına indir ve oku.
docs/00-karar-notu.md'deki A1-A4 kararlarına göre (30 dk oturum eşiği,
zaman-temelli split, min 5 frekans filtresi) ön işleme pipeline'ını
src/data/ altında yaz. Sonunda train/val/test parquet dosyalarının
boyutlarını ve örnek satır sayısını raporla.
```

```
src/eval/metrics.py altında HR@K, NDCG@K, MRR@K fonksiyonlarını yaz.
Negatif örnekleme YOK — tam katalog üzerinde sıralama varsayımıyla.
Elle hesapladığım 3 örnek üzerinde birim testi yaz ve çalıştır.
```

```
POP, S-POP ve Item-KNN modellerini src/models/ altında, CLAUDE.md'deki
ortak arayüze (fit/predict) uyacak şekilde yaz. RetailRocket validation
setinde çalıştır, sonucu results/runs.jsonl'e ekle.
```

## Hafta 2 — Öznitelik mühendisliği ve H2

```
docs/00-karar-notu.md §12a'daki 12 özniteliği src/features/ altında
üret. Item-KNN'den top-100 aday üretip bu özniteliklerle zenginleştiren
bir aday tablosu oluştur.
```

```
XGBoost-ham (sadece Item-KNN skoru) ve XGBoost-zengin (12 öznitelik)
modellerini eğit, Optuna ile 20 deneme sınırında ayarla. İkisinin
HR@10/NDCG@10 farkını Wilcoxon signed-rank testiyle karşılaştır,
Holm düzeltmesi uygula. Sonucu raporla.
```

## Hafta 3 — Derin öğrenme

```
GRU4Rec'i [PyTorch/TensorFlow — tercihini yaz] ile src/models/gru4rec.py
altında yaz. Optuna arama uzayı: docs/00-karar-notu.md §12a A6'ya göre.
5 farklı seed ile eğit, ortalama±std raporla.
```

```
SASRec'i aynı arayüze uygun şekilde yaz. GRU4Rec ile aynı protokolü
(5 seed, Optuna 20 deneme) uygula.
```

```
H1 testini çalıştır: Katman 3-4 (derin) vs Katman 1-2 (klasik),
Wilcoxon signed-rank + Holm düzeltmesi. Sonuç tablosunu markdown
formatında yaz.
```

## Hafta 4 — LSApp, on-device, final

```
Aynı pipeline'ı LSApp veri setine uygula. K değerlerini
docs/00-karar-notu.md'deki gibi küçült (HR@1, HR@3, MRR@5).
```

```
GRU4Rec ve LSTM modellerini LiteRT'e dönüştür: FP32, dinamik INT8,
tam INT8 üç varyant. src/ondevice/ altında dönüşüm scriptini yaz.
```

```
Android'de basit bir ölçüm test app'i oluştur: modeli yükle, sabit
oturum örneğiyle 220 kez çalıştır (ilk 20 ısınma), p50/p95 gecikme +
bellek + model boyutunu CSV'ye yaz.
```

```
Tüm sonuçları (H1, H2, on-device) tek bir final tablo ve 2-3 grafikte
topla. results/ altındaki runs.jsonl'den üret, elle sayı girme.
```

## Genel amaçlı

```
Şu an [X] adımındayız. docs/00-karar-notu.md'yi oku ve neyin kilitli,
neyin açık olduğunu özetle.
```

```
Bu deney beklenenden çok farklı sonuç verdi: [açıkla]. Olası bug'ları
kontrol et, durup bana sor — kapsam dışına çıkma.
```
