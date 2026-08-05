# CLAUDE.md — Proje Talimatları

> Bu dosya Claude Code'un her oturumda ilk okuduğu belgedir. Buradaki kurallar
> tartışmasızdır. Detaylı gerekçeler için `docs/00-karar-notu.md`'ye bak.

## Proje özeti

Mobil kullanıcı davranışında oturum-tabanlı bir sonraki öğe tahmini. 9 model
(POP, S-POP, Item-KNN, SR, RF, XGBoost, GRU4Rec, LSTM, SASRec), 2 veri seti
(RetailRocket, LSApp), 1 on-device fizibilite bölümü.

## Kesin kurallar

1. **Kapsam donmuştur.** Yeni model, yeni veri seti, yeni metrik ekleme. Bir fikir
   gelirse `docs/99-sonraki-calismalar.md`'ye yaz, koda dokunma.
2. **Negatif örnekleme yok.** Tüm değerlendirmeler tam katalog üzerinde yapılır.
3. **Rastgele split yasak.** Zaman-temelli split zorunlu (bkz. karar notu §11).
4. **Her deney `results/runs.jsonl`'e yazılır.** Elle not tutma, kaydı otomatikleştir.
5. **Config'ler YAML'da, kodda gömülü hiperparametre olmaz.**
6. **Emülatör gecikme ölçümü asla rapor edilmez.** Sadece fonksiyonel test.
7. **Colab'da kod yazılmaz, sadece çalıştırılır.** Mantık her zaman `src/` altında.
8. **BTPRO / müşteri verisi bu projede kesinlikle kullanılmaz.**
9. **Deterministik olmayan modeller (GRU4Rec, LSTM, SASRec) 5 seed ile eğitilir.**

## Ne zaman durup sor

- Bir model beklenenden çok daha kötü/iyi performans gösteriyorsa (bug şüphesi) — durur, sorarım.
- Bir hiperparametre araması 20 denemeyi aşacak gibiyse — durur, sorarım.
- Kapsam dışı bir şey eklenmesi gerekiyor gibi görünüyorsa — durur, sorarım.
- Aksi halde: karar notundaki kilitli kararları uygula, ilerle.

## Kod stili

- Python 3.11+, type hints zorunlu.
- Her model dosyası aynı arayüzü uygular: `fit(train)`, `predict(session) -> ranked_items`.
- Docstring'ler İngilizce, yorum satırları Türkçe olabilir (öğrenci kendi kodunu okuyacak).
- Test yok denemesi yapılmaz — en azından `eval/metrics.py` için birim testleri olacak (HR@K, NDCG@K, MRR elle hesaplanmış örneklerle doğrulanacak).

## Dizin haritası

Bkz. `docs/00-karar-notu.md` §12e (repo mimarisi).
