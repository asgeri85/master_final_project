# 00 — Proje Karar Notu

> **Bu belgenin amacı:** Sprint boyunca kapsamı dondurmak ve Claude Code'un her oturumda
> okuyacağı tek kaynak referansı oluşturmak. Buradaki `[KİLİTLİ]` kararlar tartışmaya
> kapalıdır; `[AÇIK]` olanlar ilgili adımda netleşecektir.
>
> **Konum:** `docs/00-karar-notu.md`
> **Son güncelleme:** 2 Ağustos 2026
> **Durum:** Adım 1 tamamlandı

---

## 0. Künye

| Alan                   | Değer                                                                                                           |
| ---------------------- | --------------------------------------------------------------------------------------------------------------- |
| Öğrenci                | İbrahim Asgari (240039008)                                                                                      |
| Program                | İstanbul Gedik Üniversitesi, Yapay Zeka Mühendisliği YL (Tezli)                                                 |
| Tez başlığı            | Mobil Uygulamalarda Kullanıcı Davranışlarını Tahmin Etmek İçin Makine Öğrenmesi Algoritmalarının Geliştirilmesi |
| Kod/proje bitiş hedefi | 1 Eylül 2026                                                                                                    |
| Tez teslim tarihi      | 1 Ocak 2027                                                                                                     |

---

## 1. Araştırma sorusu ve hipotezler `[KİLİTLİ]`

**Araştırma sorusu:** Mobil kullanıcıların sıralı etkileşim verisi üzerinde bir sonraki
eylemi tahmin ederken, sıralı derin öğrenme mimarileri iyi ayarlanmış geleneksel
yöntemlere kıyasla anlamlı bir üstünlük sağlıyor mu — ve bu üstünlük, cihaz üstünde
çalıştırma maliyeti hesaba katıldığında da geçerli mi?

| #      | Hipotez                                                                                                                                                                                                                                      |
| ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **H1** | Sıralı derin öğrenme modelleri (GRU4Rec, LSTM, SASRec), geçmiş davranış dizisini modelleyebildikleri için geleneksel yöntemlere (Item-KNN, SR, XGBoost) kıyasla istatistiksel olarak anlamlı düzeyde daha yüksek sıralama başarısı gösterir. |
| **H2** | Özellik mühendisliği ile zenginleştirilmiş giriş, aynı model ailesinde ham girişe kıyasla sıralama başarısını ölçülebilir biçimde artırır.                                                                                                   |

**Not:** H1'in reddedilmesi geçerli ve değerli bir bulgudur. Literatürde iyi ayarlanmış
basit baseline'ların derin öneri modellerini geçtiği çok sayıda örnek vardır
(Ferrari Dacrema ve ark., 2019; Rendle ve ark., 2019). Deney tasarımının titizliği,
sonucun yönünden daha önemlidir.

### Katkı üçlüsü

- **K1** — Mobil oturum verisinde geleneksel vs. sıralı derin modellerin titiz karşılaştırması _(H1)_
- **K2** — Özellik mühendisliğinin ölçülebilir etkisinin ablasyonla gösterilmesi _(H2)_
- **K3** — On-device dağıtım fizibilitesi: doğruluk–gecikme–boyut ödünleşimi _(ek bölüm, ama tezin "mobil" iddiasının tek dayanağı — feda edilmez)_

---

## 2. Görev tanımı `[KİLİTLİ]`

**Oturum-tabanlı bir sonraki öğe tahmini** (session-based next-item prediction).

Bir oturum `S = (i₁, i₂, …, iₙ₋₁)` verildiğinde model **tüm katalog üzerinde** skor üretir
ve top-K listesi döndürür. Hedef: gerçek `iₙ`'in listede ve mümkün olduğunca üstte olması.

Bu bir **sıralama (ranking)** problemidir; ikili sınıflandırma değildir.

---

## 3. Metrikler `[KİLİTLİ]`

### Model başarısı

- **Birincil:** HR@10 (= Recall@10), NDCG@10, MRR@10
- **İkincil:** HR@1, HR@5, HR@20, Coverage@10, önerilen öğelerin ortalama popülerliği (popülerlik yanlılığı göstergesi)
- **LSApp istisnası:** aday uzayı yalnızca 87 uygulama olduğundan K büyük seçilirse metrik doygunlaşır. LSApp'te **HR@1, HR@3, MRR@5** raporlanır ve bu tercih tezde gerekçelendirilir.

### On-device (K3)

- p50 / p95 çıkarım gecikmesi (ms)
- Model dosya boyutu (MB), FP32 ve INT8
- Tepe bellek kullanımı (MB)
- Kuantizasyon sonrası mutlak NDCG@10 kaybı

### Zorunlu kural: negatif örnekleme YOK

Değerlendirme **tam katalog** üzerinde yapılır. Örneklenmiş negatiflerle hesaplanan
metrikler model sıralamasını bozar (Krichene & Rendle, 2020). Bu, savunmada sorulabilecek
en teknik sorulardan biridir; baştan doğru yapılır.

---

## 4. Veri setleri `[KİLİTLİ]`

### 4.1 Birincil — RetailRocket

| Özellik         | Değer                                                                                |
| --------------- | ------------------------------------------------------------------------------------ |
| Kaynak          | Kaggle — `retailrocket/ecommerce-dataset`                                            |
| Toplama süresi  | 4,5 ay, gerçek e-ticaret sitesi                                                      |
| Toplam olay     | 2.756.101                                                                            |
| Kırılım         | 2.664.312 görüntüleme / 69.332 sepete ekleme / 22.457 satın alma                     |
| Tekil ziyaretçi | 1.407.580                                                                            |
| Erişim          | Kaggle hesabı gerekir (`kaggle datasets download -d retailrocket/ecommerce-dataset`) |

**Seçim gerekçesi:** E-ticaret alanı, mevcut mesleki uzmanlıkla örtüşüyor. Session-based
öneri literatüründe yaygın kullanıldığı için karşılaştırılabilir yayınlanmış skorlar mevcut.
Olay türü çeşitliliği özellik mühendisliğine (K2) zengin zemin sağlıyor.

### 4.2 İkincil — LSApp

| Özellik        | Değer                                                                                                                                                                                             |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Kaynak         | GitHub — `aliannejadi/LSApp` (`lsapp.tsv.gz`, ~7 MB, doğrudan indirilebilir)                                                                                                                      |
| Toplama        | uSearch adlı Android uygulamasıyla, katılımcı onayıyla, 8 aylık dönem                                                                                                                             |
| Kullanıcı      | 292                                                                                                                                                                                               |
| Olay           | 599.635                                                                                                                                                                                           |
| Tekil uygulama | 87                                                                                                                                                                                                |
| Atıf           | Aliannejadi, M., Zamani, H., Crestani, F., & Croft, W. B. (2021). _Context-Aware Target Apps Selection and Recommendation for Enhancing Personal Mobile Assistants._ ACM TOIS, 39(3), Article 29. |

**Seçim gerekçesi:** Tezin "mobil" iddiasını akademik olarak savunan tek veri seti —
gerçek Android cihazlardan toplanmış gerçek kullanım dizileri. Küçük katalog boyutu
on-device deneyleri için ayrıca avantaj.

**Bilinen sınırı:** 87 öğelik aday uzayı. Bölüm 3'teki K düzeltmesi bu yüzden zorunlu.

### 4.3 İlgili çalışma (K3 için doğrudan karşılaştırma noktası)

_Federated Privacy-preserving Collaborative Filtering for On-Device Next App Prediction_
(arXiv:2303.04744) — LSApp üzerinde cihaz-üstü sonraki-uygulama tahmini yapıyor.
Literatür taramasında ve K3 tartışmasında mutlaka yer almalı.

---

## 5. Model seti `[KİLİTLİ — 9 model, ekleme yok]`

| Katman | Model                 | Rol                                                               |
| ------ | --------------------- | ----------------------------------------------------------------- |
| 0      | POP                   | Akıl sağlığı kontrolü (global popülerlik)                         |
| 0      | S-POP                 | Oturum içi popülerlik                                             |
| 1      | Item-KNN              | Güçlü klasik baseline (birlikte görülme)                          |
| 1      | Sequential Rules (SR) | Güçlü klasik baseline (sıralı kural)                              |
| 2      | Random Forest         | Öznitelik tabanlı yeniden sıralama                                |
| 2      | XGBoost               | Öznitelik tabanlı yeniden sıralama — **H2'nin test edildiği yer** |
| 3      | GRU4Rec               | Sıralı derin öğrenme                                              |
| 3      | LSTM                  | Sıralı derin öğrenme                                              |
| 4      | SASRec                | Transformer temsilcisi                                            |

### Katman 2 tasarım kararı (önemli)

Ağaç tabanlı modeller doğrudan katalog üzerinde sıralama yapamaz. Kurgu:

```
Item-KNN → top-100 aday üret → XGBoost/RF öznitelikleriyle yeniden sırala → top-K
```

Bu kurgu hem endüstri standardı (candidate generation → re-ranking) hem de H2'yi temiz
test etmeyi sağlar: **aynı aday kümesi**, özniteliksiz vs. öznitelikli.

---

## 6. Hipotez test planı `[KİLİTLİ]`

| Hipotez | Karşılaştırma                                       | Test                                                                       |
| ------- | --------------------------------------------------- | -------------------------------------------------------------------------- |
| H1      | Katman 3–4 vs. Katman 1–2                           | Oturum bazlı eşleştirilmiş Wilcoxon signed-rank, HR@10 ve NDCG@10 üzerinde |
| H2      | XGBoost (ham) vs. XGBoost (mühendislik öznitelikli) | Aynı test, ablasyon kurgusu                                                |

- Çoklu karşılaştırma düzeltmesi: **Holm–Bonferroni**
- Derin modeller **5 farklı tohum (seed)** ile eğitilir; ortalama ± standart sapma raporlanır
- Deterministik modeller (POP, S-POP, Item-KNN, SR) tek çalıştırma
- Tüm tohumlar, config'ler ve rastgelelik kaynakları kayıt altına alınır

---

## 7. Altyapı ve ortam `[KİLİTLİ]`

### Eğitim

| Kaynak        | Notlar            |
| ------------- | ----------------- |
| **Colab Pro** | Ana eğitim ortamı |

Colab Pro kısıtları için zorunlu tedbirler:

1. **Checkpoint zorunlu.** Her epoch sonunda Google Drive'a kaydet. Oturum kopması varsayılan senaryo kabul edilir, istisna değil.
2. **Veri Drive'da tutulur.** Ön işlenmiş veri `.parquet` olarak Drive'a yazılır; her oturumda sıfırdan işlenmez.
3. **Eğitim betikleri notebook'tan bağımsız çalışabilmeli.** Notebook yalnızca ince bir sarmalayıcıdır; tüm mantık `src/` altında `.py` dosyalarında durur. Bu, tekrarlanabilirlik için de gereklidir.
4. **Hiperparametre bütçesi:** Optuna, model başına **20 deneme**. Aşılmaz.
5. **Deney kaydı:** her koşu için config + metrik + süre, tek bir `results/runs.jsonl` dosyasına eklenir.

### Cihaz üstü test

| Kaynak                              | Kullanım                                                                 |
| ----------------------------------- | ------------------------------------------------------------------------ |
| Birden fazla fiziksel Android cihaz | **Gecikme ve bellek ölçümünün tek geçerli kaynağı**                      |
| Android Studio emülatörleri         | Yalnızca fonksiyonel doğrulama — çıktı eşleşiyor mu, model yükleniyor mu |

> **Protokol kuralı:** Emülatörden alınan gecikme ölçümü teze **girmez**. Emülatör ana
> makinenin x86 CPU'sunda çalışır; ölçülen süre gerçek ARM cihaz performansını temsil
> etmez. Bunu tezin sınırlılıklar bölümünde de belirt.

Cihazlar tezde **anonim segment etiketleriyle** raporlanır (Cihaz A / B / C) ve her biri
için SoC, RAM, Android sürümü tablolaştırılır. En az üç segment hedeflenir: üst, orta, giriş.

---

## 8. On-device protokolü (K3) `[AÇIK — Adım 4'te netleşecek]`

Ön kararlar:

- **Runtime:** LiteRT (eski adıyla TensorFlow Lite) — birincil aday
- **Dönüşüm:** FP32 → dinamik menzilli INT8 → tam tamsayı INT8 (üç varyant karşılaştırılır)
- **Aday modeller:** GRU4Rec ve LSTM birincil; SASRec ikincil (dönüşüm zorluğu yüksek)
- **Ölçüm:** cihaz başına en az 200 çıkarım, ilk 20 ısınma koşusu atılır, p50/p95 raporlanır
- **Termal kontrol:** ölçümler arasında cihaz soğuması beklenir; termal kısıtlama (throttling) sonuçları bozar

Netleşmesi gerekenler: gömme (embedding) katmanının kuantizasyon davranışı, katalog
boyutunun model boyutuna etkisi, ölçüm harness'ının nasıl kurulacağı.

---

## 9. Sprint planı — 4 hafta

| Hafta | Tarih     | Çıktı                                                                                           | Bitiş kriteri                    |
| ----- | --------- | ----------------------------------------------------------------------------------------------- | -------------------------------- |
| **1** | 3–9 Ağu   | Veri indirme, EDA, ön işleme pipeline, zaman-temelli split, değerlendirme harness'ı, Katman 0–1 | **İlk sonuç tablosu masada**     |
| **2** | 10–16 Ağu | Özellik mühendisliği, RF + XGBoost, H2 ablasyonu                                                | H2 sonucu elde, **kapsam donar** |
| **3** | 17–23 Ağu | GRU4Rec, LSTM, SASRec + Optuna                                                                  | H1 sonucu elde                   |
| **4** | 24–31 Ağu | LSApp replikasyonu, on-device dönüşüm ve ölçüm, final tablo/grafikler                           | **Repo dondurulur**              |

### Geride kalınırsa feda sırası

1. LSApp replikasyonu
2. SASRec
3. Random Forest (XGBoost yeterli)

**On-device bölümü feda edilmez.**

---

## 10. Kapsam dondurma kuralları `[KİLİTLİ]`

1. **2. hafta sonundan sonra yeni model eklenmez.**
2. **Yeni veri seti eklenmez.**
3. **Yeni metrik eklenmez.** (Mevcutlar zaten fazlasıyla yeterli.)
4. "Şunu da bir denesek" fikirleri `docs/99-sonraki-calismalar.md` dosyasına yazılır ve tezin _Gelecek Çalışmalar_ bölümüne gider — koda girmez.
5. Bir deney başarısız olursa **silinmez, raporlanır.** Negatif sonuç da bulgudur.

---

## 11. Değerlendirme protokolü `[KİLİTLİ — Adım 2]`

| #   | Konu                | Karar                                                                                                          |
| --- | ------------------- | -------------------------------------------------------------------------------------------------------------- |
| A1  | Oturum tanımı       | İki olay arası **30 dakika** boşluk = yeni oturum                                                              |
| A2  | Filtreleme eşikleri | Min. oturum uzunluğu: **2 olay**. Min. öğe frekansı: **5** (toplamda)                                          |
| A3  | Split               | Zaman-temelli: son 1 hafta → test, önceki 1 hafta → validation, kalan → train. **Rastgele split kullanılmaz.** |
| A4  | Cold-start          | Test setinden çıkarılmaz; ayrı bir "cold-start alt-küme performansı" tablosunda raporlanır                     |

## 12a. Öznitelik listesi ve hiperparametre uzayları `[KİLİTLİ — Adım 3]`

### A5 — Öznitelik listesi (H2 ablasyonu)

| Grup            | Öznitelikler                                                                                   |
| --------------- | ---------------------------------------------------------------------------------------------- |
| Popülerlik      | Öğenin global görülme sayısı, son 7 gündeki görülme sayısı                                     |
| Birlikte-oluşum | Item-KNN aday skoru, oturumdaki son öğeyle ortak oturum sayısı                                 |
| Zamansal        | Oturumdaki geçen süre, günün saati, haftanın günü                                              |
| Oturum içi      | Oturum uzunluğu (şu ana kadar), aynı öğenin tekrar görülme sayısı, oturumdaki eşsiz öğe sayısı |

Ablasyon: XGBoost-ham (yalnız Item-KNN skoru) vs. XGBoost-zengin (12 öznitelik).

### A6 — Hiperparametre uzayları (Optuna, model başına 20 deneme)

| Model          | Aranan parametreler                                                                   |
| -------------- | ------------------------------------------------------------------------------------- |
| Item-KNN       | k (10–500, log)                                                                       |
| XGBoost / RF   | max_depth (3–10), n_estimators (100–500), learning_rate (0.01–0.3, sadece XGB)        |
| GRU4Rec / LSTM | gizli boyut (64–256), katman (1–2), dropout (0.1–0.5), öğrenme oranı (1e-4–1e-2, log) |
| SASRec         | dikkat başlığı (1–4), blok (1–3), gizli boyut (64–256), dropout (0.1–0.5)             |

Sabit: batch size 256, max sequence length 50, early stopping (val NDCG@10, patience=5).

## 12c. On-device ölçüm harness'ı `[KİLİTLİ — Adım 4]`

| Parça            | Karar                                                                                           |
| ---------------- | ----------------------------------------------------------------------------------------------- |
| Dönüşüm          | GRU4Rec ve LSTM → LiteRT (.tflite): FP32 / dinamik INT8 / tam INT8                              |
| Ölçüm uygulaması | Basit Android test app: modeli yükle, sabit örnek oturum, 220 çalıştırma (ilk 20 ısınma atılır) |
| Kayıt            | Her koşu → cihaz, varyant, p50/p95 gecikme, tepe bellek, model boyutu → tek CSV                 |
| Cihazlar         | Fiziksel cihazlar (üst/orta/giriş) tek tek; emülatör yalnız fonksiyonel kontrol                 |
| SASRec           | Dönüşümü zor, opsiyonel — feda sırasında 2.                                                     |

## 12e. Repo mimarisi ve deney izleme `[KİLİTLİ — Adım 5]`

```
thesis-project/
├── docs/00-karar-notu.md
├── data/{raw,processed}/     # raw .gitignore'da
├── src/{data,features,models,eval,ondevice}/
├── configs/*.yaml
├── notebooks/colab_runner.ipynb   # ince sarmalayıcı, mantık içermez
├── results/runs.jsonl              # her koşu: config+metrik+süre
├── requirements.txt
└── CLAUDE.md
```

Kural: her deney `runs.jsonl`'e satır ekler; config YAML'da, koda hiperparametre gömülmez.

## 12f. `CLAUDE.md` ve prompt kütüphanesi `[KİLİTLİ — Adım 6]`

`CLAUDE.md` ve `prompt-kutuphanesi.md` dosyaları oluşturuldu (bkz. proje çıktıları).
İçerik: kesin kurallar, durup-sor eşikleri, kod stili, haftalık hazır prompt seti.

## 13. Planlama tamamlandı

Adım 0–6 kilitli. Sıradaki iş kodlamaya başlamak: repoyu oluştur, GitHub'a it,
`CLAUDE.md` ve karar notunu kök dizine koy, Hafta 1 promptlarıyla başla.

---

## 14. Etik ve uyum notları

- Kullanılan iki veri seti de **kamuya açık ve anonimleştirilmiş**tir; birincil veri toplama yapılmayacaktır.
- LSApp verisi katılımcı onayıyla toplanmıştır; kullanım koşulları kaynak repoda belirtilmiştir.
- İşveren verisi (BTPRO / müşteri uygulamaları) **hiçbir biçimde kullanılmayacaktır** — KVKK ve iş sözleşmesi açısından risklidir.
- Tez metnindeki tüm karakterler standart Unicode noktalama olmalıdır. Öneri formunda görülen `․` (U+2024) ve `‚` (U+201A) karakter ikameleri temizlenmelidir; bu ikameler benzerlik yazılımlarınca **kasıtlı manipülasyon** olarak değerlendirilebilir.
- Kaynakçadaki her künye, tez yazımına başlamadan önce DOI üzerinden tek tek doğrulanacaktır.

---

## 15. Çekirdek kaynakça (metodolojik dayanak)

| Kaynak                                                                                                  | Neden gerekli                                          |
| ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| Hidasi, B. ve ark. (2016). _Session-based Recommendations with Recurrent Neural Networks._ ICLR.        | GRU4Rec'in orijinal kaynağı                            |
| Kang, W.-C., & McAuley, J. (2018). _Self-Attentive Sequential Recommendation._ ICDM.                    | SASRec'in orijinal kaynağı                             |
| Ferrari Dacrema, M., Cremonesi, P., & Jannach, D. (2019). _Are We Really Making Much Progress?_ RecSys. | H1'in reddedilebilirliğinin literatür dayanağı         |
| Rendle, S. ve ark. (2019). _On the Difficulty of Evaluating Baselines._ arXiv.                          | Baseline ayarlamanın kritikliği                        |
| Krichene, W., & Rendle, S. (2020). _On Sampled Metrics for Item Recommendation._ KDD.                   | Negatif örnekleme yasağının dayanağı                   |
| Ludewig, M., & Jannach, D. (2018). _Evaluation of Session-based Recommendation Algorithms._ UMUAI.      | Item-KNN ve SR baseline'larının kaynağı                |
| Aliannejadi, M. ve ark. (2021). ACM TOIS, 39(3).                                                        | LSApp veri seti                                        |
| arXiv:2303.04744                                                                                        | On-device sonraki-uygulama tahmini, K3 karşılaştırması |

> Bu listedeki her künye Adım 1 kapanışında DOI ile doğrulanacaktır.
