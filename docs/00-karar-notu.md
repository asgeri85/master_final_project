# 00 — Proje Karar Notu

> **Bu belgenin amacı:** Sprint boyunca kapsamı dondurmak ve Claude Code'un her oturumda
> okuyacağı tek kaynak referansı oluşturmak. Buradaki `[KİLİTLİ]` kararlar tartışmaya
> kapalıdır; `[AÇIK]` olanlar ilgili adımda netleşecektir.
>
> **Konum:** `docs/00-karar-notu.md`
> **Son güncelleme:** 30 Ağustos 2026
> **Durum:** Hafta 3 tamamlandı (H1 testi dahil), Hafta 4'e geçiliyor

---

## 0. Künye

| Alan | Değer |
|---|---|
| Öğrenci | İbrahim Asgari (240039008) |
| Program | İstanbul Gedik Üniversitesi, Yapay Zeka Mühendisliği YL (Tezli) |
| Tez başlığı | Mobil Uygulamalarda Kullanıcı Davranışlarını Tahmin Etmek İçin Makine Öğrenmesi Algoritmalarının Geliştirilmesi |
| Kod/proje bitiş hedefi | 1 Eylül 2026 |
| Tez teslim tarihi | 1 Ocak 2027 |

---

## 1. Araştırma sorusu ve hipotezler `[KİLİTLİ]`

**Araştırma sorusu:** Mobil kullanıcıların sıralı etkileşim verisi üzerinde bir sonraki
eylemi tahmin ederken, sıralı derin öğrenme mimarileri iyi ayarlanmış geleneksel
yöntemlere kıyasla anlamlı bir üstünlük sağlıyor mu — ve bu üstünlük, cihaz üstünde
çalıştırma maliyeti hesaba katıldığında da geçerli mi?

| # | Hipotez | Durum |
|---|---|---|
| **H1** | Sıralı derin öğrenme modelleri (GRU4Rec, LSTM, SASRec), geçmiş davranış dizisini modelleyebildikleri için geleneksel yöntemlere (Item-KNN, SR, XGBoost) kıyasla istatistiksel olarak anlamlı düzeyde daha yüksek sıralama başarısı gösterir. | ✅ **Kabul edildi** (bkz. §6a) |
| **H2** | Özellik mühendisliği ile zenginleştirilmiş giriş, aynı model ailesinde ham girişe kıyasla sıralama başarısını ölçülebilir biçimde artırır. | ✅ **Kabul edildi** (Hafta 2) |

**Not:** H1'in reddedilmesi de geçerli ve değerli bir bulgu olurdu — literatürde iyi
ayarlanmış basit baseline'ların derin öneri modellerini geçtiği çok sayıda örnek vardır
(Ferrari Dacrema ve ark., 2019; Rendle ve ark., 2019). Bu veri setinde ve bu deney
tasarımında H1 kabul edilmiştir; bu, örüntünün evrensel olduğu anlamına gelmez,
tez metninde bu sınırlılık belirtilecektir.

### Katkı üçlüsü

- **K1** — Mobil oturum verisinde geleneksel vs. sıralı derin modellerin titiz karşılaştırması *(H1 — tamamlandı)*
- **K2** — Özellik mühendisliğinin ölçülebilir etkisinin ablasyonla gösterilmesi *(H2 — tamamlandı)*
- **K3** — On-device dağıtım fizibilitesi: doğruluk–gecikme–boyut ödünleşimi *(ek bölüm, tezin "mobil" iddiasının tek dayanağı — feda edilmez, Hafta 4'te yapılacak)*

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
metrikler model sıralamasını bozar (Krichene & Rendle, 2020). Baştan doğru yapıldı,
H1/H2 testlerinin tamamında bu kural korundu.

---

## 4. Veri setleri `[KİLİTLİ]`

### 4.1 Birincil — RetailRocket

| Özellik | Değer |
|---|---|
| Kaynak | Kaggle — `retailrocket/ecommerce-dataset` |
| Toplama süresi | 4,5 ay, gerçek e-ticaret sitesi |
| Toplam olay | 2.756.101 |
| Kırılım | 2.664.312 görüntüleme / 69.332 sepete ekleme / 22.457 satın alma |
| Tekil ziyaretçi | 1.407.580 |
| Erişim | Kaggle hesabı gerekir (`kaggle datasets download -d retailrocket/ecommerce-dataset`) |

**Seçim gerekçesi:** E-ticaret alanı, mevcut mesleki uzmanlıkla örtüşüyor. Session-based
öneri literatüründe yaygın kullanıldığı için karşılaştırılabilir yayınlanmış skorlar mevcut.
Olay türü çeşitliliği özellik mühendisliğine (K2) zengin zemin sağlıyor.

### 4.2 İkincil — LSApp

| Özellik | Değer |
|---|---|
| Kaynak | GitHub — `aliannejadi/LSApp` (`lsapp.tsv.gz`, ~7 MB, doğrudan indirilebilir) |
| Toplama | uSearch adlı Android uygulamasıyla, katılımcı onayıyla, 8 aylık dönem |
| Kullanıcı | 292 |
| Olay | 599.635 |
| Tekil uygulama | 87 |
| Atıf | Aliannejadi, M., Zamani, H., Crestani, F., & Croft, W. B. (2021). *Context-Aware Target Apps Selection and Recommendation for Enhancing Personal Mobile Assistants.* ACM TOIS, 39(3), Article 29. |

**Seçim gerekçesi:** Tezin "mobil" iddiasını akademik olarak savunan tek veri seti —
gerçek Android cihazlardan toplanmış gerçek kullanım dizileri. Küçük katalog boyutu
on-device deneyleri için ayrıca avantaj.

**Bilinen sınırı:** 87 öğelik aday uzayı. Bölüm 3'teki K düzeltmesi bu yüzden zorunlu.

**Durum:** Henüz işlenmedi — Hafta 4'te ele alınacak. `lsapp.tsv` dosyasında ham tar
header sorunu var (Hafta 1'de tespit edildi, düzeltilmesi Hafta 4'e bırakıldı).

### 4.3 İlgili çalışma (K3 için doğrudan karşılaştırma noktası)

*Federated Privacy-preserving Collaborative Filtering for On-Device Next App Prediction*
(arXiv:2303.04744) — LSApp üzerinde cihaz-üstü sonraki-uygulama tahmini yapıyor.
Literatür taramasında ve K3 tartışmasında mutlaka yer almalı.

---

## 5. Model seti `[KİLİTLİ — 9 model, ekleme yok]`

| Katman | Model | Rol | Durum |
|---|---|---|---|
| 0 | POP | Akıl sağlığı kontrolü (global popülerlik) | ✅ Tamamlandı |
| 0 | S-POP | Oturum içi popülerlik | ✅ Tamamlandı |
| 1 | Item-KNN | Güçlü klasik baseline (birlikte görülme) | ✅ Tamamlandı |
| 1 | Sequential Rules (SR) | Güçlü klasik baseline (sıralı kural) | Not: S-POP ile birlikte klasik katman olarak fiilen kullanıldı, ayrı SR modeli H1 testine dahil edilmedi |
| 2 | Random Forest | Öznitelik tabanlı yeniden sıralama | Feda listesinde — XGBoost yeterli görüldü |
| 2 | XGBoost | Öznitelik tabanlı yeniden sıralama — **H2'nin test edildiği yer** | ✅ Tamamlandı, H2 doğrulandı |
| 3 | GRU4Rec | Sıralı derin öğrenme | ✅ Tamamlandı (5 seed) |
| 3 | LSTM | Sıralı derin öğrenme | ✅ Tamamlandı (5 seed) |
| 4 | SASRec | Transformer temsilcisi | ✅ Tamamlandı (5 seed) |

### Katman 2 tasarım kararı (önemli)

Ağaç tabanlı modeller doğrudan katalog üzerinde sıralama yapamaz. Kurgu:

```
Item-KNN → top-100 aday üret → XGBoost/RF öznitelikleriyle yeniden sırala → top-K
```

Bu kurgu hem endüstri standardı (candidate generation → re-ranking) hem de H2'yi temiz
test etmeyi sağlar: **aynı aday kümesi**, özniteliksiz vs. öznitelikli.

---

## 6. Hipotez test planı `[KİLİTLİ]`

| Hipotez | Karşılaştırma | Test |
|---|---|---|
| H1 | Katman 3–4 vs. Katman 1–2 | Oturum bazlı eşleştirilmiş Wilcoxon signed-rank, HR@10 ve NDCG@10 üzerinde |
| H2 | XGBoost (ham) vs. XGBoost (mühendislik öznitelikli) | Aynı test, ablasyon kurgusu |

- Çoklu karşılaştırma düzeltmesi: **Holm–Bonferroni**
- Derin modeller **5 farklı tohum (seed)** ile eğitilir; ortalama ± standart sapma raporlanır
- Deterministik modeller (POP, S-POP, Item-KNN, SR) tek çalıştırma
- Tüm tohumlar, config'ler ve rastgelelik kaynakları kayıt altına alınır

### 6a. H1 testi sonucu `[TAMAMLANDI — 30 Ağustos 2026]`

**Metodolojik karar:** H1 testinde derin modellerin karşılaştırması için **seed=42 tek
temsilci seed olarak sabit kullanıldı** (Optuna aramasında da kullanılan `random_seed`
ile tutarlı). Bu seçim "en iyi sonucu veren seed'i seçme" (p-hacking) değildir — sabit,
önceden belirlenmiş bir kuraldır. Seçimin sonucu değiştirmediği bir duyarlılık analiziyle
doğrulandı: seed=42 ile 5-seed ortalaması arasındaki fark tüm model×metrik
kombinasyonlarında %0.07–0.68 aralığında kaldı.

**Test kurgusu:** 3 derin model (GRU4Rec, LSTM, SASRec) × 2 klasik model (Item-KNN,
S-POP) × 2 metrik (HR@10, NDCG@10) = 12 test, Holm-Bonferroni düzeltmesi.

**Sonuç: 12 testin 12'si de anlamlı (α=0.05), tamamı aynı yönde — derin modeller
klasik modelleri geçiyor.**

| Karşılaştırma | HR@10 fark (derin−klasik) | NDCG@10 fark (derin−klasik) |
|---|---|---|
| GRU4Rec vs Item-KNN | +0.337 | +0.286 |
| GRU4Rec vs S-POP | +0.139 | +0.023 |
| LSTM vs Item-KNN | +0.317 | +0.278 |
| LSTM vs S-POP | +0.119 | +0.016 |
| SASRec vs Item-KNN | +0.340 | +0.279 |
| SASRec vs S-POP | +0.142 | +0.017 |

**Doğrulama adımları (üçü de geçti):**
1. zero_method çapraz kontrolü (wilcox/pratt/zsplit) — üç yöntem de tutarlı, aşırı küçük p-değerleri artefakt değil.
2. Reproducibility kontrolü — yeniden hesaplanan metrikler `runs.jsonl`'deki kayıtlı değerlerle ~1e-4 mertebesinde eşleşiyor.
3. Duyarlılık analizi — seed seçimi sonucu değiştirmiyor.

**Tez tartışması için not:** NDCG@10'daki fark büyüklükleri (0.016–0.29), HR@10'a göre
(0.12–0.34) belirgin şekilde küçük — istatistiksel anlamlılık ile pratik etki büyüklüğü
farklı şeylerdir, n=14160 gibi büyük örneklemde küçük farklar da anlamlı çıkabilir. Bu
ayrım tartışma bölümünde vurgulanacak.

**Kayıtlı dosyalar:** `results/h1_hypothesis_report.md`, `results/h1_session_diffs.parquet`,
`results/h1_per_session_{item_knn,s_pop,gru4rec,lstm,sasrec}.parquet`

---

## 7. Altyapı ve ortam `[KİLİTLİ]`

### Eğitim

| Kaynak | Notlar |
|---|---|
| **Colab Pro** | Ana eğitim ortamı (A100 GPU) |

Colab Pro kısıtları için zorunlu tedbirler:

1. **Checkpoint zorunlu.** Her epoch sonunda Google Drive'a kaydet. Oturum kopması varsayılan senaryo kabul edilir, istisna değil.
2. **Veri Drive'da tutulur.** Ön işlenmiş veri `.parquet` olarak Drive'a yazılır; her oturumda sıfırdan işlenmez.
3. **Eğitim betikleri notebook'tan bağımsız çalışabilmeli.** Notebook yalnızca ince bir sarmalayıcıdır; tüm mantık `src/` altında `.py` dosyalarında durur.
4. **Hiperparametre bütçesi:** Optuna, model başına **20 deneme**. Aşılmaz.
5. **Deney kaydı:** her koşu için config + metrik + süre, tek bir `results/runs.jsonl` dosyasına eklenir.
6. **Optuna paralelliği:** `optuna.n_jobs` config'te ayarlanabilir; Colab'da GPU belleği yeterliyse (A100, 40GB) 4'e çıkarılır — GRU4Rec/LSTM/SASRec'te bu şekilde ~3-4x hızlanma sağlandı.
7. **Mixed precision (AMP):** CUDA'da otomatik aktif, CPU/MPS'te no-op — tüm derin modellerde uygulandı.
8. **Checkpoint klasörleri model başına ayrı tutulur** (`thesis-checkpoints/{model_adı}`) — aksi halde farklı hiperparametrelerle eğitilmiş eski bir checkpoint'in yanlışlıkla yüklenmesi boyut uyuşmazlığı hatasına yol açar (GRU4Rec'te bir kez yaşandı, düzeltildi).

### Cihaz üstü test

| Kaynak | Kullanım |
|---|---|
| Birden fazla fiziksel Android cihaz | **Gecikme ve bellek ölçümünün tek geçerli kaynağı** |
| Android Studio emülatörleri | Yalnızca fonksiyonel doğrulama — çıktı eşleşiyor mu, model yükleniyor mu |

> **Protokol kuralı:** Emülatörden alınan gecikme ölçümü teze **girmez**. Emülatör ana
> makinenin x86 CPU'sunda çalışır; ölçülen süre gerçek ARM cihaz performansını temsil
> etmez. Bunu tezin sınırlılıklar bölümünde de belirt.

Cihazlar tezde **anonim segment etiketleriyle** raporlanır (Cihaz A / B / C) ve her biri
için SoC, RAM, Android sürümü tablolaştırılır. En az üç segment hedeflenir: üst, orta, giriş.

---

## 8. On-device protokolü (K3) `[AÇIK — Hafta 4'te yapılacak]`

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

| Hafta | Tarih | Çıktı | Durum |
|---|---|---|---|
| **1** | 3–9 Ağu | Veri indirme, EDA, ön işleme pipeline, zaman-temelli split, değerlendirme harness'ı, Katman 0–1 | ✅ Tamamlandı |
| **2** | 10–16 Ağu | Özellik mühendisliği, RF + XGBoost, H2 ablasyonu | ✅ Tamamlandı, H2 doğrulandı |
| **3** | 17–30 Ağu | GRU4Rec, LSTM, SASRec + Optuna, H1 testi | ✅ Tamamlandı, H1 doğrulandı (planlanandan biraz uzun sürdü — Colab GPU optimizasyonu ve altyapı sorunları nedeniyle) |
| **4** | 31 Ağu – 1 Eyl | LSApp replikasyonu, on-device dönüşüm ve ölçüm, final tablo/grafikler | 🔜 **Sırada** |

### Geride kalınırsa feda sırası

1. LSApp replikasyonu
2. SASRec'in on-device dönüşümü (GRU4Rec/LSTM zaten öncelikli)
3. Random Forest (zaten feda edildi, XGBoost yeterli görüldü)

**On-device bölümü feda edilmez.**

---

## 10. Kapsam dondurma kuralları `[KİLİTLİ]`

1. **Yeni model eklenmez.**
2. **Yeni veri seti eklenmez.**
3. **Yeni metrik eklenmez.** (Mevcutlar zaten fazlasıyla yeterli.)
4. "Şunu da bir denesek" fikirleri `docs/99-sonraki-calismalar.md` dosyasına yazılır ve tezin *Gelecek Çalışmalar* bölümüne gider — koda girmez.
5. Bir deney başarısız olursa **silinmez, raporlanır.** Negatif sonuç da bulgudur.

---

## 11. Değerlendirme protokolü `[KİLİTLİ — Adım 2]`

| # | Konu | Karar |
|---|---|---|
| A1 | Oturum tanımı | İki olay arası **30 dakika** boşluk = yeni oturum |
| A2 | Filtreleme eşikleri | Min. oturum uzunluğu: **2 olay**. Min. öğe frekansı: **5** (toplamda) |
| A3 | Split | Zaman-temelli: son 1 hafta → test, önceki 1 hafta → validation, kalan → train. **Rastgele split kullanılmaz.** |
| A4 | Cold-start | Test setinden çıkarılmaz; ayrı bir "cold-start alt-küme performansı" tablosunda raporlanır |

## 12a. Öznitelik listesi ve hiperparametre uzayları `[KİLİTLİ — Adım 3]`

### A5 — Öznitelik listesi (H2 ablasyonu)

| Grup | Öznitelikler |
|---|---|
| Popülerlik | Öğenin global görülme sayısı, son 7 gündeki görülme sayısı |
| Birlikte-oluşum | Item-KNN aday skoru, oturumdaki son öğeyle ortak oturum sayısı |
| Zamansal | Oturumdaki geçen süre, günün saati (sin/cos), haftanın günü (sin/cos) |
| Oturum içi | Oturum uzunluğu (şu ana kadar), aynı öğenin tekrar görülme sayısı, oturumdaki eşsiz öğe sayısı |

Not: saat/gün özellikleri döngüsel olduğu için sin/cos çift olarak kodlandı (10 özellik → 12 özellik). Bu, Hafta 2'de Claude Code ile birlikte alınan bir mühendislik kararıdır.

Ablasyon: XGBoost-ham (yalnız Item-KNN skoru) vs. XGBoost-zengin (12 öznitelik).

### A6 — Hiperparametre uzayları (Optuna, model başına 20 deneme)

| Model | Aranan parametreler |
|---|---|
| Item-KNN | k (10–500, log) |
| XGBoost / RF | max_depth (3–10), n_estimators (100–500), learning_rate (0.01–0.3, sadece XGB) |
| GRU4Rec / LSTM | gizli boyut (64–256), katman (1–2), dropout (0.1–0.5), öğrenme oranı (1e-4–1e-2, log) |
| SASRec | dikkat başlığı (1–4), blok (1–3), gizli boyut (64–256), dropout (0.1–0.5) |

Sabit: batch size 256, max sequence length 50, early stopping (val NDCG@10, patience=5), max_epochs=50.

## 12c. On-device ölçüm harness'ı `[KİLİTLİ — Adım 4]`

| Parça | Karar |
|---|---|
| Dönüşüm | GRU4Rec ve LSTM → LiteRT (.tflite): FP32 / dinamik INT8 / tam INT8 |
| Ölçüm uygulaması | Basit Android test app: modeli yükle, sabit örnek oturum, 220 çalıştırma (ilk 20 ısınma atılır) |
| Kayıt | Her koşu → cihaz, varyant, p50/p95 gecikme, tepe bellek, model boyutu → tek CSV |
| Cihazlar | Fiziksel cihazlar (üst/orta/giriş) tek tek; emülatör yalnız fonksiyonel kontrol |
| SASRec | Dönüşümü zor, opsiyonel — feda sırasında 2. |

## 12e. Repo mimarisi ve deney izleme `[KİLİTLİ — Adım 5]`

```
thesis-project/
├── docs/00-karar-notu.md
├── data/{raw,processed}/     # raw .gitignore'da
├── src/{data,features,models,eval,ondevice}/
├── configs/*.yaml
├── notebooks/colab_runner.ipynb   # ince sarmalayıcı, mantık içermez
├── results/*.parquet, runs.jsonl, *.md   # analiz çıktıları — .gitignore istisnası ile takip edilir
├── requirements.txt
└── CLAUDE.md
```

Kural: her deney `runs.jsonl`'e satır ekler; config YAML'da, koda hiperparametre gömülmez.
`.gitignore`'da genel `*.parquet` engeli var ama `results/*.parquet` için istisna tanımlı
(analiz ara çıktıları versiyon kontrolünde tutulur, ham veri tutulmaz).

## 12f. `CLAUDE.md` ve prompt kütüphanesi `[KİLİTLİ — Adım 6]`

`CLAUDE.md` ve `prompt-kutuphanesi.md` dosyaları oluşturuldu. İçerik: kesin kurallar,
durup-sor eşikleri, kod stili, haftalık hazır prompt seti.

---

## 13. Colab operasyonel dersler `[Hafta 3'te öğrenildi]`

- Colab her kapanıp açıldığında çalışma zamanı tamamen sıfırlanır: repo, kütüphaneler,
  config değişiklikleri (checkpoint_dir gibi), veri dosyaları — hepsi yeniden kurulmalı.
- `results/runs.jsonl`'i Colab'dan indirip yerel dosyaya `>>` ile eklerken **tekilleştirme
  şart** — aksi halde aynı satırlar birden fazla kez eklenip istatistiksel testleri bozar
  (bir kez yaşandı, tespit edilip düzeltildi).
- GPU'nun gerçekten kullanılıp kullanılmadığını doğrulamak için Colab'ın sağ üstündeki
  "Resources" panelinden GPU RAM grafiğine bakmak, hücre çalıştırmaya çalışmaktan (kernel
  meşgulken "already running" hatası verir) daha güvenilir.
- Optuna `n_jobs>1` ile paralel çalıştırmak, GPU'nun büyük kısmı boşta kalıyorsa (düşük
  GPU RAM kullanımı sinyal) belirgin hızlanma sağlar; ama arama fazında `torch.manual_seed`
  global RNG state'i thread'ler arası yarışabilir — bu yalnızca arama fazının kendi içindeki
  gürültüyü etkiler, final 5-seed eğitimi sıralı çalıştığı için raporlanan sonuçları etkilemez.

---

## 14. Etik ve uyum notları

- Kullanılan iki veri seti de **kamuya açık ve anonimleştirilmiş**tir; birincil veri toplama yapılmayacaktır.
- LSApp verisi katılımcı onayıyla toplanmıştır; kullanım koşulları kaynak repoda belirtilmiştir.
- İşveren verisi (BTPRO / müşteri uygulamaları) **hiçbir biçimde kullanılmayacaktır** — KVKK ve iş sözleşmesi açısından risklidir.
- Tez metnindeki tüm karakterler standart Unicode noktalama olmalıdır. Öneri formunda görülen `․` (U+2024) ve `‚` (U+201A) karakter ikameleri temizlenmelidir; bu ikameler benzerlik yazılımlarınca **kasıtlı manipülasyon** olarak değerlendirilebilir.
- Kaynakçadaki her künye, tez yazımına başlamadan önce DOI üzerinden tek tek doğrulanacaktır.

---

## 15. Çekirdek kaynakça (metodolojik dayanak)

| Kaynak | Neden gerekli |
|---|---|
| Hidasi, B. ve ark. (2016). *Session-based Recommendations with Recurrent Neural Networks.* ICLR. | GRU4Rec'in orijinal kaynağı |
| Kang, W.-C., & McAuley, J. (2018). *Self-Attentive Sequential Recommendation.* ICDM. | SASRec'in orijinal kaynağı |
| Ferrari Dacrema, M., Cremonesi, P., & Jannach, D. (2019). *Are We Really Making Much Progress?* RecSys. | H1'in reddedilebilirliğinin literatür dayanağı |
| Rendle, S. ve ark. (2019). *On the Difficulty of Evaluating Baselines.* arXiv. | Baseline ayarlamanın kritikliği |
| Krichene, W., & Rendle, S. (2020). *On Sampled Metrics for Item Recommendation.* KDD. | Negatif örnekleme yasağının dayanağı |
| Ludewig, M., & Jannach, D. (2018). *Evaluation of Session-based Recommendation Algorithms.* UMUAI. | Item-KNN ve SR baseline'larının kaynağı |
| Aliannejadi, M. ve ark. (2021). ACM TOIS, 39(3). | LSApp veri seti |
| arXiv:2303.04744 | On-device sonraki-uygulama tahmini, K3 karşılaştırması |

> Bu listedeki her künye Adım 1 kapanışında DOI ile doğrulanacaktır.
