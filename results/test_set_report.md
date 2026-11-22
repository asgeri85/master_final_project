# Test seti değerlendirmesi (A1) — retailrocket

Yeni eğitim ve hiperparametre araması yok. Derin modeller val üzerinde seçilmiş en iyi checkpoint'lerden yüklendi; klasik modeller ve XGBoost train üzerinde deterministik olarak yeniden fit edildi.

## Checkpoint doğrulaması (val yeniden puanlama vs runs.jsonl)

| Model | Seed | Kaynak | max mutlak fark | Durum |
|---|---|---|---|---|
| gru4rec | 42 | best_on_val | 2.99e-05 | OK |
| gru4rec | 43 | best_on_val | 2.19e-06 | OK |
| gru4rec | 44 | best_on_val | 2.67e-06 | OK |
| gru4rec | 45 | best_on_val | 9.58e-06 | OK |
| gru4rec | 46 | best_on_val | 2.30e-05 | OK |
| lstm | 42 | best_on_val | 9.11e-06 | OK |
| lstm | 43 | recovered_from_lstm_seed43_epoch049.pt | 2.99e-05 | OK |
| lstm | 44 | best_on_val | 2.99e-05 | OK |
| lstm | 45 | best_on_val | 2.99e-05 | OK |
| lstm | 46 | recovered_from_lstm_seed46_epoch050.pt | 2.99e-05 | OK |
| sasrec | 42 | best_on_val | 5.05e-15 | OK |
| sasrec | 43 | best_on_val | 5.27e-15 | OK |
| sasrec | 44 | best_on_val | 6.88e-15 | OK |
| sasrec | 45 | best_on_val | 7.05e-15 | OK |
| sasrec | 46 | best_on_val | 6.99e-15 | OK |

## Test sonuçları

| Model | Tür | Seed | n | hr@10 | ndcg@10 | mrr@10 |
|---|---|---|---|---|---|---|
| pop | - | - | 28449 | 0.0125 | 0.0058 | 0.0038 |
| s_pop | - | - | 28449 | 0.3713 | 0.3211 | 0.3046 |
| item_knn | - | - | 28449 | 0.2095 | 0.1206 | 0.0933 |
| gru4rec | seed | 42 | 28449 | 0.4543 | 0.3220 | 0.2804 |
| gru4rec | seed | 43 | 28449 | 0.4564 | 0.3234 | 0.2817 |
| gru4rec | seed | 44 | 28449 | 0.4570 | 0.3251 | 0.2837 |
| gru4rec | seed | 45 | 28449 | 0.4546 | 0.3222 | 0.2807 |
| gru4rec | seed | 46 | 28449 | 0.4520 | 0.3210 | 0.2799 |
| gru4rec | summary | - | 28449 | 0.4549 ± 0.0020 | 0.3227 ± 0.0016 | 0.2813 ± 0.0015 |
| lstm | seed | 42 | 28449 | 0.4393 | 0.3178 | 0.2796 |
| lstm | seed | 43 | 28449 | 0.4351 | 0.3154 | 0.2778 |
| lstm | seed | 44 | 28449 | 0.4373 | 0.3162 | 0.2782 |
| lstm | seed | 45 | 28449 | 0.4380 | 0.3174 | 0.2795 |
| lstm | seed | 46 | 28449 | 0.4399 | 0.3187 | 0.2806 |
| lstm | summary | - | 28449 | 0.4379 ± 0.0019 | 0.3171 ± 0.0013 | 0.2791 ± 0.0012 |
| sasrec | seed | 42 | 28449 | 0.4531 | 0.3151 | 0.2719 |
| sasrec | seed | 43 | 28449 | 0.4526 | 0.3149 | 0.2718 |
| sasrec | seed | 44 | 28449 | 0.4541 | 0.3148 | 0.2712 |
| sasrec | seed | 45 | 28449 | 0.4554 | 0.3153 | 0.2715 |
| sasrec | seed | 46 | 28449 | 0.4512 | 0.3135 | 0.2704 |
| sasrec | summary | - | 28449 | 0.4533 ± 0.0016 | 0.3147 ± 0.0007 | 0.2713 ± 0.0006 |
| xgboost_raw | - | - | 12122 | 0.2186 | 0.1332 | 0.1067 |
| xgboost_rich | - | - | 12122 | 0.2619 | 0.1714 | 0.1430 |

Not: XGBoost satırlarında n = oturum sayısı (H2 birimi); diğer satırlarda n = önek (tahmin çifti) sayısı.

## H1 test setinde (seed=42, Wilcoxon, Holm, 12 test)

| Çift | Metrik | n | p | Holm p | Anlamlı | Fark (derin−klasik) | sıfır | derin> | klasik> |
|---|---|---|---|---|---|---|---|---|---|
| gru4rec vs item_knn | hr@10 | 12122 | 0 | 0 | Evet | 0.327051 | 5925 | 5538 | 659 |
| gru4rec vs item_knn | ndcg@10 | 12122 | 0 | 0 | Evet | 0.274565 | 4147 | 6248 | 1727 |
| gru4rec vs s_pop | hr@10 | 12122 | 6.58363e-237 | 3.95018e-236 | Evet | 0.139422 | 7421 | 3505 | 1196 |
| gru4rec vs s_pop | ndcg@10 | 12122 | 5.34893e-39 | 1.60468e-38 | Evet | 0.0237407 | 6349 | 3546 | 2227 |
| lstm vs item_knn | hr@10 | 12122 | 0 | 0 | Evet | 0.303092 | 6062 | 5302 | 758 |
| lstm vs item_knn | ndcg@10 | 12122 | 0 | 0 | Evet | 0.265526 | 4288 | 6035 | 1799 |
| lstm vs s_pop | hr@10 | 12122 | 4.43122e-170 | 1.77249e-169 | Evet | 0.115462 | 7565 | 3255 | 1302 |
| lstm vs s_pop | ndcg@10 | 12122 | 8.54345e-27 | 1.70869e-26 | Evet | 0.0147016 | 6608 | 3354 | 2160 |
| sasrec vs item_knn | hr@10 | 12122 | 0 | 0 | Evet | 0.322089 | 5983 | 5478 | 661 |
| sasrec vs item_knn | ndcg@10 | 12122 | 0 | 0 | Evet | 0.265263 | 4218 | 6215 | 1689 |
| sasrec vs s_pop | hr@10 | 12122 | 1.18969e-222 | 5.94847e-222 | Evet | 0.13446 | 7397 | 3474 | 1251 |
| sasrec vs s_pop | ndcg@10 | 12122 | 2.82498e-17 | 2.82498e-17 | Evet | 0.0144387 | 6120 | 3454 | 2548 |

p = 0 görünen değerler float64 alt taşmasıdır; metinde p < 10^-300 yazılır.

## H2 test setinde (zengin vs ham, Wilcoxon, Holm)

| Metrik | n oturum | p | Holm p | Fark (zengin−ham) | sıfır | zengin> | ham> |
|---|---|---|---|---|---|---|---|
| hr@10 | 12122 | 1.32276e-125 | 1.32276e-125 | 0.0432994 | 10434 | 1407 | 281 |
| ndcg@10 | 12122 | 3.11339e-186 | 6.22678e-186 | 0.038213 | 8667 | 2529 | 926 |
