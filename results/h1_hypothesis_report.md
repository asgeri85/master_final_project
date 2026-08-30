# H1 Hipotez Testi: Katman 3-4 (Derin) vs Katman 1-2 (Klasik)

Oturum bazlı eşleştirilmiş Wilcoxon signed-rank testi (zero_method="wilcox"), 6 çift x 2 metrik = 12 test, Holm-Bonferroni düzeltmesi (RetailRocket val, seed=42).

## Ana sonuç tablosu

| Çift | Metrik | n_ortak_oturum | p-değeri | Holm-düzeltilmiş p | Anlamlı mı (α=0.05) | Fark yönü | Fark büyüklüğü (deep - classical) |
|---|---|---|---|---|---|---|---|
| gru4rec vs item_knn | hr@10 | 14160 | 0 | 0 | Evet | derin > klasik | 0.337287 |
| gru4rec vs item_knn | ndcg@10 | 14160 | 0 | 0 | Evet | derin > klasik | 0.285924 |
| gru4rec vs s_pop | hr@10 | 14160 | 2.9786e-267 | 1.4893e-266 | Evet | derin > klasik | 0.138719 |
| gru4rec vs s_pop | ndcg@10 | 14160 | 5.96607e-44 | 1.78982e-43 | Evet | derin > klasik | 0.0231709 |
| lstm vs item_knn | hr@10 | 14160 | 0 | 0 | Evet | derin > klasik | 0.317257 |
| lstm vs item_knn | ndcg@10 | 14160 | 0 | 0 | Evet | derin > klasik | 0.278304 |
| lstm vs s_pop | hr@10 | 14160 | 3.00548e-201 | 1.20219e-200 | Evet | derin > klasik | 0.118689 |
| lstm vs s_pop | ndcg@10 | 14160 | 7.62767e-33 | 1.52553e-32 | Evet | derin > klasik | 0.0155509 |
| sasrec vs item_knn | hr@10 | 14160 | 0 | 0 | Evet | derin > klasik | 0.340271 |
| sasrec vs item_knn | ndcg@10 | 14160 | 0 | 0 | Evet | derin > klasik | 0.279301 |
| sasrec vs s_pop | hr@10 | 14160 | 8.634e-274 | 5.1804e-273 | Evet | derin > klasik | 0.141703 |
| sasrec vs s_pop | ndcg@10 | 14160 | 1.51107e-23 | 1.51107e-23 | Evet | derin > klasik | 0.0165478 |

## Sıfır-fark oranı ve asimetri

| Çift | Metrik | n_ortak_oturum | n_sıfır | n_pozitif (deep>classical) | n_negatif (deep<classical) |
|---|---|---|---|---|---|
| gru4rec vs item_knn | hr@10 | 14160 | 6912 | 6551 | 697 |
| gru4rec vs item_knn | ndcg@10 | 14160 | 4737 | 7493 | 1930 |
| gru4rec vs s_pop | hr@10 | 14160 | 8538 | 4186 | 1436 |
| gru4rec vs s_pop | ndcg@10 | 14160 | 7275 | 4221 | 2664 |
| lstm vs item_knn | hr@10 | 14160 | 7008 | 6342 | 810 |
| lstm vs item_knn | ndcg@10 | 14160 | 4862 | 7255 | 2043 |
| lstm vs s_pop | hr@10 | 14160 | 8677 | 3922 | 1561 |
| lstm vs s_pop | ndcg@10 | 14160 | 7543 | 4043 | 2574 |
| sasrec vs item_knn | hr@10 | 14160 | 6880 | 6606 | 674 |
| sasrec vs item_knn | ndcg@10 | 14160 | 4655 | 7605 | 1900 |
| sasrec vs s_pop | hr@10 | 14160 | 8388 | 4303 | 1469 |
| sasrec vs s_pop | ndcg@10 | 14160 | 6902 | 4249 | 3009 |

## zero_method çapraz kontrolü (wilcox / pratt / zsplit)

| Çift | Metrik | wilcox p | pratt p | zsplit p | Tutarlı mı |
|---|---|---|---|---|---|
| gru4rec vs item_knn | hr@10 | 0 | 0 | 0 | Evet |
| gru4rec vs item_knn | ndcg@10 | 0 | 0 | 0 | Evet |
| gru4rec vs s_pop | hr@10 | 2.9786e-267 | 1.94535e-301 | 2.79417e-249 | Evet |
| gru4rec vs s_pop | ndcg@10 | 5.96607e-44 | 2.78484e-69 | 3.52025e-62 | Evet |
| lstm vs item_knn | hr@10 | 0 | 0 | 0 | Evet |
| lstm vs item_knn | ndcg@10 | 0 | 0 | 0 | Evet |
| lstm vs s_pop | hr@10 | 3.00548e-201 | 6.20975e-228 | 1.6582e-186 | Evet |
| lstm vs s_pop | ndcg@10 | 7.62767e-33 | 3.64881e-61 | 3.60387e-54 | Evet |
| sasrec vs item_knn | hr@10 | 0 | 0 | 0 | Evet |
| sasrec vs item_knn | ndcg@10 | 0 | 0 | 0 | Evet |
| sasrec vs s_pop | hr@10 | 8.634e-274 | 0 | 3.69768e-260 | Evet |
| sasrec vs s_pop | ndcg@10 | 1.51107e-23 | 4.05504e-40 | 1.10437e-36 | Evet |

## Yeniden üretim (reproducibility) kontrolü

Prefix-ağırlıklı ortalama (orijinal `run_*.py` scriptlerinin `average_at_k` ile hesapladığı ve `results/runs.jsonl`'e kaydettiği yöntemin birebir aynısı) ile kayıtlı değer arasındaki mutlak fark; tolerans=0.01. (Bu, Wilcoxon testinde kullanılan session-gruplu ortalamadan farklıdır -- session uzunluğu metrikle ilişkiliyse o ikisi arasında beklenen, hataya işaret etmeyen bir fark oluşur; ayrı olarak izlenmez.)

| Model | Metrik | Prefix-ağırlıklı ortalama (yeniden fit) | Kayıtlı (runs.jsonl) | Mutlak fark |
|---|---|---|---|---|
| item_knn | hr@10 | 0.207097 | 0.207186 | 8.96781e-05 |
| item_knn | ndcg@10 | 0.119017 | 0.119057 | 4.0116e-05 |
| s_pop | hr@10 | 0.379816 | 0.379787 | 2.98927e-05 |
| s_pop | ndcg@10 | 0.332762 | 0.332752 | 9.89995e-06 |
| gru4rec | hr@10 | 0.468239 | 0.468239 | 0 |
| gru4rec | ndcg@10 | 0.334646 | 0.334646 | 4.996e-15 |
| lstm | hr@10 | 0.451768 | 0.451768 | 0 |
| lstm | ndcg@10 | 0.330475 | 0.330475 | 6.66134e-15 |
| sasrec | hr@10 | 0.472125 | 0.471976 | 0.000149463 |
| sasrec | ndcg@10 | 0.330221 | 0.330327 | 0.000106206 |

## Duyarlılık analizi: 5-seed ortalaması vs seed=42 (H1 testi tekrarlanmadı)

| Model | Metrik | seed=42 değeri | 5-seed ortalama±std | Mutlak fark | Oransal fark (%) |
|---|---|---|---|---|---|
| gru4rec | hr@10 | 0.468239 | 0.467898±0.00189443 | 0.000340777 | 0.0728314 |
| gru4rec | ndcg@10 | 0.334646 | 0.334875±0.000843238 | 0.000228742 | 0.0683067 |
| lstm | hr@10 | 0.451768 | 0.449341±0.0018898 | 0.00242729 | 0.540188 |
| lstm | ndcg@10 | 0.330475 | 0.328235±0.00140281 | 0.0022392 | 0.682192 |
| sasrec | hr@10 | 0.471976 | 0.470511±0.00142923 | 0.00146474 | 0.311309 |
| sasrec | ndcg@10 | 0.330327 | 0.329371±0.000897389 | 0.000956422 | 0.290379 |
