# On-device Dönüşüm Raporu (Aşama 1): GRU4Rec/LSTM/SASRec → LiteRT

RetailRocket val seti üzerinde, PyTorch FP32 baseline'a karşı FP32/dinamik INT8/tam INT8 TFLite varyantlarının dosya boyutu ve NDCG@10 kaybı (docs/00-karar-notu.md §8/§12c).

| Model | Varyant | Dosya boyutu (MB) | NDCG@10 (orijinal) | NDCG@10 (kuantize) | Fark (mutlak) | Fark (%) | Not |
|---|---|---|---|---|---|---|---|
| gru4rec | fp32 | 28.135 | 0.334613 | 0.334613 | 0.000000 | 0.00 | |
| gru4rec | dynamic_int8 | 7.826 | 0.334613 | 0.334606 | 0.000006 | 0.00 | |
| gru4rec | full_int8 | - | 0.334613 | - | - | - | dönüşüm başarısız: RuntimeError: tensorflow/lite/kernels/reshape.cc:94 num_input_elements != num_output_elements (0 != 78)Node number 1003 (RESHAPE) failed to prepare. |
| lstm | fp32 | 66.966 | 0.330444 | 0.330444 | 0.000000 | 0.00 | |
| lstm | dynamic_int8 | 17.535 | 0.330444 | 0.330254 | 0.000190 | 0.06 | |
| lstm | full_int8 | - | 0.330444 | - | - | - | dönüşüm başarısız: RuntimeError: tensorflow/lite/kernels/reshape.cc:94 num_input_elements != num_output_elements (0 != 185)Node number 121 (RESHAPE) failed to prepare. |
| sasrec | - | - | - | - | - | - | checkpoint bulunamadı: checkpoints/sasrec/seed_42/sasrec_seed42_best.pt |
