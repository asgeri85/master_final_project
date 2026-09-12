Bu klasöre kurulum sırasında şu 5 dosya kopyalanmalı (git'e eklenmez, bkz. .gitignore):

  gru4rec_fp32.tflite
  gru4rec_dynamic_int8.tflite
  lstm_fp32.tflite
  lstm_dynamic_int8.tflite
  benchmark_session.json

İlk 4'ü results/ondevice/ altında zaten var (Aşama 1 çıktısı).
benchmark_session.json için önce şunu çalıştır:

  python -m src.ondevice.prepare_benchmark_session

Sonra hepsini kopyala:

  cp results/ondevice/gru4rec_fp32.tflite results/ondevice/gru4rec_dynamic_int8.tflite \
     results/ondevice/lstm_fp32.tflite results/ondevice/lstm_dynamic_int8.tflite \
     results/ondevice/benchmark_session.json \
     android/app/src/main/assets/
