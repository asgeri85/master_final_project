package com.thesis.ondevicebench

import android.app.Activity
import android.os.Build
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.TextView
import java.util.Locale

/** Basit, tek seferlik kullanım için tasarlanmış ölçüm uygulaması (karar notu
 * §12c) -- production kalitesinde olması gerekmiyor, sadece güvenilir ölçüm
 * yapması yeterli. */
class MainActivity : Activity() {

    private lateinit var startButton: Button
    private lateinit var resultText: TextView
    private lateinit var warningText: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        startButton = findViewById(R.id.startButton)
        resultText = findViewById(R.id.resultText)
        warningText = findViewById(R.id.warningText)

        if (isProbablyEmulator()) {
            warningText.visibility = View.VISIBLE
            warningText.text = "Emülatör tespit edildi — bu ölçüm teze dahil edilmez, yalnızca fonksiyonel kontrol içindir (CLAUDE.md kural 6)."
        }

        startButton.setOnClickListener { runBenchmark() }
    }

    private fun runBenchmark() {
        startButton.isEnabled = false
        resultText.text = "Ölçülüyor (4 model × 220 koşu)... bu birkaç dakika sürebilir."

        Thread {
            try {
                val results = BenchmarkRunner(applicationContext).runAll()
                val fileName = try {
                    CsvExporter.exportToDownloads(applicationContext, results)
                } catch (e: Exception) {
                    null
                }
                val summary = formatResults(results) +
                    if (fileName != null) "\n\nCSV kaydedildi: Downloads/$fileName" else "\n\nCSV kaydedilemedi (bkz. logcat)."
                runOnUiThread {
                    resultText.text = summary
                    startButton.isEnabled = true
                }
            } catch (e: Exception) {
                runOnUiThread {
                    resultText.text = "Hata: ${e.message}"
                    startButton.isEnabled = true
                }
            }
        }.start()
    }

    private fun formatResults(results: List<BenchmarkResult>): String {
        // Locale.US: bkz. CsvExporter.buildCsv -- aynı ondalık-ayıraç sorunu ekran
        // metninde de oluşur (cihaz Türkçe locale'de "2,62" gösterir).
        val header = String.format(Locale.US, "%-8s %-13s %8s %8s %9s %9s", "model", "variant", "p50_ms", "p95_ms", "peak_mb", "size_mb")
        val rows = results.joinToString("\n") { r ->
            String.format(
                Locale.US,
                "%-8s %-13s %8.2f %8.2f %9.2f %9.2f",
                r.model, r.variant, r.p50Ms, r.p95Ms, r.peakMemoryMb, r.modelSizeMb,
            )
        }
        return "$header\n$rows"
    }

    /** İyi bilinen emülatör tespit sezgiselleri (Build alanları) -- sadece
     * ekranda uyarı göstermek için, ölçüm mantığını değiştirmez. */
    private fun isProbablyEmulator(): Boolean {
        return Build.FINGERPRINT.startsWith("generic") ||
            Build.FINGERPRINT.startsWith("unknown") ||
            Build.MODEL.contains("google_sdk") ||
            Build.MODEL.contains("Emulator") ||
            Build.MODEL.contains("Android SDK built for") ||
            Build.MANUFACTURER.contains("Genymotion") ||
            (Build.BRAND.startsWith("generic") && Build.DEVICE.startsWith("generic")) ||
            Build.PRODUCT == "google_sdk" ||
            // Modern Pixel-profile AVD'ler (ör. "sdk_gphone64_arm64") yukarıdaki
            // eski sezgisellere uymuyor -- ranchu/goldfish donanım katmanı ve
            // "sdk_gphone" ürün adı her zaman emülatöre işaret eder.
            Build.HARDWARE.contains("ranchu") ||
            Build.HARDWARE.contains("goldfish") ||
            Build.PRODUCT.contains("sdk_gphone")
    }
}
