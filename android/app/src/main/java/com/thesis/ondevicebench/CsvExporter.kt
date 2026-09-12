package com.thesis.ondevicebench

import android.content.ContentValues
import android.content.Context
import android.os.Environment
import android.provider.MediaStore
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** CSV kolonları karar notu §12c'ye göre sabit: model, varyant, p50/p95 gecikme,
 * tepe bellek, model boyutu, cihaz, Android sürümü. Her cihaz/çalıştırma kendi
 * dosyasını üretir; farklı cihazlardan toplanan CSV'ler aynı şemayı paylaştığı
 * için sonradan elle birleştirilir (bkz. plan §4c). */
object CsvExporter {

    private const val CSV_HEADER = "model,variant,p50_ms,p95_ms,peak_memory_mb,model_size_mb,device_name,android_version"

    fun exportToDownloads(context: Context, results: List<BenchmarkResult>): String {
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val deviceTag = android.os.Build.MODEL.replace(Regex("[^A-Za-z0-9]+"), "_")
        val fileName = "ondevice_benchmark_${deviceTag}_$timestamp.csv"

        val resolver = context.contentResolver
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, fileName)
            put(MediaStore.MediaColumns.MIME_TYPE, "text/csv")
            put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS)
        }
        val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
            ?: throw IllegalStateException("MediaStore.Downloads insert başarısız")
        resolver.openOutputStream(uri)?.use { out ->
            out.write(buildCsv(results).toByteArray())
        } ?: throw IllegalStateException("CSV için OutputStream açılamadı")
        return fileName
    }

    private fun buildCsv(results: List<BenchmarkResult>): String {
        val rows = results.joinToString("\n") { r ->
            listOf(
                r.model,
                r.variant,
                "%.3f".format(r.p50Ms),
                "%.3f".format(r.p95Ms),
                "%.3f".format(r.peakMemoryMb),
                "%.3f".format(r.modelSizeMb),
                r.deviceName,
                r.androidVersion,
            ).joinToString(",")
        }
        return "$CSV_HEADER\n$rows\n"
    }
}
