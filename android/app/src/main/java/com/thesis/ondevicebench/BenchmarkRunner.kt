package com.thesis.ondevicebench

import android.content.Context
import android.os.Build
import android.os.Debug
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.ceil

/** docs/00-karar-notu.md §12c: 4 sabit model/varyant kombinasyonu -- fp32 ve
 * dynamic_int8 (full_int8 her iki modelde de Aşama 1'de dönüşemedi, SASRec
 * checkpoint'i yok, ikisi de kapsam dışı). */
private val MODEL_SPECS = listOf(
    ModelSpec("gru4rec", "fp32", "gru4rec_fp32.tflite"),
    ModelSpec("gru4rec", "dynamic_int8", "gru4rec_dynamic_int8.tflite"),
    ModelSpec("lstm", "fp32", "lstm_fp32.tflite"),
    ModelSpec("lstm", "dynamic_int8", "lstm_dynamic_int8.tflite"),
)

private const val WARMUP_RUNS = 20
private const val MEASURED_RUNS = 200
private const val SESSION_ASSET = "benchmark_session.json"

data class ModelSpec(val model: String, val variant: String, val assetFileName: String)

data class BenchmarkResult(
    val model: String,
    val variant: String,
    val p50Ms: Double,
    val p95Ms: Double,
    val peakMemoryMb: Double,
    val modelSizeMb: Double,
    val deviceName: String,
    val androidVersion: String,
)

/** Aynı sabit oturumu (assets/benchmark_session.json, src/ondevice/
 * prepare_benchmark_session.py ile üretilmiş) her modelde kullanır -- adil
 * karşılaştırma için (karar notu §12c). */
private class BenchmarkSession(json: JSONObject) {
    val itemIds: LongArray
    val lastIndex: Long

    init {
        val padded = json.getJSONArray("item_indices_padded")
        itemIds = LongArray(padded.length()) { padded.getLong(it) }
        lastIndex = json.getLong("last_index")
    }
}

class BenchmarkRunner(private val context: Context) {

    fun runAll(): List<BenchmarkResult> {
        val session = loadSession()
        return MODEL_SPECS.map { spec -> runOne(spec, session) }
    }

    private fun loadSession(): BenchmarkSession {
        val text = context.assets.open(SESSION_ASSET).bufferedReader().use { it.readText() }
        return BenchmarkSession(JSONObject(text))
    }

    private fun runOne(spec: ModelSpec, session: BenchmarkSession): BenchmarkResult {
        val modelSizeMb = context.assets.openFd(spec.assetFileName).use { it.declaredLength / (1024.0 * 1024.0) }
        val interpreter = loadInterpreter(spec.assetFileName)
        try {
            val (itemIdsIndex, lastIndexIndex) = resolveInputIndices(interpreter)
            val itemIdsInput = arrayOf(session.itemIds)
            val lastIndexInput = longArrayOf(session.lastIndex)
            val output = Array(1) { FloatArray(interpreter.getOutputTensor(0).shape()[1]) }

            val inputs = arrayOfNulls<Any>(interpreter.inputTensorCount)
            inputs[itemIdsIndex] = itemIdsInput
            inputs[lastIndexIndex] = lastIndexInput
            val outputs = mutableMapOf<Int, Any>(0 to output)

            repeat(WARMUP_RUNS) {
                interpreter.runForMultipleInputsOutputs(inputs.requireNoNulls(), outputs)
            }

            val durationsMs = DoubleArray(MEASURED_RUNS)
            var peakHeapBytes = 0L
            repeat(MEASURED_RUNS) { i ->
                val start = System.nanoTime()
                interpreter.runForMultipleInputsOutputs(inputs.requireNoNulls(), outputs)
                val end = System.nanoTime()
                durationsMs[i] = (end - start) / 1_000_000.0
                peakHeapBytes = maxOf(peakHeapBytes, Debug.getNativeHeapAllocatedSize())
            }

            durationsMs.sort()
            return BenchmarkResult(
                model = spec.model,
                variant = spec.variant,
                p50Ms = percentile(durationsMs, 50.0),
                p95Ms = percentile(durationsMs, 95.0),
                peakMemoryMb = peakHeapBytes / (1024.0 * 1024.0),
                modelSizeMb = modelSizeMb,
                deviceName = Build.MODEL,
                androidVersion = Build.VERSION.RELEASE,
            )
        } finally {
            interpreter.close()
        }
    }

    private fun loadInterpreter(assetFileName: String): Interpreter {
        val fd = context.assets.openFd(assetFileName)
        val buffer: MappedByteBuffer = FileInputStream(fd.fileDescriptor).use { input ->
            input.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        }
        // Tek thread: cihazlar arası zamanlama gürültüsünü azaltmak için (karar notu §7: adil karşılaştırma).
        // XNNPACK varsayılan (açık) bırakılır -- gerçek bir üretim uygulamasının alacağı
        // varsayılan CPU performansını temsil eder. Not: bazı emülatör imajlarında (ör. Apple
        // Silicon üzerinde Pixel-profilli AVD) XNNPACK'in bazı ARM SIMD kodyolları SIGILL ile
        // çöküyor -- bu sadece emülatöre özgü bir kusur olarak gözlemlendi (fiziksel cihazlarda
        // beklenmiyor); zaten karar notu §7 emülatör ölçümünü teze dahil etmiyor.
        val options = Interpreter.Options().setNumThreads(1)
        return Interpreter(buffer, options)
    }

    /** onnx2tf ONNX giriş adlarını ("item_ids"/"last_index") korur, ama giriş
     * sırası garanti değildir -- ada göre eşle (bkz. src/ondevice/tflite_model.py
     * ile aynı yaklaşım). */
    private fun resolveInputIndices(interpreter: Interpreter): Pair<Int, Int> {
        var itemIdsIndex = -1
        var lastIndexIndex = -1
        for (i in 0 until interpreter.inputTensorCount) {
            val name = interpreter.getInputTensor(i).name()
            if (name.contains("item_ids")) itemIdsIndex = i
            if (name.contains("last_index")) lastIndexIndex = i
        }
        require(itemIdsIndex >= 0 && lastIndexIndex >= 0) {
            "beklenen giriş tensörleri (item_ids/last_index) bulunamadı"
        }
        return itemIdsIndex to lastIndexIndex
    }

    private fun percentile(sortedMs: DoubleArray, p: Double): Double {
        val rank = ceil(p / 100.0 * sortedMs.size).toInt().coerceIn(1, sortedMs.size)
        return sortedMs[rank - 1]
    }
}
