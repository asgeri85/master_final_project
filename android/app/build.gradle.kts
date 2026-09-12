plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.thesis.ondevicebench"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.thesis.ondevicebench"
        minSdk = 29
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlin {
        compilerOptions {
            jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
        }
    }

    // .tflite dosyaları büyük (8-70MB) -- sıkıştırılmadan pakete konmazsa
    // Interpreter'ın assets'ten doğrudan mmap ile yüklemesi başarısız olabilir.
    androidResources {
        noCompress += "tflite"
    }
}

dependencies {
    // org.tensorflow:tensorflow-lite (en yeni: 2.17.0) FULLY_CONNECTED op v12'yi
    // desteklemiyor -- .tflite dosyaları TF 2.21 ile dönüştürüldü (bkz.
    // requirements.txt). LiteRT (aynı org.tensorflow.lite.* Java API'si,
    // TFLite'ın yeni adı/aktif olarak güncellenen sürümü) bunu destekliyor.
    implementation("com.google.ai.edge.litert:litert:2.2.0")
}
