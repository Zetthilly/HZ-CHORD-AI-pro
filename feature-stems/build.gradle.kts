plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "com.zetthilly.ichi.stems"
    compileSdk = 34
    defaultConfig { minSdk = 24; targetSdk = 34 }
    buildFeatures { compose = true }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
}

dependencies {
    implementation(project(":core-audio"))
    implementation(project(":core-ui"))
    implementation("androidx.core:core-ktx:1.13.1")
    implementation(platform("androidx.compose:compose-bom:2024.06.00"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.4")

    // Real on-device inference runtime for the trained separation model.
    // All computation happens on-device via this library — no network calls,
    // no server round-trip. tensorflow-lite-gpu is optional (uncomment once
    // you've confirmed your converted model's ops are GPU-delegate compatible;
    // not every op has a GPU kernel, so test before relying on it).
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
    // implementation("org.tensorflow:tensorflow-lite-gpu:2.16.1")
}
