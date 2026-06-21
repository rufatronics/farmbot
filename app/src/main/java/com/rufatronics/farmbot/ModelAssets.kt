package com.rufatronics.farmbot

import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileOutputStream

/**
 * Both models (~111MB GGUF language model + 4.56MB TFLite vision model) are
 * BAKED INTO the APK as raw assets -- no network download required, ever.
 * This matters for the target users: African farmers on legacy devices with
 * unreliable or expensive mobile data. The app must work fully offline from
 * the moment it's installed.
 *
 * Native code (llama.cpp via JNI) and TFLite's Interpreter both need a real
 * filesystem path, not a compressed APK asset stream, so on first launch we
 * copy the files out of assets/ into the app's private internal storage once.
 * Subsequent launches just check the files already exist and skip this step.
 */
object ModelAssets {

    private const val TAG = "ModelAssets"

    private const val LLM_ASSET    = "models/farmbot-Q4_K_M.gguf"
    private const val VISION_ASSET = "models/crop_doctor.tflite"

    fun llmFile(context: Context) = File(context.filesDir, "farmbot-Q4_K_M.gguf")
    fun visionFile(context: Context) = File(context.filesDir, "crop_doctor.tflite")

    fun modelsReady(context: Context): Boolean =
        llmFile(context).exists() && visionFile(context).exists()

    /**
     * Copies bundled assets to internal storage if not already done.
     * Call once at app startup, e.g. in a splash/loading screen, since the
     * 111MB copy can take a few seconds on slow storage (legacy devices).
     */
    fun prepareModels(context: Context, onProgress: (String) -> Unit = {}) {
        if (!visionFile(context).exists()) {
            onProgress("Preparing vision model...")
            copyAsset(context, VISION_ASSET, visionFile(context))
        }
        if (!llmFile(context).exists()) {
            onProgress("Preparing language model...")
            copyAsset(context, LLM_ASSET, llmFile(context))
        }
        Log.i(TAG, "Models ready at ${context.filesDir}")
    }

    private fun copyAsset(context: Context, assetPath: String, dest: File) {
        val tmp = File(dest.parentFile, dest.name + ".tmp")
        context.assets.open(assetPath).use { input ->
            FileOutputStream(tmp).use { output ->
                val buffer = ByteArray(1 shl 16) // 64KB chunks
                var bytesRead: Int
                while (input.read(buffer).also { bytesRead = it } != -1) {
                    output.write(buffer, 0, bytesRead)
                }
            }
        }
        tmp.renameTo(dest)
    }
}
