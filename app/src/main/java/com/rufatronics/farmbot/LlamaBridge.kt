package com.rufatronics.farmbot

/**
 * Thin Kotlin wrapper around the native llama.cpp JNI bridge.
 * All heavy lifting happens in farmbot_jni.cpp / llama.cpp itself.
 */
class LlamaBridge {

    companion object {
        init {
            System.loadLibrary("farmbot_jni")
        }
    }

    external fun loadModel(modelPath: String, nThreads: Int, nCtx: Int): Boolean

    external fun generate(
        prompt: String,
        maxNewTokens: Int,
        temperature: Float,
        topK: Int,
        repeatPenalty: Float
    ): String

    external fun unloadModel()
}
