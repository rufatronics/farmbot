// farmbot_jni.cpp
// Minimal JNI bridge between Kotlin and llama.cpp for on-device FarmBot inference.
// Loads a GGUF model from app-private storage and runs text generation.

#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <sstream>

#include "llama.h"

#define LOG_TAG "FarmBotJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {
    llama_model*   g_model   = nullptr;
    llama_context* g_ctx     = nullptr;
    const llama_vocab* g_vocab = nullptr;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_rufatronics_farmbot_LlamaBridge_loadModel(
        JNIEnv* env, jobject /* this */, jstring modelPath, jint nThreads, jint nCtx) {

    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    LOGI("Loading model from: %s", path);

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;   // CPU-only — required for legacy device support

    g_model = llama_model_load_from_file(path, model_params);
    env->ReleaseStringUTFChars(modelPath, path);

    if (g_model == nullptr) {
        LOGE("Failed to load model");
        return JNI_FALSE;
    }

    g_vocab = llama_model_get_vocab(g_model);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx     = nCtx > 0 ? nCtx : 512;
    ctx_params.n_threads = nThreads > 0 ? nThreads : 4;
    ctx_params.n_threads_batch = ctx_params.n_threads;

    g_ctx = llama_init_from_model(g_model, ctx_params);
    if (g_ctx == nullptr) {
        LOGE("Failed to create context");
        llama_model_free(g_model);
        g_model = nullptr;
        return JNI_FALSE;
    }

    LOGI("Model loaded successfully. Context size: %d", ctx_params.n_ctx);
    return JNI_TRUE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_rufatronics_farmbot_LlamaBridge_generate(
        JNIEnv* env, jobject /* this */, jstring prompt,
        jint maxNewTokens, jfloat temperature, jint topK, jfloat repeatPenalty) {

    if (g_model == nullptr || g_ctx == nullptr) {
        LOGE("Model not loaded");
        return env->NewStringUTF("[Error: model not loaded]");
    }

    const char* prompt_cstr = env->GetStringUTFChars(prompt, nullptr);
    std::string prompt_str(prompt_cstr);
    env->ReleaseStringUTFChars(prompt, prompt_cstr);

    // Tokenize prompt
    const int n_prompt_tokens = -llama_tokenize(
        g_vocab, prompt_str.c_str(), (int32_t)prompt_str.size(),
        nullptr, 0, true, true);

    std::vector<llama_token> tokens(n_prompt_tokens);
    if (llama_tokenize(g_vocab, prompt_str.c_str(), (int32_t)prompt_str.size(),
                        tokens.data(), (int32_t)tokens.size(), true, true) < 0) {
        LOGE("Tokenization failed");
        return env->NewStringUTF("[Error: tokenization failed]");
    }

    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());

    if (llama_decode(g_ctx, batch) != 0) {
        LOGE("Initial decode failed");
        return env->NewStringUTF("[Error: decode failed]");
    }

    // Sampler chain — mirrors the recommended inference settings from the model card:
    // temperature=0.3, top_k=20, repeat_penalty=1.2
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler* sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(topK));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(64, repeatPenalty, 0.0f, 0.0f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    std::ostringstream result;
    const llama_token eos_token = llama_vocab_eos(g_vocab);

    int n_generated = 0;
    llama_token new_token;

    while (n_generated < maxNewTokens) {
        new_token = llama_sampler_sample(sampler, g_ctx, -1);

        if (new_token == eos_token || llama_vocab_is_eog(g_vocab, new_token)) {
            break;
        }

        char piece[256];
        int n_chars = llama_token_to_piece(g_vocab, new_token, piece, sizeof(piece), 0, true);
        if (n_chars > 0) {
            result.write(piece, n_chars);
        }

        llama_batch next_batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(g_ctx, next_batch) != 0) {
            LOGE("Decode failed mid-generation");
            break;
        }

        n_generated++;
    }

    llama_sampler_free(sampler);

    return env->NewStringUTF(result.str().c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_rufatronics_farmbot_LlamaBridge_unloadModel(JNIEnv* env, jobject /* this */) {
    if (g_ctx != nullptr) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_model != nullptr) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
    llama_backend_free();
    LOGI("Model unloaded");
}
