package com.rufatronics.farmbot

import android.graphics.Bitmap
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {

    private lateinit var statusText: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var chatLog: TextView
    private lateinit var questionInput: EditText
    private lateinit var sendButton: Button
    private lateinit var scanButton: Button

    private lateinit var engine: FarmBotEngine
    private lateinit var llama: LlamaBridge
    private var vision: VisionClassifier? = null

    private var modelsLoaded = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        statusText    = findViewById(R.id.statusText)
        progressBar   = findViewById(R.id.progressBar)
        chatLog       = findViewById(R.id.chatLog)
        questionInput = findViewById(R.id.questionInput)
        sendButton    = findViewById(R.id.sendButton)
        scanButton    = findViewById(R.id.scanButton)

        sendButton.isEnabled = false
        scanButton.isEnabled = false

        llama = LlamaBridge()
        engine = FarmBotEngine(llama)

        initializeModels()

        sendButton.setOnClickListener {
            val question = questionInput.text.toString().trim()
            if (question.isNotEmpty() && modelsLoaded) {
                appendChat("You", question)
                questionInput.text.clear()
                runInference(question)
            }
        }

        scanButton.setOnClickListener {
            // Hook point for CameraX capture -> Bitmap -> vision.classify(bitmap)
            // Kept minimal here; wire to CameraX PreviewView + ImageCapture as needed.
            statusText.text = "Camera capture not wired in this build — use text input for demo"
        }
    }

    private fun initializeModels() {
        statusText.text = "Preparing models (first launch only)..."
        progressBar.visibility = ProgressBar.VISIBLE

        lifecycleScope.launch {
            // Step 1: copy bundled assets to internal storage (fast after first run)
            withContext(Dispatchers.IO) {
                ModelAssets.prepareModels(applicationContext) { msg ->
                    runOnUiThread { statusText.text = msg }
                }
            }

            // Step 2: load language model into llama.cpp
            statusText.text = "Loading FarmBot language model..."
            val llmPath = ModelAssets.llmFile(applicationContext).absolutePath
            val llmOk = withContext(Dispatchers.Default) {
                llama.loadModel(llmPath, nThreads = 4, nCtx = 512)
            }

            // Step 3: load vision model
            statusText.text = "Loading vision model..."
            val visionOk = withContext(Dispatchers.Default) {
                try {
                    vision = VisionClassifier(applicationContext, ModelAssets.visionFile(applicationContext))
                    true
                } catch (e: Exception) {
                    false
                }
            }

            progressBar.visibility = ProgressBar.GONE

            if (llmOk) {
                modelsLoaded = true
                sendButton.isEnabled = true
                scanButton.isEnabled = visionOk
                statusText.text = "FarmBot is ready. Ask me about your crops!"
                appendChat(
                    "FarmBot",
                    "Welcome my friend! I dey here to help you with cassava, cocoa, cowpea, " +
                    "maize, groundnut, mango, plantain, rice and tomato. Wetin be your farm problem today?"
                )
            } else {
                statusText.text = "Failed to load language model. Please reinstall the app."
            }
        }
    }

    private fun runInference(question: String) {
        sendButton.isEnabled = false
        statusText.text = "FarmBot is thinking..."

        lifecycleScope.launch {
            val response = withContext(Dispatchers.Default) {
                engine.respond(question)
            }
            appendChat("FarmBot", response)
            statusText.text = "FarmBot is ready. Ask me about your crops!"
            sendButton.isEnabled = true
        }
    }

    private fun onDiseaseDetected(bitmap: Bitmap) {
        val v = vision ?: return
        lifecycleScope.launch {
            statusText.text = "Analyzing photo..."
            val prediction = withContext(Dispatchers.Default) { v.classify(bitmap) }
            appendChat("You", "[Photo uploaded — detected: ${prediction.label}, " +
                    "confidence ${(prediction.confidence * 100).toInt()}%]")

            if (prediction.confidence < 0.5f) {
                appendChat("FarmBot",
                    "The image isn't clear enough for me to be confident. " +
                    "Can you describe what you see on the plant instead?")
                return@launch
            }

            statusText.text = "FarmBot is thinking..."
            val response = withContext(Dispatchers.Default) {
                engine.respondToDiseaseLabel(prediction.label)
            }
            appendChat("FarmBot", response)
            statusText.text = "FarmBot is ready. Ask me about your crops!"
        }
    }

    private fun appendChat(speaker: String, message: String) {
        chatLog.append("\n\n$speaker: $message")
    }

    override fun onDestroy() {
        super.onDestroy()
        llama.unloadModel()
        vision?.close()
    }
}
