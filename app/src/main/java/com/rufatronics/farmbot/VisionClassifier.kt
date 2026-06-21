package com.rufatronics.farmbot

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.File
import java.nio.MappedByteBuffer

/**
 * Wraps the crop_doctor.tflite model (EfficientNet-Lite, 10 classes, 4.56MB).
 * Loaded from app-private storage at runtime, same as the language model,
 * so the whole app works fully offline once both files are bundled/downloaded.
 */
class VisionClassifier(context: Context, modelFile: File) {

    private val interpreter: Interpreter
    private val inputSize = 224  // standard EfficientNet-Lite input size

    // Must match the order used when your friend trained crop_doctor.tflite
    private val labels = listOf(
        "cassava_mosaic",
        "cocoa_black_pod",
        "cowpea_aphid",
        "fall_armyworm",
        "groundnut_rosette",
        "maize_streak_virus",
        "mango_anthracnose",
        "plantain_bunchy_top",
        "rice_blast",
        "tomato_blight"
    )

    init {
        val options = Interpreter.Options().apply {
            setNumThreads(4)
        }
        interpreter = Interpreter(loadModelFile(modelFile), options)
    }

    private fun loadModelFile(file: File): MappedByteBuffer {
        return FileUtil.loadMappedFile(file.absolutePath.let {
            // FileUtil expects a context+asset path normally; for a raw file path
            // we memory-map it directly.
            file
        })
    }

    data class Prediction(val label: String, val confidence: Float)

    fun classify(bitmap: Bitmap): Prediction {
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(inputSize, inputSize, ResizeOp.ResizeMethod.BILINEAR))
            .build()

        var tensorImage = TensorImage(org.tensorflow.lite.DataType.FLOAT32)
        tensorImage.load(bitmap)
        tensorImage = imageProcessor.process(tensorImage)

        val output = Array(1) { FloatArray(labels.size) }
        interpreter.run(tensorImage.buffer, output)

        val scores = output[0]
        var maxIdx = 0
        for (i in scores.indices) {
            if (scores[i] > scores[maxIdx]) maxIdx = i
        }

        return Prediction(labels[maxIdx], scores[maxIdx])
    }

    fun close() {
        interpreter.close()
    }
}
