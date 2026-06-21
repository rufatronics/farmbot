package com.rufatronics.farmbot

/**
 * The regex safety net discussed in the model card's "Known Limitations" section.
 * Since the model's own out-of-scope detection is unreliable (benchmarked at 0/4
 * in raw generation), this catches obvious off-topic questions BEFORE they reach
 * the model, guaranteeing a clean, on-brand decline every time.
 */
object SafetyFilter {

    private val oosPatterns = listOf(
        Regex("\\b(petrol|diesel|fuel|dollar|naira|exchange rate|bitcoin)\\b", RegexOption.IGNORE_CASE),
        Regex("\\b(football|match|messi|premier league|champions league)\\b", RegexOption.IGNORE_CASE),
        Regex("\\b(jollof|egusi|pepper soup|recipe|cook)\\b", RegexOption.IGNORE_CASE),
        Regex("\\b(goat|chicken|cow|pig|poultry)\\b.*\\b(sick|disease|medicine)\\b", RegexOption.IGNORE_CASE),
        Regex("\\b(president|election|politics|government)\\b", RegexOption.IGNORE_CASE),
        Regex("\\b(phone|laptop|wifi|whatsapp|instagram|tiktok)\\b", RegexOption.IGNORE_CASE)
    )

    const val STATIC_OOS_RESPONSE =
        "I can only help with cassava, cocoa, cowpea, maize, groundnut, mango, " +
        "plantain, rice and tomato my friend. For anything else please contact " +
        "your local extension officer or agricultural expert."

    fun isOutOfScope(question: String): Boolean =
        oosPatterns.any { it.containsMatchIn(question) }
}

/**
 * Maps the vision model's 10 disease class outputs to natural-language questions
 * that FarmBot's training data was built around.
 */
object DiseaseLabelMapper {

    private val labelToQuestion = mapOf(
        "cassava_mosaic"        to "my cassava leaves have yellow and green patterns what is wrong",
        "cocoa_black_pod"       to "my cocoa pods are turning black and rotting",
        "cowpea_aphid"          to "I see black insects on my cowpea plant",
        "fall_armyworm"         to "my maize leaves have holes and ragged marks",
        "groundnut_rosette"     to "my groundnut plants are short, bushy and yellow",
        "maize_streak_virus"    to "my maize leaves have streaky yellow lines what is this",
        "mango_anthracnose"     to "my mango fruits have dark spots and are turning black",
        "plantain_bunchy_top"   to "my plantain leaves are short and bunched at the top",
        "rice_blast"            to "my rice leaves have diamond shaped spots",
        "tomato_blight"         to "my tomato plant has brown spots and is dying"
    )

    fun questionFor(diseaseLabel: String): String =
        labelToQuestion[diseaseLabel]
            ?: "what can you tell me about $diseaseLabel"
}

/**
 * Coordinates the full pipeline: vision label or free-text question -> safety filter
 * -> prompt formatting -> native model call -> response.
 */
class FarmBotEngine(private val llama: LlamaBridge) {

    fun respond(
        question: String,
        maxNewTokens: Int = 120,
        temperature: Float = 0.3f,
        topK: Int = 20,
        repeatPenalty: Float = 1.2f
    ): String {
        if (SafetyFilter.isOutOfScope(question)) {
            return SafetyFilter.STATIC_OOS_RESPONSE
        }

        val prompt = "<|im_start|>user\n$question<|im_end|>\n<|im_start|>assistant\n"
        return llama.generate(prompt, maxNewTokens, temperature, topK, repeatPenalty).trim()
    }

    fun respondToDiseaseLabel(diseaseLabel: String): String {
        val question = DiseaseLabelMapper.questionFor(diseaseLabel)
        return respond(question)
    }
}
