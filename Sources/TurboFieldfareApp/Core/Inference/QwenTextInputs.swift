import Foundation
@preconcurrency import MLX
@preconcurrency import MLXLMCommon

/// Do not let a text processor silently discard an attachment.
enum QwenTextInputs {
    static func prepare(_ request: AppGenerationRequest,
                        tokenizer: any MLXLMCommon.Tokenizer) throws -> LMInput {
        guard request.imageAttachments.isEmpty else {
            throw AppInferenceError.invalidRequest("Image support is unavailable for this text-only Qwen model")
        }
        let tokens = try tokenizer.applyChatTemplate(
            messages: [["role": "user", "content": request.prompt]], tools: nil,
            additionalContext: ["enable_thinking": true, "preserve_thinking": true])
        return LMInput(tokens: MLXArray(tokens))
    }
}
