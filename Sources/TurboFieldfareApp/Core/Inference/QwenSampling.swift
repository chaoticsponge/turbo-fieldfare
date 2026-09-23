import MLX
import MLXLMCommon

enum QwenSampling {
    /// Qwen's model input is [batch, tokens], while MLX's penalty processors
    /// require [tokens]. Keep the model shape intact and adapt only penalties.
    static func iterator(input: LMInput, model: any LanguageModel,
        cache: [any KVCache], parameters: GenerateParameters) throws -> TokenIterator {
        try TokenIterator(input: input, model: model, cache: cache,
            processor: parameters.processor().map { FlatPromptProcessor(base: $0) },
            sampler: parameters.sampler(), prefillStepSize: parameters.prefillStepSize,
            maxTokens: parameters.maxTokens)
    }

    private struct FlatPromptProcessor: LogitProcessor {
        var base: any LogitProcessor
        mutating func prompt(_ prompt: MLXArray) { base.prompt(prompt.reshaped(-1)) }
        func process(logits: MLXArray) -> MLXArray { base.process(logits: logits) }
        mutating func didSample(token: MLXArray) { base.didSample(token: token) }
    }
}
