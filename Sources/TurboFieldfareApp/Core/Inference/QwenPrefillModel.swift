import Foundation
@preconcurrency import MLX
@preconcurrency import MLXNN
@preconcurrency import MLXLMCommon

/// MLX's Qwen VLM `prepare` starts image position bookkeeping again. Text
/// continuations must use the retained position state instead. This adapter
/// also bounds text prefill, which the pinned VLM's prepare does not chunk.
final class QwenPrefillModel: Module, LanguageModel {
    let base: any LanguageModel
    let startingFresh: Bool
    let step: Int
    let checkCancellation: () throws -> Void
    let progress: (Int, Int) -> Void

    init(base: any LanguageModel, startingFresh: Bool, step: Int,
        checkCancellation: @escaping () throws -> Void,
        progress: @escaping (Int, Int) -> Void) {
        self.base = base
        self.startingFresh = startingFresh
        self.step = max(1, step)
        self.checkCancellation = checkCancellation
        self.progress = progress
        super.init()
    }

    func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        base.newCache(parameters: parameters)
    }
    func callAsFunction(_ input: LMInput.Text, cache: [any KVCache]?, state: LMOutput.State?) -> LMOutput {
        base(input, cache: cache, state: state)
    }
    func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws -> PrepareResult {
        try checkCancellation()
        let total = input.text.tokens.size
        progress(0, total)
        if input.image != nil || input.video != nil {
            // New image positions are computed over the complete retained
            // lineage; the caller supplies a fresh cache for this case.
            let result = try base.prepare(input, cache: cache, windowSize: windowSize)
            eval(cache)
            try checkCancellation()
            progress(total, total)
            return result
        }
        var last: LMOutput?
        for offset in stride(from: 0, to: total, by: step) {
            try checkCancellation()
            let end = min(total, offset + step)
            let text = LMInput.Text(tokens: input.text.tokens[0..., offset..<end])
            let output: LMOutput
            if offset == 0 && startingFresh {
                switch try base.prepare(.init(text: text), cache: cache, windowSize: step) {
                case .logits(let prepared): output = prepared
                case .tokens(let rest): output = base(rest, cache: cache, state: nil)
                }
            } else {
                output = base(text, cache: cache, state: nil)
            }
            if end == total {
                // Only the final chunk supplies the next-token distribution.
                eval(output.logits, cache)
                last = output
            } else {
                // MLX is lazy: evaluating just the KV/recurrent state avoids
                // running the unused vocabulary projection for this chunk.
                // Materialize all layers together to bound the graph without
                // synchronizing once per layer or retaining previous logits.
                eval(cache)
            }
            try checkCancellation()
            progress(end, total)
        }
        guard let last else { throw AppInferenceError.invalidRequest("Qwen prompt is empty") }
        return .logits(last)
    }
}
