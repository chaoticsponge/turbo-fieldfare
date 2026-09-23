import Foundation
import Testing
import MLX
import MLXNN
import MLXLMCommon
@testable import TurboFieldfareAppCore

@Suite(.serialized) struct QwenPrefillTests {
    /// Small recurrent model: final logits depend on every prior chunk, so
    /// dropping a cache update changes both the first token and continuation.
    private final class SumModel: Module, LanguageModel {
        func newCache(parameters: GenerateParameters?) -> [any KVCache] { [ArraysCache(size: 1)] }
        func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws -> PrepareResult {
            .logits(self(input.text, cache: cache, state: nil))
        }
        func callAsFunction(_ input: LMInput.Text, cache: [any KVCache]?, state: LMOutput.State?) -> LMOutput {
            let memory = cache![0] as! ArraysCache
            let running = cumsum(input.tokens.asType(.float32), axis: 1) + (memory[0] ?? MLXArray(Float(0)))
            memory[0] = running[0, -1]
            return .init(logits: stacked([running, running * 2], axis: -1))
        }
    }

    @Test func cacheOnlyIntermediatePrefillMatchesFullPromptAndContinuation() throws {
        let model = SumModel()
        let chunkedCache = model.newCache(parameters: nil)
        let fullCache = model.newCache(parameters: nil)
        let input = LMInput(tokens: MLXArray([1, 2, 3, 4, 5], [1, 5]))
        let chunked = QwenPrefillModel(base: model, startingFresh: true, step: 2,
            checkCancellation: {}, progress: { _, _ in })
        guard case .logits(let result) = try chunked.prepare(input, cache: chunkedCache, windowSize: nil),
              case .logits(let reference) = try model.prepare(input, cache: fullCache, windowSize: nil) else {
            Issue.record("Missing logits"); return
        }
        #expect(result.logits[0, -1].asArray(Float.self) == reference.logits[0, -1].asArray(Float.self))
        #expect((chunkedCache[0] as! ArraysCache)[0]!.item(Float.self) == 15)
        let continuation = QwenPrefillModel(base: model, startingFresh: false, step: 1,
            checkCancellation: {}, progress: { _, _ in })
        guard case .logits(let next) = try continuation.prepare(
            .init(tokens: MLXArray([6, 7], [1, 2])), cache: chunkedCache, windowSize: nil) else {
            Issue.record("Missing continuation logits"); return
        }
        #expect(next.logits[0, -1].asArray(Float.self) == [28, 56])
    }

    private final class TraceModel: Module, LanguageModel {
        var preparations = 0
        var evaluated: [[Int]] = []
        func newCache(parameters: GenerateParameters?) -> [any KVCache] { [] }
        func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws -> PrepareResult {
            preparations += 1
            return .logits(self(input.text, cache: cache, state: nil))
        }
        func callAsFunction(_ input: LMInput.Text, cache: [any KVCache]?, state: LMOutput.State?) -> LMOutput {
            evaluated.append(input.tokens.asArray(Int.self))
            return .init(logits: MLXArray.zeros([1, input.tokens.size, 8], stream: .cpu))
        }
    }

    @Test func textPrefillIsBoundedAndContinuationKeepsPositionState() throws {
        let base = TraceModel()
        var progress: [Int] = []
        let opening = QwenPrefillModel(base: base, startingFresh: true, step: 2,
            checkCancellation: {}, progress: { done, _ in progress.append(done) })
        _ = try opening.prepare(.init(tokens: MLXArray([1, 2, 3, 4, 5], [1, 5])), cache: [], windowSize: nil)
        #expect(base.preparations == 1)
        #expect(base.evaluated == [[1, 2], [3, 4], [5]])
        #expect(progress == [0, 2, 4, 5])
        let continuation = QwenPrefillModel(base: base, startingFresh: false, step: 2,
            checkCancellation: {}, progress: { _, _ in })
        _ = try continuation.prepare(.init(tokens: MLXArray([6, 7, 8], [1, 3])), cache: [], windowSize: nil)
        #expect(base.preparations == 1)
        #expect(base.evaluated == [[1, 2], [3, 4], [5], [6, 7], [8]])
    }

    @Test func batchedPromptCanSampleWithRepetitionPenalty() throws {
        // Previously loadPrompt treated [1, 5] as one token, created an
        // oversized ring buffer, and trapped on the first sampled token.
        for penalty: Float in [1.0, 1.1] {
            let base = TraceModel()
            let model = QwenPrefillModel(base: base, startingFresh: true, step: 2,
                checkCancellation: {}, progress: { _, _ in })
            var iterator = try QwenSampling.iterator(
                input: .init(tokens: MLXArray([1, 2, 3, 4, 5], [1, 5])),
                model: model, cache: [], parameters: GenerateParameters(
                    maxTokens: 2, temperature: 0, repetitionPenalty: penalty))
            #expect(iterator.next() != nil)
            #expect(iterator.next() != nil)
            #expect(iterator.next() == nil)
            #expect(base.evaluated.prefix(3).elementsEqual([[1, 2], [3, 4], [5]]))
        }
    }

    @Test func cancellationStopsBeforeTheNextChunk() {
        let base = TraceModel()
        var stopped = false
        let model = QwenPrefillModel(base: base, startingFresh: true, step: 2,
            checkCancellation: { if stopped { throw CancellationError() } },
            progress: { done, _ in if done == 2 { stopped = true } })
        #expect(throws: CancellationError.self) {
            _ = try model.prepare(.init(tokens: MLXArray([1, 2, 3, 4], [1, 4])), cache: [], windowSize: nil)
        }
        #expect(base.evaluated == [[1, 2]])
    }
}
