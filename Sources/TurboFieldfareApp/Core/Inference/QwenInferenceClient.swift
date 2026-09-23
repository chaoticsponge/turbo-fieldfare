import Foundation
import Synchronization
import TurboFieldfare
@preconcurrency import MLX
@preconcurrency import MLXLMCommon
@preconcurrency import MLXVLM
@preconcurrency import MLXLLM

/// Native MLX model owner, used only by the sibling decode service.
public final class QwenInferenceClient: AppModelLifecycleClient, Sendable {
    private let session = QwenInferenceSession()
    private let tasks = GenerationTaskRegistry()

    public init() {}
    public var currentVisionTowerBytes: UInt64? { nil }
    public var currentConversationTokens: Int { session.count.withLock { $0 } }
    public var conversationTokenCount: Int { get async { currentConversationTokens } }

    public func ensureLoaded(modelDirectory: URL, maxContextTokens: Int,
        options: AppRuntimeOptions, forceLogitsHead: Bool,
        onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws {
        guard !tasks.hasActiveGeneration else { throw AppInferenceError.generationInFlight }
        do {
            try await session.load(directory: modelDirectory, maximum: maxContextTokens,
                options: options, onState: onState)
        } catch {
            // The factory's local weight references have unwound by this point.
            // Release reusable Metal allocations even when loading never
            // published a context (including cancellation during load).
            await session.unload()
            throw error
        }
    }
    public func unload() async {
        let task = tasks.currentTask
        task?.cancel()
        await task?.value
        await session.unload()
    }
    public func resetConversation() async {
        let task = tasks.currentTask
        task?.cancel()
        await task?.value
        await session.reset()
    }
    public func resetConversation(epoch: UUID) async throws { await resetConversation() }
    public func restoreConversation(_ lineage: AppConversationLineage, epoch: UUID = UUID(),
        options: AppRuntimeOptions, maxContextTokens: Int,
        onPrefillProgress: @escaping @Sendable (Int, Int) -> Void = { _, _ in }) async throws -> Int {
        guard !tasks.hasActiveGeneration else { throw AppInferenceError.generationInFlight }
        return try await session.restore(lineage, options: options, progress: onPrefillProgress)
    }
    public func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream { continuation in
            let id = UUID()
            guard tasks.reserve(id) else {
                continuation.finish(throwing: AppInferenceError.generationInFlight)
                return
            }
            session.stopRequested.withLock { $0 = false }
            let task = Task { [self] in
                await session.run(request, continuation: continuation)
                tasks.clear(id)
            }
            tasks.attach(task, to: id)
            continuation.onTermination = { [tasks] _ in tasks.take(id)?.cancel() }
        }
    }
    public func cancel() { tasks.currentTask?.cancel() }
    public func stop() { session.stopRequested.withLock { $0 = true } }
}

private actor QwenInferenceSession {
    nonisolated let count = Mutex(0)
    nonisolated let stopRequested = Mutex(false)
    private var context: ModelContext?
    private var cache: [any KVCache] = []
    private var directory: URL?
    private var maximum = 4096
    private var variant: QwenModelVariant = .default
    private var pendingBoundary: [Int] = []
    private var retainedTokens: [Int32] = []
    private var retainedImages: LMInput.ProcessedImage?
    private var needsReplay = false
    private let memory = AppMemorySampler()

    func load(directory: URL, maximum: Int, options: AppRuntimeOptions,
        onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws {
        try QwenModelPackage.requireMetalLibrary()
        unload()
        let start = Date()
        onState(.loading(.validatingDirectory))
        guard maximum > 0 else { throw AppInferenceError.invalidRequest("Context must be positive") }
        let selected = try QwenModelPackage.variant(at: directory)
        try QwenModelPackage.validateWeightBudget(for: selected,
            recommendedWorkingSetBytes: GPU.maxRecommendedWorkingSetBytes())
        onState(.loading(.verifyingWeights))
        try QwenModelPackage.validate(at: directory, hashes: options.modelVerification == .fullSha256)
        try Task.checkCancellation()
        // Unified memory is shared with the OS. Bound reusable allocations;
        // do not raise the system's wired-memory limit.
        Memory.cacheLimit = options.expertCacheSlots * 4 * 1_048_576
        onState(.loading(.preparingRunner))
        var loaded: ModelContext
        if selected.supportsImages {
            loaded = try await VLMModelFactory.shared.load(from: directory, using: QwenTokenizerLoader())
            loaded.processor = try QwenImageInputs.processor(in: directory, tokenizer: loaded.tokenizer)
        } else {
            loaded = try await LLMModelFactory.shared.load(from: directory, using: QwenTokenizerLoader())
        }
        try Task.checkCancellation()
        context = loaded
        variant = selected
        self.directory = directory.standardizedFileURL
        self.maximum = maximum
        reset()
        onState(.ready(modelDirectory: directory, loadSeconds: Date().timeIntervalSince(start)))
    }

    func reset() {
        cache = context?.model.newCache(parameters: nil) ?? []
        pendingBoundary = []
        retainedTokens = []
        retainedImages = nil
        needsReplay = false
        count.withLock { $0 = 0 }
        // New Chat releases the prior lineage's reusable GPU allocations as
        // well as its references. Live quantized model weights remain loaded.
        Memory.clearCache()
    }
    func unload() {
        cache = []
        context = nil
        directory = nil
        pendingBoundary = []
        retainedTokens = []
        retainedImages = nil
        needsReplay = false
        count.withLock { $0 = 0 }
        Memory.clearCache()
    }

    func restore(_ lineage: AppConversationLineage, options: AppRuntimeOptions,
        progress: @escaping @Sendable (Int, Int) -> Void) async throws -> Int {
        guard let context else { throw AppInferenceError.modelNotLoaded }
        guard !lineage.tokenIDs.isEmpty, lineage.tokenIDs.count <= maximum,
              lineage.tokenIDs.allSatisfy({ (0..<variant.vocabularySize).contains(Int($0)) }),
              lineage.boundaryTokenIDs.allSatisfy({ (0..<variant.vocabularySize).contains(Int($0)) }) else {
            throw AppInferenceError.conversationRestoreFailed("Invalid Qwen token lineage")
        }
        try variant.validateImageCount(lineage.images.count)
        reset()
        do {
            let images = variant.supportsImages
                ? try QwenImageInputs.replay(lineage.images, tokens: lineage.tokenIDs) : nil
            let input = LMInput(text: .init(tokens: MLXArray(lineage.tokenIDs).expandedDimensions(axis: 0)), image: images)
            progress(0, lineage.tokenIDs.count)
            let step = options.prefillEnabled ? options.prefillChunkTokens : 1
            let model = QwenPrefillModel(base: context.model, startingFresh: true, step: step,
                checkCancellation: { try Task.checkCancellation() }, progress: progress)
            _ = try model.prepare(input, cache: cache, windowSize: step)
            eval(cache)
            try Task.checkCancellation()
            count.withLock { $0 = lineage.tokenIDs.count }
            pendingBoundary = lineage.boundaryNeedsReplay ? lineage.boundaryTokenIDs.map(Int.init) : []
            retainedTokens = lineage.tokenIDs
            retainedImages = images
            progress(lineage.tokenIDs.count, lineage.tokenIDs.count)
            return lineage.tokenIDs.count
        } catch {
            reset()
            throw error
        }
    }

    func run(_ request: AppGenerationRequest,
        continuation: AsyncThrowingStream<AppInferenceEvent, Error>.Continuation) async {
        guard let budget = GPU.maxRecommendedWorkingSetBytes(), budget > 0 else {
            await runWithResidentMemory(request, continuation: continuation)
            return
        }
        // Use only the GPU's recommended process budget. The ticket restores
        // the prior residency limit when generation completes or is cancelled.
        let ticket = WiredFixedPolicy(limit: budget).ticket(size: 0)
        _ = await ticket.start()
        // The inner operation handles all errors, including cancellation, so
        // every return releases the ticket without crossing actor isolation.
        await runWithResidentMemory(request, continuation: continuation)
        _ = await ticket.end()
    }

    private func runWithResidentMemory(_ request: AppGenerationRequest,
        continuation: AsyncThrowingStream<AppInferenceEvent, Error>.Continuation) async {
        let originalCount = count.withLock { $0 }
        let originalBoundary = pendingBoundary
        let originalTokens = retainedTokens
        let originalImages = retainedImages
        let previous = cache.map { $0.copy() }
        memory.resetPeak()
        do {
            try request.validate()
            guard let context, directory == request.modelDirectory.standardizedFileURL,
                  request.maxContextTokens == maximum else { throw AppInferenceError.modelNotLoaded }
            if !request.continuesConversation { reset() }
            let carried = count.withLock { $0 }
            let start = Date()
            try variant.validateImageCount(request.imageAttachments.count)
            let prepared: LMInput
            if variant.supportsImages {
                prepared = try await QwenImageInputs.prepare(request, context: context)
            } else {
                prepared = try QwenTextInputs.prepare(request, tokenizer: context.tokenizer)
            }
            try Task.checkCancellation()
            if stopRequested.withLock({ $0 }) { throw CancellationError() }
            var prompt = prepared.text.tokens.asArray(Int.self)
            if carried > 0 {
                prompt = try QwenPromptPolicy.continuation(prompt,
                    userPrefix: context.tokenizer.encode(text: "<|im_start|>user\n", addSpecialTokens: false))
                prompt = pendingBoundary + context.tokenizer.encode(text: "\n", addSpecialTokens: false) + prompt
            }
            guard !prompt.isEmpty, carried + prompt.count < maximum else {
                throw AppInferenceError.invalidRequest("This conversation fills the Qwen context. Start a new chat or increase context length.")
            }
            let limit = min(request.maxNewTokens, maximum - carried - prompt.count)
            let parameters = GenerateParameters(maxTokens: limit, temperature: request.temperature,
                topP: request.topP ?? 1, topK: request.topK ?? 0,
                repetitionPenalty: request.repetitionPenalty,
                prefillStepSize: request.runtimeOptions.prefillChunkTokens)
            // Appending a new image changes multimodal positions. Replay all
            // recorded inputs then; plain text turns keep their live cache.
            let hasNewImages = prepared.image != nil
            let replay = carried > 0 && (hasNewImages || needsReplay)
            let allImages = QwenImageInputs.combine(retainedImages, prepared.image)
            let evaluatedTokens = replay ? retainedTokens.map(Int.init) + prompt : prompt
            if replay { cache = context.model.newCache(parameters: nil) }
            let input = LMInput(text: .init(tokens: MLXArray(evaluatedTokens).expandedDimensions(axis: 0)),
                image: replay ? allImages : prepared.image)
            let session = self
            let memory = memory
            let model = QwenPrefillModel(base: context.model, startingFresh: carried == 0 || replay,
                step: request.runtimeOptions.prefillEnabled ? request.runtimeOptions.prefillChunkTokens : 1,
                checkCancellation: {
                    try Task.checkCancellation()
                    if session.stopRequested.withLock({ $0 }) { throw CancellationError() }
                }, progress: { done, total in
                    _ = memory.sample()
                    continuation.yield(.prefillProgress(done: done, total: total))
                })
            var iterator = try QwenSampling.iterator(input: input, model: model, cache: cache, parameters: parameters)
            let prefillEnd = Date()
            continuation.yield(.prefillProgress(done: prompt.count, total: prompt.count))
            let endToken = context.tokenizer.convertTokenToId("<|im_end|>")
            let eos = Set([endToken, context.tokenizer.eosTokenId].compactMap { $0 })
            var generated: [Int32] = []
            var detokenizer = NaiveStreamingDetokenizer(tokenizer: context.tokenizer)
            var reason: AppStopReason = .maxTokens
            var firstToken: Date?
            while let token = iterator.next() {
                try Task.checkCancellation()
                firstToken = firstToken ?? Date()
                generated.append(Int32(token))
                _ = memory.sample()
                if eos.contains(token) { reason = .endOfTurn; break }
                detokenizer.append(token: token)
                continuation.yield(.token(AppTokenEvent(index: generated.count - 1,
                    textDelta: detokenizer.next() ?? "",
                    elapsedDecodeSeconds: Date().timeIntervalSince(prefillEnd))))
                if stopRequested.withLock({ $0 }) { reason = .cancelled; break }
            }
            eval(cache)
            try Task.checkCancellation()
            pendingBoundary = reason == .endOfTurn ? [] : endToken.map { [$0] } ?? []
            retainedTokens += prompt.map(Int32.init) + generated
            retainedImages = allImages
            needsReplay = false
            count.withLock { $0 = carried + prompt.count + generated.count }
            let elapsed = Date().timeIntervalSince(prefillEnd)
            continuation.yield(.finished(AppDiagnostics(generatedTokens: generated.count,
                stopReason: reason, promptTokenCount: carried + prompt.count,
                cachedPromptTokens: replay ? 0 : carried, computedPrefillTokens: evaluatedTokens.count,
                conversationTokens: currentCount, promptTokenIDs: prompt.map(Int32.init),
                generatedTokenIDs: generated, boundaryTokenIDs: pendingBoundary.map(Int32.init),
                boundaryNeedsReplay: !pendingBoundary.isEmpty,
                prefillSeconds: prefillEnd.timeIntervalSince(start),
                timeToFirstTokenSeconds: firstToken.map { $0.timeIntervalSince(prefillEnd) },
                decodeSeconds: elapsed, tokensPerSecond: elapsed > 0 ? Double(generated.count) / elapsed : 0,
                peakMemoryBytes: memory.peakBytes, runtimeOptions: request.runtimeOptions)))
            continuation.finish()
        } catch {
            cache = previous
            pendingBoundary = originalBoundary
            retainedTokens = originalTokens
            retainedImages = originalImages
            needsReplay = originalCount > 0
            count.withLock { $0 = originalCount }
            let failure = error is CancellationError ? AppInferenceError.cancelled
                : (error as? AppInferenceError ?? .unknown(String(describing: error)))
            continuation.yield(.failed(failure, partial: nil))
            continuation.finish(throwing: failure)
        }
    }
    private var currentCount: Int { count.withLock { $0 } }
}
