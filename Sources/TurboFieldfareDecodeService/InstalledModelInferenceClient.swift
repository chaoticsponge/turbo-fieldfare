import Foundation
import Synchronization
import TurboFieldfareAppCore

/// The command queue serializes load/unload/generate. Only stop is called from
/// the input thread, so publish the selected backend under a lock.
final class InstalledModelInferenceClient: Sendable {
    private enum Backend: Sendable {
        case gemma(RealInferenceClient)
        case qwen(QwenInferenceClient)

        var lifecycle: any AppModelLifecycleClient {
            switch self {
            case .gemma(let client): client
            case .qwen(let client): client
            }
        }
    }
    private let backend = Mutex<Backend>(.qwen(QwenInferenceClient()))
    private var current: Backend { backend.withLock { $0 } }

    func ensureLoaded(modelDirectory: URL, maxContextTokens: Int,
                      options: AppRuntimeOptions, forceLogitsHead: Bool,
                      onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws {
        let previous = current
        let next: Backend
        switch (previous, QwenModelPackage.isQwen(at: modelDirectory)) {
        case (.gemma, false), (.qwen, true): next = previous
        case (_, true):
            await previous.lifecycle.unload()
            next = .qwen(QwenInferenceClient())
        case (_, false):
            await previous.lifecycle.unload()
            next = .gemma(RealInferenceClient())
        }
        backend.withLock { $0 = next }
        try Task.checkCancellation()
        try await next.lifecycle.ensureLoaded(modelDirectory: modelDirectory,
            maxContextTokens: maxContextTokens, options: options,
            forceLogitsHead: forceLogitsHead, onState: onState)
    }

    func unload() async { await current.lifecycle.unload() }
    func resetConversation() async {
        switch current {
        case .gemma(let client): await client.resetConversation()
        case .qwen(let client): await client.resetConversation()
        }
    }
    func stop() {
        switch current {
        case .gemma(let client): client.stop()
        case .qwen(let client): client.stop()
        }
    }
    var currentVisionTowerBytes: UInt64? {
        switch current {
        case .gemma(let client): client.currentVisionTowerBytes
        case .qwen(let client): client.currentVisionTowerBytes
        }
    }
    var currentConversationTokens: Int {
        switch current {
        case .gemma(let client): client.currentConversationTokens
        case .qwen(let client): client.currentConversationTokens
        }
    }
    var conversationTokenCount: Int {
        get async {
            switch current {
            case .gemma(let client): await client.conversationTokenCount
            case .qwen(let client): await client.conversationTokenCount
            }
        }
    }
    func restoreConversation(_ lineage: AppConversationLineage, epoch: UUID,
        options: AppRuntimeOptions, maxContextTokens: Int,
        onPrefillProgress: @escaping @Sendable (Int, Int) -> Void) async throws -> Int {
        try await current.lifecycle.restoreConversation(lineage, epoch: epoch,
            options: options, maxContextTokens: maxContextTokens,
            onPrefillProgress: onPrefillProgress)
    }
    func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error> {
        current.lifecycle.generate(request)
    }
}
