import Foundation
import Testing
@testable import TurboFieldfareAppCore

/// Explicit opt-in: run only after the documented single-model preflight.
@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["QWEN_MODEL_SMOKE_TEST"] == "1"))
struct QwenInstalledModelTests {
    @Test func helloAndContinuationFinishWithoutLosingTheRuntime() async throws {
        let client = DecodeServiceInferenceClient()
        defer { client.shutdownForTermination() }
        let directory = QwenModelPackage.defaultDirectory()
        let options = AppRuntimeOptions()
        try QwenModelPackage.validate(at: directory)
        do {
            try await client.ensureLoaded(modelDirectory: directory, maxContextTokens: 4096,
                options: options, forceLogitsHead: false, onState: {
                    FileHandle.standardError.write(Data("Qwen smoke load: \($0)\n".utf8))
                })
            let epoch = UUID()
            try await client.resetConversation(epoch: epoch)
            var conversationTokens = 0
            for (index, prompt) in ["hello", "Say hello in one short sentence."].enumerated() {
                var finished = false
                let request = AppGenerationRequest(modelDirectory: directory, prompt: prompt,
                    maxNewTokens: 32, maxContextTokens: 4096, runtimeOptions: options,
                    continuesConversation: index > 0,
                    conversationTokens: conversationTokens,
                    conversationEpoch: epoch, turnIndex: index)
                for try await event in client.generate(request) {
                    switch event {
                    case .token(let token):
                        FileHandle.standardError.write(Data("Qwen smoke turn \(index), token \(token.index)\n".utf8))
                    case .finished(let diagnostics):
                        finished = true
                        conversationTokens = diagnostics.conversationTokens ?? 0
                        #expect(diagnostics.generatedTokens > 0)
                        print("Qwen smoke turn \(index): \(diagnostics)")
                    case .failed(let error, _): throw error
                    default: break
                    }
                }
                let output = client.generationTranscriptMailbox.completeText
                #expect(finished)
                #expect(!output.isEmpty)
                print("Qwen smoke output \(index): \(output)")
            }
            await client.unload()
        } catch {
            await client.unload()
            throw error
        }
    }
}
