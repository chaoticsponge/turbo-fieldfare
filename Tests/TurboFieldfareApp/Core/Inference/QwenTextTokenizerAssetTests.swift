import Foundation
import Testing
import MLX
import MLXLLM
import MLXLMCommon
import TurboFieldfare
@testable import TurboFieldfareAppCore

/// Metadata and tiny token arrays only; no model or weight allocations.
@Suite(.enabled(if: ProcessInfo.processInfo.environment["QWEN_TEXT_TOKENIZER_ASSETS"] != nil))
struct QwenTextTokenizerAssetTests {
    private func directory(_ size: String) throws -> URL {
        let root = try #require(ProcessInfo.processInfo.environment["QWEN_TEXT_TOKENIZER_ASSETS"])
        return URL(fileURLWithPath: root).appendingPathComponent(size)
    }

    @Test(arguments: ["14", "32"])
    func configurationMatchesPinnedTextRuntime(_ size: String) throws {
        let data = try Data(contentsOf: directory(size).appendingPathComponent("config.json"))
        let config = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(config["model_type"] as? String == "qwen3")
        #expect(config["vocab_size"] as? Int == 151_936)
        #expect(config["image_token_id"] == nil)
        _ = try JSONDecoder().decode(Qwen3Configuration.self, from: data)
    }

    @Test(arguments: ["14", "32"])
    func embeddedTemplatePreservesThinkingAcrossTextTurns(_ size: String) async throws {
        let folder = try directory(size)
        let tokenizer = try await QwenTokenizerLoader().load(from: folder)
        let flags: [String: any Sendable] = ["enable_thinking": true, "preserve_thinking": true]
        let first = try tokenizer.applyChatTemplate(messages: [["role": "user", "content": "Hello"]],
            tools: nil, additionalContext: flags)
        let request = AppGenerationRequest(modelDirectory: folder, prompt: "Hello")
        let prepared = try QwenTextInputs.prepare(request, tokenizer: tokenizer)
        #expect(prepared.image == nil)
        #expect(prepared.text.tokens.asArray(Int.self) == first)
        let second = try tokenizer.applyChatTemplate(messages: [["role": "user", "content": "And you?"]],
            tools: nil, additionalContext: flags)
        let suffix = try QwenPromptPolicy.continuation(second,
            userPrefix: tokenizer.encode(text: "<|im_start|>user\n", addSpecialTokens: false))
        // Qwen3 emits the opening think tag itself, unlike the Qwen3.5 template.
        let generated = tokenizer.encode(
            text: "<think>\nI can greet the user.\n</think>\n\nHello!<|im_end|>\n", addSpecialTokens: false)
        let complete = try tokenizer.applyChatTemplate(messages: [
            ["role": "user", "content": "Hello"],
            ["role": "assistant", "reasoning_content": "I can greet the user.", "content": "Hello!"],
            ["role": "user", "content": "And you?"],
        ], tools: nil, additionalContext: flags)
        #expect(first + generated + suffix == complete)
        #expect(tokenizer.convertTokenToId("<|im_end|>") == 151_645)
        #expect(complete.allSatisfy { (0..<151_936).contains($0) })
        var withImage = request
        withImage.imageAttachments = [StagedImage(fileURL: folder.appendingPathComponent("missing.png"),
            displayName: "missing.png", encodedBytes: 1, sha256: String(repeating: "0", count: 64))]
        #expect(throws: AppInferenceError.self) {
            try QwenTextInputs.prepare(withImage, tokenizer: tokenizer)
        }
    }
}
