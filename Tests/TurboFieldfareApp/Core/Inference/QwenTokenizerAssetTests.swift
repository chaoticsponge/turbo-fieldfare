import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import Testing
import MLX
import MLXLMCommon
import MLXVLM
import TurboFieldfare
@testable import TurboFieldfareAppCore

/// These tests use tokenizer metadata only, never model weights.
@Suite(.enabled(if: ProcessInfo.processInfo.environment["QWEN_TOKENIZER_TEST_DIRECTORY"] != nil))
struct QwenTokenizerAssetTests {
    @Test func configurationMatchesThePinnedVLMArchitecture() throws {
        let path = try #require(ProcessInfo.processInfo.environment["QWEN_TOKENIZER_TEST_DIRECTORY"])
        let data = try Data(contentsOf: URL(fileURLWithPath: path).appendingPathComponent("config.json"))
        let config = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(config["model_type"] as? String == "qwen3_5")
        #expect(config["image_token_id"] as? Int == Int(QwenImageInputs.imageToken))
        let text = try #require(config["text_config"] as? [String: Any])
        #expect(text["vocab_size"] as? Int == 248_320)
        // Decode using the actual pinned model implementation, without creating
        // a model or allocating its weights.
        _ = try JSONDecoder().decode(Qwen35Configuration.self, from: data)
    }

    @Test func preparedImageSurvivesStoredReplayAndMissingInputsFailClosed() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["QWEN_TOKENIZER_TEST_DIRECTORY"])
        let modelDirectory = URL(fileURLWithPath: path)
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = directory.appendingPathComponent("red.png")
        let canvas = try #require(CGContext(data: nil, width: 32, height: 32,
            bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        canvas.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1))
        canvas.fill(CGRect(x: 0, y: 0, width: 32, height: 32))
        let destination = try #require(CGImageDestinationCreateWithURL(source as CFURL,
            UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, try #require(canvas.makeImage()), nil)
        #expect(CGImageDestinationFinalize(destination))
        let attachment = StagedImage(fileURL: source, displayName: "red.png",
            encodedBytes: try Data(contentsOf: source).count,
            sha256: try Sha256Verifier.hashFile(at: source, chunkBytes: 65_536))
        let result = await QwenImageInputs.store([attachment], modelDirectory: modelDirectory,
            into: directory.appendingPathComponent("images"))
        #expect(result.failed == 0)
        let record = try #require(result.records.first)
        let tokenizer = try await QwenTokenizerLoader().load(from: modelDirectory)
        let processor = try QwenImageInputs.processor(in: modelDirectory, tokenizer: tokenizer)
        let input = QwenImageInputs.userInput(prompt: "What color?", images: [.url(source)])
        let live = try await processor.prepare(input: input)
        let tokens = live.text.tokens.asArray(Int32.self)
        let lower = try #require(tokens.firstIndex(of: QwenImageInputs.imageToken))
        #expect(tokens.filter { $0 == QwenImageInputs.imageToken }.count == record.softTokens)
        let image = AppConversationReplayImage(tokenLowerBound: lower, tokenCount: record.softTokens,
            fileURL: directory.appendingPathComponent(record.pixelsFile), expectedDigest: record.modelInputDigest)
        #expect(try Sha256Verifier.hashFile(at: image.fileURL, chunkBytes: 65_536) == image.expectedDigest)
        let replayed = try QwenImageInputs.replay([image], tokens: tokens)
        let restored = try #require(replayed)
        let original = try #require(live.image)
        #expect(restored.pixels.shape == original.pixels.shape)
        #expect(MLX.all(restored.pixels .== original.pixels).item(Bool.self))
        try FileManager.default.removeItem(at: image.fileURL)
        #expect(throws: (any Error).self) { try QwenImageInputs.replay([image], tokens: tokens) }
    }

    @Test func retainedTokenFramingMatchesThePinnedChatTemplate() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["QWEN_TOKENIZER_TEST_DIRECTORY"])
        let tokenizer = try await QwenTokenizerLoader().load(from: URL(fileURLWithPath: path))
        let flags: [String: any Sendable] = [
            "enable_thinking": true, "reasoning_effort": "low", "preserve_thinking": true,
        ]
        let first = try tokenizer.applyChatTemplate(messages: [["role": "user", "content": "Hello"]],
            tools: nil, additionalContext: flags)
        let second = try tokenizer.applyChatTemplate(messages: [["role": "user", "content": "And you?"]],
            tools: nil, additionalContext: flags)
        let suffix = try QwenPromptPolicy.continuation(second,
            userPrefix: tokenizer.encode(text: "<|im_start|>user\n", addSpecialTokens: false))
        let body = "I can greet the user.\n</think>\n\nHello!"
        let generated = tokenizer.encode(text: body + "<|im_end|>\n", addSpecialTokens: false)
        let complete = try tokenizer.applyChatTemplate(messages: [
            ["role": "user", "content": "Hello"],
            ["role": "assistant", "reasoning_content": "I can greet the user.", "content": "Hello!"],
            ["role": "user", "content": "And you?"],
        ], tools: nil, additionalContext: flags)
        #expect(first + generated + suffix == complete)
        #expect(tokenizer.convertTokenToId("<|image_pad|>") == Int(QwenImageInputs.imageToken))
    }
}
