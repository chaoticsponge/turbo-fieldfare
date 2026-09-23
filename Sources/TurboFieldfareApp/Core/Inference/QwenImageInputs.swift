import CoreGraphics
import Foundation
import TurboFieldfare
@preconcurrency import MLX
@preconcurrency import MLXLMCommon
@preconcurrency import MLXVLM

enum QwenImageInputs {
    static let imageToken: Int32 = 248_056
    static let processing = UserInput.Processing(resize: CGSize(width: 512, height: 512))

    static func userInput(prompt: String, images: [UserInput.Image]) -> UserInput {
        // The pinned prompt:images: initializer leaves its flattened images
        // array empty. The chat initializer populates both representations.
        UserInput(chat: [.user(prompt, images: images)], processing: processing,
            additionalContext: ["enable_thinking": true, "reasoning_effort": "low", "preserve_thinking": true])
    }

    static func processor(in directory: URL, tokenizer: any MLXLMCommon.Tokenizer) throws -> Qwen3VLProcessor {
        let data = try Data(contentsOf: directory.appendingPathComponent("preprocessor_config.json"))
        guard var configuration = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw AppInferenceError.invalidRequest("Invalid Qwen image processor configuration")
        }
        // The pinned Swift processor reads min_pixels/max_pixels. Qwen3.8
        // publishes the equivalent bounds under size instead.
        let size = configuration["size"] as? [String: Int] ?? [:]
        configuration["min_pixels"] = size["shortest_edge"] ?? 65_536
        configuration["max_pixels"] = min(size["longest_edge"] ?? 262_144, 262_144)
        let resolved = try JSONDecoder().decode(Qwen3VLProcessorConfiguration.self,
            from: JSONSerialization.data(withJSONObject: configuration))
        return Qwen3VLProcessor(resolved, tokenizer: tokenizer)
    }

    static func combine(_ first: LMInput.ProcessedImage?, _ second: LMInput.ProcessedImage?) -> LMInput.ProcessedImage? {
        guard let first else { return second }
        guard let second else { return first }
        return .init(pixels: concatenated([first.pixels, second.pixels]),
            frames: (first.frames ?? []) + (second.frames ?? []))
    }

    static func prepare(_ request: AppGenerationRequest, context: ModelContext,
        isolation: isolated (any Actor)? = #isolation) async throws -> LMInput {
        for image in request.imageAttachments {
            guard try Sha256Verifier.hashFile(at: image.fileURL, chunkBytes: 65_536) == image.sha256 else {
                throw AppInferenceError.invalidRequest("Image changed after attachment: \(image.displayName)")
            }
        }
        let input = userInput(prompt: request.prompt,
            images: request.imageAttachments.map { .url($0.fileURL) })
        let prepared = try await context.processor.prepare(input: input)
        guard request.imageAttachments.isEmpty || prepared.image?.frames?.count == request.imageAttachments.count else {
            throw AppInferenceError.invalidRequest("Qwen could not prepare every attached image")
        }
        return prepared
    }

    static func replay(_ images: [AppConversationReplayImage], tokens: [Int32]) throws -> LMInput.ProcessedImage? {
        if images.isEmpty {
            guard !tokens.contains(imageToken) else {
                throw AppInferenceError.conversationRestoreFailed("Qwen image input is missing")
            }
            return nil
        }
        var arrays: [MLXArray] = []
        var frames: [THW] = []
        var expectedPosition = 0
        for image in images {
            guard image.tokenLowerBound >= expectedPosition, image.tokenCount > 0,
                  image.tokenLowerBound <= tokens.count,
                  image.tokenCount <= tokens.count - image.tokenLowerBound,
                  tokens[image.tokenLowerBound..<(image.tokenLowerBound + image.tokenCount)].allSatisfy({ $0 == imageToken }),
                  try Sha256Verifier.hashFile(at: image.fileURL, chunkBytes: 65_536) == image.expectedDigest else {
                throw AppInferenceError.conversationRestoreFailed("Qwen image input does not match its recorded tokens")
            }
            let size = try image.fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
            guard size > 0, size <= 32 * 1_048_576 else {
                throw AppInferenceError.conversationRestoreFailed("Invalid Qwen image tensor size")
            }
            let stored = try MLX.loadArrays(url: image.fileURL)
            guard let pixels = stored["pixels"], let grid = stored["grid"], grid.size == 3 else {
                throw AppInferenceError.conversationRestoreFailed("Qwen image tensors are incomplete")
            }
            let values = grid.asArray(Int.self)
            guard values[0] == 1, (1...1024).contains(values[1]), (1...1024).contains(values[2]),
                  values[1] * values[2] <= 1024, values[1] * values[2] / 4 == image.tokenCount,
                  pixels.shape == [values[1] * values[2], 1536], pixels.dtype == .float32 else {
                throw AppInferenceError.conversationRestoreFailed("Qwen image grid does not match its token span")
            }
            arrays.append(pixels)
            frames.append(THW(values[0], values[1], values[2]))
            expectedPosition = image.tokenLowerBound + image.tokenCount
        }
        guard tokens.filter({ $0 == imageToken }).count == images.reduce(0, { $0 + $1.tokenCount }) else {
            throw AppInferenceError.conversationRestoreFailed("A Qwen image has no recorded tensor")
        }
        return .init(pixels: concatenated(arrays), frames: frames)
    }

    static func store(_ images: [StagedImage], modelDirectory: URL, into directory: URL) async -> TurnImageWriteOutcome {
        do {
            let tokenizer = try await QwenTokenizerLoader().load(from: modelDirectory)
            let processor = try processor(in: modelDirectory, tokenizer: tokenizer)
            let writer = try ConversationImageWriter()
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            var result = TurnImageWriteOutcome()
            for image in images {
                try Task.checkCancellation()
                guard try Sha256Verifier.hashFile(at: image.fileURL, chunkBytes: 65_536) == image.sha256 else {
                    throw AppInferenceError.invalidRequest("Image changed before it could be saved")
                }
                let (pixels, frame) = try processor.preprocess(images: [UserInput.Image.url(image.fileURL).asCIImage()], processing: processing)
                let temporary = directory.appendingPathComponent(UUID().uuidString + ".safetensors")
                defer { try? FileManager.default.removeItem(at: temporary) }
                try MLX.save(arrays: ["pixels": pixels, "grid": MLXArray([frame.t, frame.h, frame.w])], url: temporary)
                let digest = try Sha256Verifier.hashFile(at: temporary, chunkBytes: 65_536)
                let name = image.sha256 + "-" + digest + ".safetensors"
                let url = directory.appendingPathComponent(name)
                if !FileManager.default.fileExists(atPath: url.path) {
                    try FileManager.default.moveItem(at: temporary, to: url)
                } else if try Sha256Verifier.hashFile(at: url, chunkBytes: 65_536) != digest {
                    throw AppInferenceError.invalidRequest("A stored Qwen image tensor was modified")
                }
                let thumbnail = image.sha256 + ".thumb.jpg"
                try writer.writeThumbnail(from: image.fileURL, to: directory.appendingPathComponent(thumbnail))
                result.records.append(ConversationImageRecord(id: image.id, displayName: image.displayName,
                    pixelsFile: "images/" + name, thumbnailFile: "images/" + thumbnail,
                    sourceDigest: image.sha256, modelInputDigest: digest,
                    width: frame.w * 16, height: frame.h * 16, softTokens: frame.product / 4))
            }
            return result
        } catch {
            return TurnImageWriteOutcome(failed: images.count, reason: "Could not save Qwen image inputs: \(error)")
        }
    }
}
