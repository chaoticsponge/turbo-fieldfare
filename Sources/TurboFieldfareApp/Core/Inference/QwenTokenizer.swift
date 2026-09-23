import Foundation
import Tokenizers
import MLXLMCommon

struct QwenTokenizerLoader: MLXLMCommon.TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let templateURL = directory.appendingPathComponent("chat_template.jinja")
        let template: String
        if FileManager.default.fileExists(atPath: templateURL.path) {
            template = try String(contentsOf: templateURL, encoding: .utf8)
        } else {
            let data = try Data(contentsOf: directory.appendingPathComponent("tokenizer_config.json"))
            guard let config = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let embedded = config["chat_template"] as? String, !embedded.isEmpty else {
                throw AppInferenceError.modelLoadFailed("Qwen chat template is missing")
            }
            template = embedded
        }
        // Qwen3 and Qwen3.5 normally removes older reasoning when rerendering a chat.
        // This app retains its exact token lineage, so preserve it consistently
        // for both first turns and replay. The pinned Qwen3.8 template already
        // implements preserve_thinking and is left intact.
        let retainedTemplate = template.replacingOccurrences(
            of: "{%- if loop.index0 > ns.last_query_index %}",
            with: "{%- if loop.index0 > ns.last_query_index or preserve_thinking is defined and preserve_thinking %}")
        return QwenTokenizer(base: try await AutoTokenizer.from(modelFolder: directory),
                             chatTemplate: retainedTemplate)
    }
}

struct QwenTokenizer: MLXLMCommon.Tokenizer {
    let base: any Tokenizers.Tokenizer
    var chatTemplate: String? = nil
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        base.encode(text: text, addSpecialTokens: addSpecialTokens)
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        base.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }
    func convertTokenToId(_ token: String) -> Int? { base.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { base.convertIdToToken(id) }
    var bosToken: String? { base.bosToken }
    var eosToken: String? { base.eosToken }
    var unknownToken: String? { base.unknownToken }
    func applyChatTemplate(messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?, additionalContext: [String: any Sendable]?) throws -> [Int] {
        try base.applyChatTemplate(messages: messages,
            chatTemplate: chatTemplate.map { .literal($0) }, addGenerationPrompt: true,
            truncation: false, maxLength: nil, tools: tools, additionalContext: additionalContext)
    }
}

/// The first turn carries the template's system instructions. Later turns
/// append only their user/assistant framing to the retained token lineage.
enum QwenPromptPolicy {
    static func continuation(_ tokens: [Int], userPrefix: [Int]) throws -> [Int] {
        guard !userPrefix.isEmpty, tokens.count >= userPrefix.count else {
            throw AppInferenceError.invalidRequest("Qwen template has no user turn")
        }
        for offset in 0...(tokens.count - userPrefix.count) {
            if tokens[offset..<(offset + userPrefix.count)].elementsEqual(userPrefix) {
                return Array(tokens[offset...])
            }
        }
        throw AppInferenceError.invalidRequest("Qwen template has no user turn")
    }
}
