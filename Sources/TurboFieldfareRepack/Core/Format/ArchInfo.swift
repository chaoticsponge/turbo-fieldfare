import Foundation
import CoreFoundation

/// Architecture facts mirrored into `manifest.json -> arch`. Cross-checked by
/// the runtime loader at startup.
struct ArchInfo: Sendable, Equatable {
    let hiddenSize: Int
    let intermediateSize: Int          // shared expert FFN
    let moeIntermediateSize: Int       // per-expert FFN
    let numHeads: Int
    let numKVHeads: Int
    let numFullKVHeads: Int
    let headDim: Int
    let fullHeadDim: Int
    let vocabSize: Int
    let slidingWindow: Int
    let finalLogitSoftcap: Double
    let ropeTheta: Double
    let fullRopeTheta: Double
    let partialRotaryFactor: Double
    let numLayers: Int
    let numExperts: Int
    let topKExperts: Int
    let tieWordEmbeddings: Bool
    let attentionKEqV: Bool
    /// 1 if `full_attention`, 0 if `sliding_attention`. Indexed by layer.
    let fullAttentionLayerMask: [UInt8]
    let hiddenActivation: String

    static func load(configPath: String) throws -> ArchInfo {
        let data = try Posix.readBoundedData(configPath, maximumBytes: 1024 * 1024)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let tc = root["text_config"] as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "no text_config")
        }
        func number(_ value: Any?, _ key: String) throws -> Double {
            guard let n = value as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID(),
                  n.doubleValue.isFinite else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "invalid number \(key)")
            }
            return n.doubleValue
        }
        func i(_ key: String) throws -> Int {
            let value = try number(tc[key], key)
            guard value.rounded() == value, value > 0, value <= 1_048_576 else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "invalid dimension \(key)")
            }
            return Int(value)
        }
        func d(_ key: String) throws -> Double { try number(tc[key], key) }
        let layerTypes = (tc["layer_types"] as? [String]) ?? []
        let mask = layerTypes.map { UInt8($0 == "full_attention" ? 1 : 0) }
        let rope = (tc["rope_parameters"] as? [String: Any]) ?? [:]
        let ropeFull = (rope["full_attention"] as? [String: Any]) ?? [:]
        let ropeSWA  = (rope["sliding_attention"] as? [String: Any]) ?? [:]
        let prf = try number(ropeFull["partial_rotary_factor"] ?? 0.25, "partial_rotary_factor")
        let fullTheta = try number(ropeFull["rope_theta"] ?? 1_000_000.0, "full rope_theta")
        let swaTheta = try number(ropeSWA["rope_theta"] ?? 10_000.0, "sliding rope_theta")
        guard layerTypes.allSatisfy({ ["full_attention", "sliding_attention"].contains($0) }) else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "unsupported layer type")
        }
        let kEqV = (tc["attention_k_eq_v"] as? Bool) ?? false
        let tie = (tc["tie_word_embeddings"] as? Bool) ?? false
        let act = (tc["hidden_activation"] as? String) ?? "gelu_pytorch_tanh"
        let arch = ArchInfo(
            hiddenSize: try i("hidden_size"),
            intermediateSize: try i("intermediate_size"),
            moeIntermediateSize: try i("moe_intermediate_size"),
            numHeads: try i("num_attention_heads"),
            numKVHeads: try i("num_key_value_heads"),
            numFullKVHeads: try i("num_global_key_value_heads"),
            headDim: try i("head_dim"),
            fullHeadDim: try i("global_head_dim"),
            vocabSize: try i("vocab_size"),
            slidingWindow: try i("sliding_window"),
            finalLogitSoftcap: try d("final_logit_softcapping"),
            ropeTheta: swaTheta,
            fullRopeTheta: fullTheta,
            partialRotaryFactor: prf,
            numLayers: try i("num_hidden_layers"),
            numExperts: try i("num_experts"),
            topKExperts: try i("top_k_experts"),
            tieWordEmbeddings: tie,
            attentionKEqV: kEqV,
            fullAttentionLayerMask: mask,
            hiddenActivation: act)
        try arch.validate()
        return arch
    }

    /// Also called by the planner: programmatic inputs must obey the same bounds.
    func validate() throws {
        let dimensions = [hiddenSize, intermediateSize, moeIntermediateSize]
        guard dimensions.allSatisfy({ (1...65_536).contains($0) }),
              (1...256).contains(numLayers), (1...1024).contains(numExperts),
              (1...numExperts).contains(topKExperts),
              (1...512).contains(numHeads), (1...numHeads).contains(numKVHeads),
              (1...numHeads).contains(numFullKVHeads),
              numHeads % numKVHeads == 0, numHeads % numFullKVHeads == 0,
              (1...4096).contains(headDim), (1...4096).contains(fullHeadDim),
              (1...1_048_576).contains(vocabSize), (1...1_048_576).contains(slidingWindow),
              fullAttentionLayerMask.count == numLayers,
              fullAttentionLayerMask.allSatisfy({ $0 <= 1 }),
              finalLogitSoftcap.isFinite, finalLogitSoftcap > 0,
              ropeTheta.isFinite, ropeTheta > 0, fullRopeTheta.isFinite, fullRopeTheta > 0,
              partialRotaryFactor.isFinite, partialRotaryFactor > 0, partialRotaryFactor <= 1 else {
            throw RepackError.configurationInvalid(detail: "invalid or unsupported architecture bounds")
        }
    }
}
