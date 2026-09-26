import Foundation
import Testing
@testable import TurboFieldfareRepackCore

@Suite struct ArchInfoValidationTests {
    private func check(_ change: (inout [String: Any]) -> Void) throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        _ = try SyntheticSnapshot.build(at: root.path)
        let file = root.appendingPathComponent("config.json")
        var config = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any])
        var text = try #require(config["text_config"] as? [String: Any])
        change(&text)
        config["text_config"] = text
        try JSONSerialization.data(withJSONObject: config).write(to: file)
        #expect(throws: (any Error).self) { try ArchInfo.load(configPath: file.path) }
    }

    @Test func rejectsInvalidDimensionsWithoutTrapping() throws {
        for value: Any in [-1, 0, 1.5, true, "2", 1e30] {
            try check { $0["num_hidden_layers"] = value }
        }
        try check { $0["num_experts"] = 0 }
        try check { $0["num_hidden_layers"] = 257 }
        try check { $0["num_experts"] = 1025 }
    }

    @Test func rejectsInconsistentArchitectureAndRope() throws {
        try check { $0["top_k_experts"] = 3 }
        try check { $0["num_key_value_heads"] = 3 }
        try check { $0["layer_types"] = ["unknown", "full_attention"] }
        try check { $0["layer_types"] = ["full_attention"] }
        try check { $0["rope_parameters"] = ["full_attention": ["partial_rotary_factor": 2]] }
        try check { $0["final_logit_softcapping"] = -1 }
    }

    @Test func validSyntheticArchitectureStillPlans() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        _ = try SyntheticSnapshot.build(at: root.path)
        let arch = try ArchInfo.load(configPath: root.appendingPathComponent("config.json").path)
        try arch.validate()
        #expect(arch.numLayers == 2)
        let meta = try IndexLoader.load(snapshotDir: root.path)
        let plan = try RepackPlanner.plan(meta: meta, arch: arch, shardHeaders: [], outputDir: "/unused")
        #expect(plan.layers.count == 2)
    }
}
