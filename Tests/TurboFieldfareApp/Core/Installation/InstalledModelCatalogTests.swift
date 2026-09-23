import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

@Suite struct InstalledModelCatalogTests {
    @Test func excludesIncompleteAndUnsupportedPackages() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        for name in ["unknown.mlx", "partial.gturbo", "gemma4.vision.gturbo"] {
            let directory = root.appendingPathComponent(name)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try Data("{}".utf8).write(to: directory.appendingPathComponent("config.json"))
        }
        #expect(InstalledModelCatalog.discover(in: [root]).isEmpty)
    }

    @MainActor
    @Test func switchingUpdatesFamilyAndRejectsChangesDuringLoading() throws {
        let directory = try makeCompleteModelInstall("select")
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data().write(to: directory.appendingPathComponent("model_weights.bin"))
        for layer in 0..<ArchConfig.gemma4_26B_A4B.numLayers {
            try Data().write(to: directory.appendingPathComponent(
                String(format: "packed_experts/layer_%02d.bin", layer)))
        }
        let original = directory.appendingPathComponent("missing-qwen")
        let model = AppModel(modelDirectory: original, client: MockInferenceClient(),
                             installer: QwenModelInstallerClient())
        model.loadState = .loading(.verifyingWeights)
        model.selectInstalledModel(directory)
        #expect(model.modelPathText == original.path)
        #expect(model.isQwenModel)
        model.loadState = .notLoaded
        model.selectInstalledModel(directory)
        #expect(model.modelPathText == directory.path)
        #expect(!model.isQwenModel)
        #expect(model.isModelInstalled)
        #expect(model.canLoadModel)
        #expect(try model.conversationIdentityProvider(directory)
            == ConversationIdentity.forModelDirectory(directory))
    }

    @Test func checksWeightsAndDeduplicatesLocations() throws {
        let directory = try makeCompleteModelInstall("catalog")
        defer { try? FileManager.default.removeItem(at: directory) }
        // A receipt alone must not make missing weights selectable.
        #expect(InstalledModelCatalog.installedModel(at: directory) == nil)
        try Data().write(to: directory.appendingPathComponent("model_weights.bin"))
        for layer in 0..<ArchConfig.gemma4_26B_A4B.numLayers {
            try Data().write(to: directory.appendingPathComponent(
                String(format: "packed_experts/layer_%02d.bin", layer)))
        }
        let models = InstalledModelCatalog.discover(in: [], including: [directory, directory])
        #expect(models.count == 1)
        #expect(models.first?.descriptor == .default)
    }
}
