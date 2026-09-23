import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

@Suite struct QwenModelVariantTests {
    @Test func catalogsPinDistinctCompletePackages() throws {
        #expect(QwenModelVariant.allCases.count == 6)
        #expect(Set(QwenModelVariant.allCases.map(\.directoryName)).count == 6)
        for variant in QwenModelVariant.allCases {
            let catalog = try QwenModelPackage.catalog(for: variant)
            let descriptor = try #require(variant.installDescriptor)
            #expect(catalog.repoID == variant.repoID)
            #expect(catalog.revision.count == 40 && catalog.revision.allSatisfy { $0.isHexDigit })
            #expect(descriptor.installedBytes == catalog.files.reduce(0) { $0 + $1.bytes })
            let names = Set(catalog.files.map(\.name))
            #expect(names.count == catalog.files.count)
            #expect(names.isSuperset(of: ["config.json", "tokenizer.json", "tokenizer_config.json",
                                          "model.safetensors.index.json"]))
            if variant.supportsImages {
                #expect(names.isSuperset(of: ["chat_template.jinja", "preprocessor_config.json"]))
            }
            #expect(names.contains { $0.hasSuffix(".safetensors") })
            for file in catalog.files {
                #expect(file.bytes > 0 && file.sha256.count == 64)
                #expect(file.sha256.allSatisfy { $0.isHexDigit })
                #expect(!file.name.contains("/") && !file.name.contains(".."))
            }
            let installer = try QwenModelInstallerClient(variant: variant)
            #expect(installer.descriptor == descriptor)
        }
    }

    @Test func eightBitPackageKeepsQuantizationsAndReceiptsSeparate() throws {
        let original = try QwenModelPackage.catalog(for: .original27B)
        let quality = try QwenModelPackage.catalog(for: .quality27B)
        #expect(QwenModelVariant.default == .original27B)
        #expect(quality.repoID == "mlx-community/Qwen3.8-27B-8bit")
        #expect(QwenModelVariant.quality27B.supportsImages)
        #expect(QwenModelVariant.quality27B.vocabularySize == 248_320)
        #expect(QwenModelPackage.defaultDirectory(for: .quality27B)
            != QwenModelPackage.defaultDirectory(for: .original27B))
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try JSONEncoder().encode(quality).write(to: root.appendingPathComponent(QwenModelPackage.receiptName))
        #expect(try QwenModelPackage.variant(at: root) == .quality27B)
        try JSONEncoder().encode(original).write(to: root.appendingPathComponent(QwenModelPackage.downloadMarkerName))
        #expect(throws: (any Error).self) {
            try QwenModelPackage.validateDownloadMarker(in: root, catalog: quality)
        }
    }

    @Test func oversizedWeightsAreRejectedBeforeAllocatingTheModel() throws {
        #expect(throws: AppInferenceError.self) {
            try QwenModelPackage.validateWeightBudget(for: .quality27B,
                recommendedWorkingSetBytes: 14 * 1_073_741_824)
        }
        try QwenModelPackage.validateWeightBudget(for: .quality27B,
            recommendedWorkingSetBytes: 48 * 1_073_741_824)
        try QwenModelPackage.validateWeightBudget(for: .original27B,
            recommendedWorkingSetBytes: 14 * 1_073_741_824)
        // A backend that cannot report its budget must not report a false OOM.
        try QwenModelPackage.validateWeightBudget(for: .quality27B,
            recommendedWorkingSetBytes: nil)
    }

    @MainActor
    @Test(arguments: [QwenModelVariant.text14B, .text32B])
    func textOnlyModelsCannotAdvertiseOrAcceptImages(_ variant: QwenModelVariant) throws {
        #expect(!variant.supportsImages)
        #expect(variant.vocabularySize == 151_936)
        try variant.validateImageCount(0)
        #expect(throws: AppInferenceError.self) { try variant.validateImageCount(1) }
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try JSONEncoder().encode(QwenModelPackage.catalog(for: variant)).write(
            to: directory.appendingPathComponent(QwenModelPackage.receiptName))
        #expect(AppVisionPackInstallationProbe.status(at: directory) == .unsupportedLayout)
        let model = AppModel(modelDirectory: directory, client: MockInferenceClient(),
                             installer: try QwenModelInstallerClient(variant: variant))
        #expect(model.isTextOnlyQwenModel)
        #expect(!model.isImageInputAvailable)
        #expect(!model.canInstallVisionPack)
    }

    @Test func receiptsAndPausedDownloadsCannotCrossModelBoundaries() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let four = try QwenModelPackage.catalog(for: .small4B)
        let nine = try QwenModelPackage.catalog(for: .small9B)
        try JSONEncoder().encode(four).write(to: root.appendingPathComponent(QwenModelPackage.receiptName))
        #expect(try QwenModelPackage.variant(at: root) == .small4B)
        // Valid metadata without weights must still be unavailable to load.
        #expect(throws: (any Error).self) { try QwenModelPackage.validate(at: root) }
        try JSONEncoder().encode(four).write(to: root.appendingPathComponent(QwenModelPackage.downloadMarkerName))
        try QwenModelPackage.validateDownloadMarker(in: root, catalog: four)
        #expect(throws: (any Error).self) {
            try QwenModelPackage.validateDownloadMarker(in: root, catalog: nine)
        }
        let changed = QwenModelPackage.Catalog(repoID: four.repoID, revision: nine.revision, files: four.files)
        try JSONEncoder().encode(changed).write(to: root.appendingPathComponent(QwenModelPackage.receiptName))
        #expect(throws: (any Error).self) { try QwenModelPackage.variant(at: root) }
    }

    @MainActor
    @Test func choosingDownloadUpdatesSizePathAndFamilyWithoutStartingInstall() throws {
        let model = AppModel(modelDirectory: URL(fileURLWithPath: "/tmp/model-choice-\(UUID())"),
                             client: MockInferenceClient())
        model.selectQwenModelForInstallation(.small4B)
        #expect(model.isQwenModel)
        #expect(model.modelPathText == QwenModelPackage.defaultDirectory(for: .small4B).path)
        #expect(model.installDescriptor == QwenModelVariant.small4B.installDescriptor)
        #expect(model.installState == .idle)
        model.selectQwenModelForInstallation(.quality27B)
        #expect(model.modelPathText == QwenModelPackage.defaultDirectory(for: .quality27B).path)
        #expect(model.installDescriptor == QwenModelVariant.quality27B.installDescriptor)
        #expect(model.installState == .idle)
        model.selectQwenModelForInstallation(.small4B)
        model.loadState = .loading(.verifyingWeights)
        model.selectQwenModelForInstallation(.small9B)
        #expect(model.installDescriptor.repoID == QwenModelVariant.small4B.repoID)
        model.loadState = .notLoaded
        model.selectQwenModelForInstallation(.small9B)
        #expect(model.installDescriptor == QwenModelVariant.small9B.installDescriptor)
        #expect(model.installState == .idle)
    }
}
