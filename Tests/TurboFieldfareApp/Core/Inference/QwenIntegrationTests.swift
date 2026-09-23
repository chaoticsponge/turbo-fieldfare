import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

@Suite struct QwenIntegrationTests {
    @Test func continuationDoesNotRepeatSystemInstructions() throws {
        let prefix = [248_045, 42, 198]
        let tokens = [248_045, 17, 198, 99, 248_046, 198] + prefix + [123, 248_046, 198]
        #expect(try QwenPromptPolicy.continuation(tokens, userPrefix: prefix) == prefix + [123, 248_046, 198])
    }

    @Test func missingUserFramingIsRefused() {
        #expect(throws: AppInferenceError.self) {
            try QwenPromptPolicy.continuation([1, 2, 3], userPrefix: [4, 5])
        }
        #expect(throws: AppInferenceError.self) {
            try QwenPromptPolicy.continuation([], userPrefix: [])
        }
    }

    @Test func catalogPinsEveryPayloadAndMatchesDownloadSize() throws {
        let catalog = try QwenModelPackage.catalog()
        #expect(catalog.repoID == QwenModelPackage.repoID)
        #expect(catalog.revision == QwenModelPackage.revision)
        #expect(catalog.files.reduce(UInt64(0)) { $0 + $1.bytes } == QwenModelPackage.descriptor.installedBytes)
        #expect(Set(catalog.files.map(\.name)).count == catalog.files.count)
        for file in catalog.files {
            #expect(file.bytes > 0)
            #expect(file.sha256.count == 64)
            #expect(file.sha256.allSatisfy { $0.isHexDigit && !$0.isUppercase })
            #expect(!file.name.contains("/") && !file.name.contains(".."))
        }
    }

    @Test func verificationRejectsChangedPayloadAndSymlinks() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("payload")
        try Data("good".utf8).write(to: file)
        let expected = QwenModelPackage.File(name: "payload", bytes: 4,
            sha256: try Sha256Verifier.hashFile(at: file, chunkBytes: 64))
        #expect(try QwenModelPackage.matches(expected, at: file, hashes: true))
        try Data("evil".utf8).write(to: file)
        #expect(try !QwenModelPackage.matches(expected, at: file, hashes: true))
        let link = directory.appendingPathComponent("link")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: file)
        #expect(try !QwenModelPackage.matches(expected, at: link, hashes: false))
        let partial = directory.appendingPathComponent("partial")
        try FileManager.default.createDirectory(at: partial, withIntermediateDirectories: false)
        let catalog = try QwenModelPackage.catalog()
        try JSONEncoder().encode(catalog).write(to: partial.appendingPathComponent(QwenModelPackage.downloadMarkerName))
        let directoryLink = directory.appendingPathComponent("directory-link")
        try FileManager.default.createSymbolicLink(at: directoryLink, withDestinationURL: partial)
        #expect(throws: (any Error).self) {
            try QwenModelPackage.validateDownloadMarker(in: directoryLink, catalog: catalog)
        }
    }

    @Test func receiptAloneCannotMakeMissingWeightsLoadable() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try JSONEncoder().encode(QwenModelPackage.catalog()).write(to:
            directory.appendingPathComponent(QwenModelPackage.receiptName))
        #expect(throws: (any Error).self) { try QwenModelPackage.validate(at: directory) }
        if case .partial = AppModelInstallationProbe.status(at: directory) {} else {
            Issue.record("A receipt without model weights must not be installed")
        }
        if case .partial = AppVisionPackInstallationProbe.status(at: directory) {} else {
            Issue.record("An incomplete Qwen model must not advertise image support")
        }
    }
}
