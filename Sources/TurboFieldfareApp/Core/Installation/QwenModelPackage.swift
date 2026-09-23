import Foundation
import TurboFieldfare
import TurboFieldfareRepackCore

/// Qwen uses MLX safetensors, not the Gemma-specific gturbo wire format.
public enum QwenModelPackage {
    public static let repoID = "leonsarmiento/Qwen3.8-27B-3bit-mlx"
    public static let revision = "5fc234d9e6080b8388a11286380e801b7c9f535c"
    public static let receiptName = "qwen-install.json"
    static let downloadMarkerName = "qwen-download.json"

    public static func defaultDirectory(for variant: QwenModelVariant = .default) -> URL {
        AppModelLocation.defaultURL().deletingLastPathComponent()
            .appendingPathComponent(variant.directoryName, isDirectory: true)
    }

    static func requireMetalLibrary() throws {
        var candidates: [URL] = []
        if let executable = Bundle.main.executableURL {
            candidates.append(executable.deletingLastPathComponent().appendingPathComponent("mlx.metallib"))
        }
        for bundle in Bundle.allBundles + Bundle.allFrameworks {
            if let resources = bundle.resourceURL {
                candidates.append(resources.appendingPathComponent("default.metallib"))
            }
        }
        guard candidates.contains(where: { FileManager.default.fileExists(atPath: $0.path) }) else {
            throw AppInferenceError.modelLoadFailed("MLX GPU shaders are missing. Build with bash Scripts/build-qwen.sh, then launch the rebuilt app.")
        }
    }

    struct File: Codable, Equatable, Sendable {
        let name: String
        let bytes: UInt64
        let sha256: String
    }
    struct Catalog: Codable, Equatable, Sendable {
        let repoID: String
        let revision: String
        let files: [File]
    }

    static func catalog(for variant: QwenModelVariant = .default) throws -> Catalog {
        guard let url = Bundle.module.url(forResource: variant.resourceName, withExtension: "json") else {
            throw AppInferenceError.modelLoadFailed("Qwen download catalog is missing")
        }
        return try JSONDecoder().decode(Catalog.self, from: Data(contentsOf: url))
    }

    public static let descriptor = AppModelInstallDescriptor(
        displayName: "Qwen3.8 27B MLX 3-bit", repoID: repoID, revision: revision,
        sourceIndexSHA256: "", approximateDownloadBytes: 12_729_681_276,
        installedBytes: 12_729_681_276, rangeStagingBytes: 0,
        reserveBytes: 1_073_741_824)

    /// Reject a package whose weights alone exceed Metal's recommended budget.
    /// Passing this check is not a promise that every context/image workload fits.
    static func validateWeightBudget(for variant: QwenModelVariant,
                                     recommendedWorkingSetBytes: Int?) throws {
        guard let budget = recommendedWorkingSetBytes, budget > 0 else { return }
        let weights = try catalog(for: variant).files
            .filter { $0.name.hasSuffix(".safetensors") }
            .reduce(UInt64(0)) { $0 + $1.bytes }
        guard weights < UInt64(budget) else {
            let size = ByteCountFormatter.string(fromByteCount: Int64(weights), countStyle: .memory)
            throw AppInferenceError.modelLoadFailed(
                "This Qwen package needs approximately \(size) for weights alone, exceeding this Mac's recommended GPU memory budget. Choose a smaller quantization or use a Mac with more unified memory.")
        }
    }

    public static func isQwen(at directory: URL) -> Bool {
        FileManager.default.fileExists(atPath: directory.appendingPathComponent(receiptName).path)
    }

    static func installedCatalog(at directory: URL) throws -> Catalog {
        let receipt = try JSONDecoder().decode(Catalog.self, from: Data(contentsOf:
            directory.appendingPathComponent(receiptName)))
        guard let variant = QwenModelVariant.matching(repoID: receipt.repoID),
              receipt == (try catalog(for: variant)) else {
            throw AppInferenceError.modelLoadFailed("Qwen installation does not match a supported pinned checkpoint")
        }
        return receipt
    }

    public static func variant(at directory: URL) throws -> QwenModelVariant {
        let receipt = try installedCatalog(at: directory)
        guard let variant = QwenModelVariant.matching(repoID: receipt.repoID) else {
            throw AppInferenceError.modelLoadFailed("Unsupported Qwen installation")
        }
        return variant
    }

    /// Cheap ingress check; loading verifies the hashes before MLX sees weights.
    public static func validate(at directory: URL, hashes: Bool = false) throws {
        let directoryValues = try directory.resourceValues(forKeys: [.isSymbolicLinkKey, .isDirectoryKey])
        guard directoryValues.isSymbolicLink != true, directoryValues.isDirectory == true else {
            throw AppInferenceError.modelLoadFailed("Qwen model location must be a real directory")
        }
        let expected = try installedCatalog(at: directory)
        for file in expected.files {
            try Task.checkCancellation()
            let url = directory.appendingPathComponent(file.name)
            let values = try url.resourceValues(forKeys: [.isSymbolicLinkKey, .isRegularFileKey, .fileSizeKey])
            guard values.isSymbolicLink != true, values.isRegularFile == true,
                  UInt64(values.fileSize ?? 0) == file.bytes else {
                throw AppInferenceError.modelLoadFailed("Missing or incomplete Qwen file: \(file.name)")
            }
            if hashes, try Sha256Verifier.hashFile(at: url, chunkBytes: 1_048_576) != file.sha256 {
                throw AppInferenceError.modelLoadFailed("Qwen checksum mismatch: \(file.name)")
            }
        }
    }

    static func matches(_ file: File, at url: URL, hashes: Bool) throws -> Bool {
        let values = try url.resourceValues(forKeys: [.isSymbolicLinkKey, .isRegularFileKey, .fileSizeKey])
        guard values.isSymbolicLink != true, values.isRegularFile == true,
              UInt64(values.fileSize ?? 0) == file.bytes else { return false }
        guard hashes else { return true }
        return try Sha256Verifier.hashFile(at: url, chunkBytes: 1_048_576) == file.sha256
    }

    static func validateDownloadMarker(in directory: URL, catalog: Catalog) throws {
        let values = try directory.resourceValues(forKeys: [.isSymbolicLinkKey, .isDirectoryKey])
        guard values.isSymbolicLink != true, values.isDirectory == true else {
            throw AppInferenceError.modelLoadFailed("Partial Qwen download must be a real directory")
        }
        let marker = directory.appendingPathComponent(downloadMarkerName)
        let contents = try JSONDecoder().decode(Catalog.self, from: Data(contentsOf: marker))
        guard contents == catalog else {
            throw AppInferenceError.modelLoadFailed("Partial download belongs to another checkpoint")
        }
    }

    public static func identity(at directory: URL) throws -> ConversationIdentity {
        try validate(at: directory)
        let catalog = try installedCatalog(at: directory)
        return ConversationIdentity(modelID: catalog.repoID, sourceSnapshotHash: catalog.revision,
            templateIdentity: try variant(at: directory).templateIdentity,
            imageProcessingVersion: 38)
    }
}

/// Downloads directly into one installation, retaining verified complete files
/// after cancellation. The receipt is published only after every hash passes.
public final class QwenModelInstallerClient: AppModelInstallerClient, Sendable {
    public let descriptor: AppModelInstallDescriptor
    private let runner: RepackModelInstallerClient

    public convenience init() {
        self.init(variant: .default, descriptor: QwenModelPackage.descriptor)
    }

    public convenience init(variant: QwenModelVariant) throws {
        guard let descriptor = variant.installDescriptor else {
            throw AppInferenceError.modelLoadFailed("Qwen download catalog is missing or invalid")
        }
        self.init(variant: variant, descriptor: descriptor)
    }

    private init(variant: QwenModelVariant, descriptor: AppModelInstallDescriptor) {
        self.descriptor = descriptor
        runner = RepackModelInstallerClient(descriptor: descriptor,
            runInstall: { directory, progress in
                let lock = try InstallLock.acquire(outputDirectory: directory.path)
                defer { withExtendedLifetime(lock) {} }
                let fm = FileManager.default
                let catalog = try QwenModelPackage.catalog(for: variant)
                let partial = URL(fileURLWithPath: lock.paths.partialDirectory, isDirectory: true)
                if QwenModelPackage.isQwen(at: directory) {
                    let values = try directory.resourceValues(forKeys: [.isSymbolicLinkKey, .isDirectoryKey])
                    guard values.isSymbolicLink != true, values.isDirectory == true else {
                        throw AppInferenceError.modelLoadFailed("Qwen installation must be a real directory")
                    }
                    guard try QwenModelPackage.variant(at: directory) == variant else {
                        throw AppInferenceError.modelLoadFailed("This location belongs to a different Qwen model")
                    }
                    if (try? QwenModelPackage.validate(at: directory, hashes: true)) != nil { return directory }
                    // Repair only our own receipt-bound installation, preserving
                    // good files. Never replace an unrelated model directory.
                    let receipt = try JSONDecoder().decode(QwenModelPackage.Catalog.self,
                        from: Data(contentsOf: directory.appendingPathComponent(QwenModelPackage.receiptName)))
                    guard receipt == catalog, !fm.fileExists(atPath: partial.path) else {
                        throw AppInferenceError.modelLoadFailed("Cannot repair this location while a separate partial download exists")
                    }
                    try JSONEncoder().encode(catalog).write(to: directory.appendingPathComponent(QwenModelPackage.downloadMarkerName), options: .atomic)
                    try fm.moveItem(at: directory, to: partial)
                }
                guard !fm.fileExists(atPath: directory.path) else {
                    throw AppInferenceError.modelLoadFailed("Choose an empty Qwen model location; this directory already exists")
                }
                if fm.fileExists(atPath: partial.path) {
                    try QwenModelPackage.validateDownloadMarker(in: partial, catalog: catalog)
                } else {
                    try fm.createDirectory(at: partial, withIntermediateDirectories: true)
                    try JSONEncoder().encode(catalog).write(to: partial.appendingPathComponent(QwenModelPackage.downloadMarkerName), options: .atomic)
                }
                let total = catalog.files.reduce(UInt64(0)) { $0 + $1.bytes }
                var reused: UInt64 = 0
                var downloaded: UInt64 = 0
                for file in catalog.files {
                    try Task.checkCancellation()
                    let target = partial.appendingPathComponent(file.name)
                    if fm.fileExists(atPath: target.path),
                       try QwenModelPackage.matches(file, at: target, hashes: true) {
                        reused += file.bytes
                    } else {
                        let url = URL(string: "https://huggingface.co/\(catalog.repoID)/resolve/\(catalog.revision)/\(file.name)")!
                        let priorReused = reused
                        let priorDownloaded = downloaded
                        let temporary = partial.appendingPathComponent("." + UUID().uuidString + ".download")
                        defer { try? fm.removeItem(at: temporary) }
                        try await QwenFileDownload.download(from: url, to: temporary) { bytes in
                            progress(.copyingPayload(reusedBytes: priorReused,
                                downloadedThisRunBytes: priorDownloaded + min(bytes, file.bytes), totalBytes: total))
                        }
                        progress(.hashingOutput(file.name))
                        guard try Sha256Verifier.hashFile(at: temporary, chunkBytes: 1_048_576) == file.sha256 else {
                            throw AppInferenceError.modelLoadFailed("Download checksum mismatch for \(file.name)")
                        }
                        if fm.fileExists(atPath: target.path) { try fm.removeItem(at: target) }
                        try fm.moveItem(at: temporary, to: target)
                        downloaded += file.bytes
                    }
                    progress(.copyingPayload(reusedBytes: reused, downloadedThisRunBytes: downloaded, totalBytes: total))
                }
                progress(.finalizing)
                try JSONEncoder().encode(catalog).write(to: partial.appendingPathComponent(QwenModelPackage.receiptName), options: .atomic)
                try QwenModelPackage.validate(at: partial)
                try Task.checkCancellation()
                try fm.moveItem(at: partial, to: directory)
                return directory
            }, runDiscard: { directory in
                let lock = try InstallLock.acquire(outputDirectory: directory.path)
                defer { withExtendedLifetime(lock) {} }
                let partial = URL(fileURLWithPath: lock.paths.partialDirectory)
                if FileManager.default.fileExists(atPath: partial.path) {
                    try QwenModelPackage.validateDownloadMarker(in: partial, catalog: QwenModelPackage.catalog(for: variant))
                    try FileManager.default.removeItem(at: partial)
                }
            })
    }

    public func checkInstallRequirement(outputDirectory: URL) throws -> AppModelInstallRequirement {
        let assessment = try DiskSpaceChecker.assess(path: outputDirectory.path,
            bytes: descriptor.installedBytes, reserveBytes: descriptor.reserveBytes)
        return AppModelInstallRequirement(probePath: assessment.path,
            requiredBytes: assessment.requiredBytes, availableBytes: assessment.availableBytes)
    }
    public func installDefaultModel(outputDirectory: URL) -> AsyncThrowingStream<AppModelInstallEvent, Error> {
        runner.installDefaultModel(outputDirectory: outputDirectory)
    }
    public func cancel() { runner.cancel() }
    public func discardPartialInstall(outputDirectory: URL) async throws {
        try await runner.discardPartialInstall(outputDirectory: outputDirectory)
    }
}
