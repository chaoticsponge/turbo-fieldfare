import Foundation

/// Only packages checked against the pinned MLX Qwen implementations belong here.
public enum QwenModelVariant: String, CaseIterable, Identifiable, Sendable {
    case small4B, small9B, text14B, original27B, quality27B, text32B

    public var id: String { rawValue }
    public static let `default`: Self = .original27B

    var resourceName: String {
        switch self {
        case .small4B: "qwen-3.5-4b"
        case .small9B: "qwen-3.5-9b"
        case .text14B: "qwen-3-14b"
        case .text32B: "qwen-3-32b"
        case .original27B: "qwen-model"
        case .quality27B: "qwen-3.8-27b-8bit"
        }
    }

    public var directoryName: String {
        switch self {
        case .small4B: "qwen3.5-4b-4bit.mlx"
        case .small9B: "qwen3.5-9b-4bit.mlx"
        case .text14B: "qwen3-14b-4bit.mlx"
        case .text32B: "qwen3-32b-4bit.mlx"
        case .original27B: "qwen3.8-27b.mlx"
        case .quality27B: "qwen3.8-27b-8bit.mlx"
        }
    }

    public var displayName: String {
        switch self {
        case .small4B: "Qwen3.5 4B MLX 4-bit"
        case .small9B: "Qwen3.5 9B MLX 4-bit"
        case .text14B: "Qwen3 14B MLX 4-bit (text only)"
        case .text32B: "Qwen3 32B MLX 4-bit (text only)"
        case .original27B: "Qwen3.8 27B MLX 3-bit"
        case .quality27B: "Qwen3.8 27B MLX 8-bit (64 GB Mac)"
        }
    }

    public var repoID: String {
        switch self {
        case .small4B: "mlx-community/Qwen3.5-4B-4bit"
        case .small9B: "mlx-community/Qwen3.5-9B-4bit"
        case .text14B: "mlx-community/Qwen3-14B-4bit"
        case .text32B: "mlx-community/Qwen3-32B-4bit"
        case .original27B: QwenModelPackage.repoID
        case .quality27B: "mlx-community/Qwen3.8-27B-8bit"
        }
    }

    public var supportsImages: Bool {
        switch self {
        case .text14B, .text32B: false
        case .small4B, .small9B, .original27B, .quality27B: true
        }
    }

    var vocabularySize: Int { supportsImages ? 248_320 : 151_936 }

    var templateIdentity: String {
        switch self {
        case .original27B, .quality27B: "qwen3.8-mlx-3.31.3-preserve-thinking-v1"
        case .small4B, .small9B: "qwen3.5-mlx-3.31.3-preserve-thinking-v1"
        case .text14B, .text32B: "qwen3-mlx-3.31.3-preserve-thinking-v1"
        }
    }

    func validateImageCount(_ count: Int) throws {
        guard supportsImages || count == 0 else {
            throw AppInferenceError.invalidRequest("Image support is unavailable for this text-only Qwen model")
        }
    }

    public static func matching(repoID: String) -> Self? {
        allCases.first { $0.repoID == repoID }
    }

    /// An unreadable bundled catalog is unavailable for installation.
    public var installDescriptor: AppModelInstallDescriptor? {
        guard let catalog = try? QwenModelPackage.catalog(for: self) else { return nil }
        let bytes = catalog.files.reduce(UInt64(0)) { $0 + $1.bytes }
        return AppModelInstallDescriptor(displayName: displayName, repoID: repoID,
            revision: catalog.revision, sourceIndexSHA256: "", approximateDownloadBytes: bytes,
            installedBytes: bytes, rangeStagingBytes: 0, reserveBytes: 1_073_741_824)
    }
}
