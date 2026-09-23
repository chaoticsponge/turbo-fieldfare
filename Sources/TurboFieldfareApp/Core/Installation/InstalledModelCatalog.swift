import Foundation
import TurboFieldfare

public struct InstalledModel: Identifiable, Equatable, Sendable {
    public var id: String { directory.path }
    public let directory: URL
    public let descriptor: AppModelInstallDescriptor
}

/// Metadata-only discovery. Full weight verification remains part of loading.
public enum InstalledModelCatalog {
    public static func searchDirectories(current: URL) -> [URL] {
        let support = FileManager.default.urls(for: .applicationSupportDirectory,
                                               in: .userDomainMask).first
        return [current.deletingLastPathComponent(),
                AppModelLocation.defaultURL().deletingLastPathComponent(),
                support?.appendingPathComponent("TurboFieldfare", isDirectory: true)]
            .compactMap { $0 }
    }

    public static func discover(in roots: [URL], including: [URL] = []) -> [InstalledModel] {
        var candidates = including
        for root in roots {
            candidates += (try? FileManager.default.contentsOfDirectory(
                at: root, includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles])) ?? []
        }
        var seen: Set<String> = []
        return candidates.compactMap { candidate in
            let directory = candidate.standardizedFileURL
            guard let installed = installedModel(at: directory),
                  seen.insert(directory.resolvingSymlinksInPath().path).inserted else { return nil }
            return installed
        }.sorted { $0.descriptor.displayName == $1.descriptor.displayName
            ? $0.id < $1.id : $0.descriptor.displayName < $1.descriptor.displayName }
    }

    public static func installedModel(at directory: URL) -> InstalledModel? {
        let directory = directory.standardizedFileURL
        guard let values = try? directory.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey]),
              values.isDirectory == true, values.isSymbolicLink != true,
              AppModelInstallationProbe.status(at: directory) == .complete else { return nil }
        let qwen = QwenModelPackage.isQwen(at: directory)
        if !qwen {
            guard let manifest = try? ManifestReader.load(directoryURL: directory, expecting: .gemma4_26B_A4B),
                  manifest.files.allSatisfy({ name, entry in
                      let file = directory.appendingPathComponent(name)
                      guard let values = try? file.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey]) else { return false }
                      return values.isRegularFile == true && values.isSymbolicLink != true
                          && UInt64(values.fileSize ?? 0) == entry.size
                  }) else { return nil }
        }
        let descriptor: AppModelInstallDescriptor
        if qwen {
            guard let variant = try? QwenModelPackage.variant(at: directory),
                  let supported = variant.installDescriptor else { return nil }
            descriptor = supported
        } else {
            descriptor = .default
        }
        return InstalledModel(directory: directory, descriptor: descriptor)
    }
}
