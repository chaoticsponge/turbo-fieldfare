import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

@Suite(.serialized) struct QwenDownloadTests {
    private func fixture(_ body: String) throws -> (URL, URL) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let executable = root.appendingPathComponent("transport")
        let script = """
        #!/bin/sh
        while [ "$#" -gt 0 ]; do
          if [ "$1" = "--output" ]; then shift; target="$1"; fi
          shift
        done
        printf 'payload' > "$target"
        \(body)
        """
        try script.write(to: executable, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: executable.path)
        return (root, executable)
    }

    @Test func unsuccessfulResponsesRemoveOnlyAttemptFiles() async throws {
        for body in ["printf '503'; exit 22", "printf '206'; exit 0"] {
            let (root, executable) = try fixture(body)
            defer { try? FileManager.default.removeItem(at: root) }
            let target = root.appendingPathComponent("attempt.download")
            let verified = root.appendingPathComponent("verified")
            try Data("keep".utf8).write(to: verified)
            do {
                try await QwenFileDownload.download(from: URL(string: "https://example.com/model")!,
                    to: target, executable: executable, progress: { _ in })
                Issue.record("An unsuccessful or partial HTTP response was accepted")
            } catch let error as AppInferenceError {
                #expect(String(describing: error).contains("HTTP"))
            }
            #expect(!FileManager.default.fileExists(atPath: target.path))
            #expect(!FileManager.default.fileExists(atPath: target.appendingPathExtension("log").path))
            #expect(try Data(contentsOf: verified) == Data("keep".utf8))
        }
    }

    @Test func cancellationStopsTheChildAndRemovesItsPartialOutput() async throws {
        let (root, executable) = try fixture("exec /bin/sleep 30")
        defer { try? FileManager.default.removeItem(at: root) }
        let target = root.appendingPathComponent("attempt.download")
        let task = Task {
            try await QwenFileDownload.download(from: URL(string: "https://example.com/model")!,
                to: target, executable: executable, progress: { _ in })
        }
        defer { task.cancel() }
        let deadline = Date().addingTimeInterval(5)
        while !FileManager.default.fileExists(atPath: target.path), Date() < deadline {
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(FileManager.default.fileExists(atPath: target.path))
        task.cancel()
        do { try await task.value; Issue.record("Cancellation was ignored") }
        catch is CancellationError {}
        #expect(!FileManager.default.fileExists(atPath: target.path))
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["QWEN_DOWNLOAD_SMOKE_TEST"] == "1"))
    func pinnedFirstFileDownloadsAndPassesItsChecksum() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let catalog = try QwenModelPackage.catalog()
        let file = try #require(catalog.files.first)
        let target = root.appendingPathComponent("metadata.download")
        let source = URL(string: "https://huggingface.co/\(catalog.repoID)/resolve/\(catalog.revision)/\(file.name)")!
        try await QwenFileDownload.download(from: source, to: target, progress: { _ in })
        #expect(try QwenModelPackage.matches(file, at: target, hashes: true))
    }
}
