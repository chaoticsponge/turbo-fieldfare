import Foundation

/// CFNetwork can time out on Hugging Face even when macOS curl can reach it.
/// Keep payloads on disk and preserve the installer's independent hash checks.
enum QwenFileDownload {
    static func download(from source: URL, to destination: URL,
        executable: URL = URL(fileURLWithPath: "/usr/bin/curl"),
        progress: @escaping @Sendable (UInt64) -> Void) async throws {
        guard source.scheme == "https" else {
            throw AppInferenceError.modelLoadFailed("Qwen downloads require HTTPS")
        }
        try Task.checkCancellation()
        #if os(macOS)
        let process = Process()
        let log = destination.appendingPathExtension("log")
        let status = destination.appendingPathExtension("status")
        let fm = FileManager.default
        // Unique files belong to this attempt, never to a verified payload.
        guard !fm.fileExists(atPath: destination.path),
              !fm.fileExists(atPath: log.path), !fm.fileExists(atPath: status.path) else {
            throw AppInferenceError.modelLoadFailed("Download staging path already exists")
        }
        var completed = false
        defer {
            try? fm.removeItem(at: log)
            try? fm.removeItem(at: status)
            if !completed { try? fm.removeItem(at: destination) }
        }
        try Data().write(to: log, options: .withoutOverwriting)
        try Data().write(to: status, options: .withoutOverwriting)
        let errors = try FileHandle(forWritingTo: log)
        let output = try FileHandle(forWritingTo: status)
        defer { try? errors.close(); try? output.close() }
        process.executableURL = executable
        // --disable must be first: user curl configuration must not change
        // redirects, TLS verification, output paths, or authentication.
        process.arguments = ["--disable", "--location", "--fail", "--silent", "--show-error",
            "--proto", "=https", "--proto-redir", "=https", "--max-redirs", "10",
            "--connect-timeout", "20", "--speed-limit", "1", "--speed-time", "60",
            "--max-time", "21600", "--retry", "3", "--retry-delay", "2",
            "--retry-max-time", "21600", "--retry-connrefused",
            "--output", destination.path, "--write-out", "%{http_code}", source.absoluteString]
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = output
        process.standardError = errors
        try process.run()
        do {
            while process.isRunning {
                try await Task.sleep(for: .milliseconds(100))
                let attributes = try? fm.attributesOfItem(atPath: destination.path)
                progress((attributes?[.size] as? NSNumber)?.uint64Value ?? 0)
            }
            process.waitUntilExit()
            try Task.checkCancellation()
        } catch {
            // Wait for our own child before removing its output or releasing
            // the install lock. Cancellation never leaves a downloader running.
            if process.isRunning { process.terminate() }
            process.waitUntilExit()
            throw error
        }
        let httpStatus = (try? String(contentsOf: status, encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? "unknown"
        guard process.terminationReason == .exit, process.terminationStatus == 0, httpStatus == "200" else {
            // Keep signed redirect URLs and raw network diagnostics out of UI.
            throw AppInferenceError.modelLoadFailed(
                "Download failed for \(source.lastPathComponent) (HTTP \(httpStatus), curl \(process.terminationStatus)). Check your connection and retry; verified files are kept.")
        }
        completed = true
        #else
        let session = URLSession(configuration: .ephemeral)
        defer { session.invalidateAndCancel() }
        let (file, response) = try await session.download(from: source)
        guard (response as? HTTPURLResponse)?.statusCode == 200 else {
            throw AppInferenceError.modelLoadFailed("Download failed for \(source.lastPathComponent)")
        }
        try FileManager.default.moveItem(at: file, to: destination)
        #endif
        let attributes = try FileManager.default.attributesOfItem(atPath: destination.path)
        progress((attributes[.size] as? NSNumber)?.uint64Value ?? 0)
    }
}
