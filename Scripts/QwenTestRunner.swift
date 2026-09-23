import Darwin
import Foundation
import Testing

/// Diagnostic entry point when the toolchain helper returns without tests.
@main struct QwenTestRunner {
    static func main() async {
        guard let path = ProcessInfo.processInfo.environment["QWEN_TEST_BUNDLE"],
              dlopen(path, RTLD_NOW | RTLD_GLOBAL) != nil else {
            let message = dlerror().map { String(cString: $0) } ?? "Missing QWEN_TEST_BUNDLE"
            fputs("Could not load test bundle: \(message)\n", stderr)
            exit(1)
        }
        await Testing.__swiftPMEntryPoint() as Never
    }
}
