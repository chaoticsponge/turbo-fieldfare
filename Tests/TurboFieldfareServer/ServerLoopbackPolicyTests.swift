import NIOHTTP1
import Testing
@testable import TurboFieldfareServerCore

@Suite struct ServerLoopbackPolicyTests {
    @Test func allowsLocalHarnessAndSameOriginBrowser() {
        for host in ["127.0.0.1:8080", "localhost:8080", "[::1]:8080"] {
            var headers = HTTPHeaders([("host", host)])
            #expect(ServerLoopbackPolicy.allows(headers, port: 8080))
            headers.add(name: "origin", value: "http://" + host)
            #expect(ServerLoopbackPolicy.allows(headers, port: 8080))
        }
    }

    @Test func rejectsRebindingAndAmbiguousAuthority() {
        for host in ["evil.example:8080", "localhost:9999", "127.0.0.1:8080@evil.example", "localhost:8080/path"] {
            #expect(!ServerLoopbackPolicy.allows(HTTPHeaders([("host", host)]), port: 8080))
        }
        #expect(!ServerLoopbackPolicy.allows(HTTPHeaders(), port: 8080))
        #expect(!ServerLoopbackPolicy.allows(HTTPHeaders([("host", "localhost:8080"), ("host", "localhost:8080")]), port: 8080))
    }

    @Test func rejectsCrossOriginAndOpaqueBrowserRequests() {
        for origin in ["https://evil.example", "null", "http://localhost:9999"] {
            #expect(!ServerLoopbackPolicy.allows(HTTPHeaders([("host", "localhost:8080"), ("origin", origin)]), port: 8080))
        }
        #expect(!ServerLoopbackPolicy.allows(HTTPHeaders([("host", "localhost:8080"), ("sec-fetch-site", "cross-site")]), port: 8080))
    }
}
