import Foundation
import NIOHTTP1

/// Binding loopback alone does not reject browser DNS-rebinding requests.
enum ServerLoopbackPolicy {
    static func allows(_ headers: HTTPHeaders, port: Int?) -> Bool {
        let hosts = headers["host"]
        guard hosts.count == 1,
              let host = hosts.first,
              let authority = URLComponents(string: "http://" + host),
              let hostname = authority.host?.lowercased(),
              ["localhost", "127.0.0.1", "::1", "[::1]"].contains(hostname),
              authority.user == nil, authority.password == nil,
              authority.path.isEmpty, authority.query == nil, authority.fragment == nil,
              port == nil || (authority.port ?? 80) == port else { return false }
        let origins = headers["origin"]
        guard origins.count <= 1,
              origins.first.map({ $0.lowercased() == "http://" + host.lowercased() }) ?? true,
              !headers["sec-fetch-site"].contains(where: { $0.lowercased() == "cross-site" }) else {
            return false
        }
        return true
    }
}
