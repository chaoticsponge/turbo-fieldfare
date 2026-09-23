import Foundation
import SwiftUI
import TurboFieldfareAppCore
import TurboFieldfareMacPresentation

struct ModelPickerView: View {
    let model: AppModel

    private var selection: String {
        if model.installedModels.contains(where: { $0.id == model.modelPathText }) {
            return "installed:" + model.modelPathText
        }
        if let variant = QwenModelVariant.matching(repoID: model.installDescriptor.repoID) {
            return "download:" + variant.id
        }
        return "current"
    }

    var body: some View {
        Picker("Model", selection: Binding(get: { selection }, set: { value in
            if value.hasPrefix("installed:") {
                model.selectInstalledModel(URL(fileURLWithPath: String(value.dropFirst("installed:".count))))
            } else if value.hasPrefix("download:"),
                      let variant = QwenModelVariant(rawValue: String(value.dropFirst("download:".count))) {
                model.selectQwenModelForInstallation(variant)
            }
        })) {
            if selection == "current" {
                Text(model.installDescriptor.displayName).tag("current").disabled(true)
            }
            Section("Installed") {
                ForEach(model.installedModels) { installed in
                    Text(installed.descriptor.displayName + " — " + installed.directory.lastPathComponent)
                        .tag("installed:" + installed.id)
                        .help(installed.directory.path)
                }
            }
            Section("Available to Download") {
                ForEach(model.downloadableQwenModels.filter { variant in
                    !model.installedModels.contains { $0.descriptor.repoID == variant.repoID }
                        || selection == "download:" + variant.id
                }) { variant in
                    if let descriptor = variant.installDescriptor {
                        Text(variant.displayName + " — " + MetricFormat.storage(descriptor.approximateDownloadBytes))
                            .tag("download:" + variant.id)
                    }
                }
            }
        }
        .pickerStyle(.menu)
        .disabled(!model.canSelectInstalledModel)
        .onAppear { model.refreshInstalledModels() }
        .onChange(of: model.installationStatus) { model.refreshInstalledModels() }
    }
}
