import TurboFieldfareAppCore
import SwiftUI
import TurboFieldfareMacPresentation

struct ModelStatusBadge: View {
    let model: AppModel

    var body: some View {
        HStack(spacing: 6) {
            statusDot
            ModelPickerView(model: model)
                .labelsHidden()
                .lineLimit(1)
                .frame(maxWidth: 240)
                .help("Choose a model to load or download. " + model.modelPathText)
                .accessibilityIdentifier(.hudStatus)
        }
    }

    @ViewBuilder
    private var statusDot: some View {
        switch model.presentation.severity {
        case .neutral: dot(.gray)
        case .active, .warning: dot(.orange)
        case .success: dot(.green)
        case .error: dot(.red)
        }
    }

    private func dot(_ color: Color) -> some View {
        Circle().fill(color).frame(width: 8, height: 8).accessibilityHidden(true)
    }
}
