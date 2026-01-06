//
//  SmolcamWidgetControl.swift
//  SmolcamWidget
//
//  Created by Jasper Morgal on 2026-01-05.
//

import AppIntents
import SwiftUI
import WidgetKit

struct SmolcamWidgetControl: ControlWidget {
    var body: some ControlWidgetConfiguration {
        StaticControlConfiguration(
            kind: "com.jasper.Smolcam.SmolcamWidget"
        ) {
            ControlWidgetButton(action: LaunchSmolcamIntent()) {
                Label("Open Smolcam", systemImage: "camera.viewfinder")
            }
        }
        .displayName("Open Smolcam")
        .description("Opens the Smolcam app")
    }
}
