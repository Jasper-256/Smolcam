//
//  LaunchSmolcamIntent.swift
//  Smolcam
//
//  Add to both Smolcam and SmolcamWidgetExtension targets.
//

import AppIntents

struct LaunchSmolcamIntent: OpenIntent {
    static var title: LocalizedStringResource = "Open Smolcam"

    @Parameter(title: "Target")
    var target: LaunchTarget
}

enum LaunchTarget: String, AppEnum {
    case camera
    
    static var typeDisplayRepresentation = TypeDisplayRepresentation("Smolcam")
    static var caseDisplayRepresentations: [LaunchTarget: DisplayRepresentation] = [
        .camera: DisplayRepresentation("Camera")
    ]
}
