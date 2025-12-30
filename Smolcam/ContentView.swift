import SwiftUI
import Photos
import ImageIO

struct ContentView: View {
    @StateObject private var camera = CameraManager()
    @State private var fadeOpacity = 0.0
    @State private var iconRotation: Double = 0
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            
            VStack(spacing: 20) {
                ZStack {
                    if let preview = camera.previewImage {
                        Image(uiImage: preview)
                            .resizable()
                            .interpolation(.none)
                            .aspectRatio(contentMode: .fit)
                    }
                    
                    Color.black
                        .opacity(fadeOpacity)
                        .allowsHitTesting(false)
                }
                .aspectRatio(480.0/640.0, contentMode: .fit)
                .onTapGesture(count: 2) { flip() }
                
                ZStack {
                    Button(action: capture) {
                        Circle()
                            .stroke(Color.white, lineWidth: 4)
                            .frame(width: 70, height: 70)
                    }
                    
                    HStack {
                        Button { openPhotos() } label: {
                            Image(systemName: "photo")
                                .font(.system(size: 24))
                                .foregroundColor(.white)
                                .rotationEffect(.degrees(iconRotation))
                                .frame(width: 70, height: 70)
                        }
                        Spacer()
                        Button(action: flip) {
                            Image(systemName: "arrow.trianglehead.2.clockwise.rotate.90")
                                .font(.system(size: 28))
                                .foregroundColor(.white)
                                .rotationEffect(.degrees(iconRotation))
                                .frame(width: 70, height: 70)
                        }
                    }
                }
                .padding(.horizontal, 30)
                
                HStack(spacing: 0) {
                    ForEach(1...8, id: \.self) { n in
                        Button { camera.bitsPerComponent = n } label: {
                            Text("\(n)")
                                .font(.system(size: 14, weight: camera.bitsPerComponent == n ? .bold : .regular))
                                .frame(maxWidth: .infinity, minHeight: 44)
                                .background(camera.bitsPerComponent == n ? Color.white : Color.clear)
                                .foregroundColor(camera.bitsPerComponent == n ? .black : .white)
                        }
                    }
                }
                .background(Color.white.opacity(0.2))
                .cornerRadius(8)
                .padding(.horizontal, 30)
                
                Text("\(1 << (camera.bitsPerComponent * 3)) colors")
                    .foregroundColor(.gray)
                    .font(.system(size: 14))
                    .padding(.bottom, 30)
            }
        }
        .statusBarHidden(true)
        .onAppear { camera.start() }
        .onDisappear { camera.stop() }
        .onChange(of: camera.deviceOrientation) { updateIconRotation() }
    }
    
    private func flip() {
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
        camera.flipCamera()
    }
    
    private func openPhotos() {
        if let url = URL(string: "photos-redirect://") {
            UIApplication.shared.open(url)
        }
    }
    
    private func updateIconRotation() {
        let target: Double = switch camera.deviceOrientation {
        case .left: 90
        case .right: -90
        case .down: 180
        default: 0
        }
        var delta = target - iconRotation.truncatingRemainder(dividingBy: 360)
        if delta > 180 { delta -= 360 }
        if delta < -180 { delta += 360 }
        let duration = abs(delta) > 135 ? 0.4 : 0.3
        withAnimation(.easeInOut(duration: duration)) { iconRotation += delta }
    }
    
    private func capture() {
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        camera.capture()
        
        withAnimation(.easeInOut(duration: 0.2)) {
            fadeOpacity = 1.0
        }
        
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) {
            withAnimation(.easeInOut(duration: 0.2)) {
                fadeOpacity = 0.0
            }
        }
        
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            if let img = camera.capturedImage,
               let cgImage = img.cgImage,
               let data = pngDataWithMetadata(cgImage, bits: camera.bitsPerComponent) {
                PHPhotoLibrary.shared().performChanges {
                    let request = PHAssetCreationRequest.forAsset()
                    request.addResource(with: .photo, data: data, options: nil)
                }
            }
        }
    }
}

private func pngDataWithMetadata(_ cgImage: CGImage, bits: Int) -> Data? {
    let data = NSMutableData()
    guard let dest = CGImageDestinationCreateWithData(data, "public.png" as CFString, 1, nil) else { return nil }
    
    let metadata: [String: Any] = [
        kCGImagePropertyExifDictionary as String: [
            kCGImagePropertyExifLensModel as String: "Smolcam \(bits)-bit"
        ]
    ]
    
    CGImageDestinationAddImage(dest, cgImage, metadata as CFDictionary)
    CGImageDestinationFinalize(dest)
    return data as Data
}

#Preview {
    ContentView()
}
