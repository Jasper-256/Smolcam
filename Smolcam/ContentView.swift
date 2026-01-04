import SwiftUI
import Photos
import ImageIO
import MetalKit

struct MetalPreviewView: UIViewRepresentable {
    let camera: CameraManager
    
    func makeUIView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: camera.device)
        view.delegate = camera
        view.isPaused = true
        view.enableSetNeedsDisplay = false
        view.framebufferOnly = true
        view.colorPixelFormat = .bgra8Unorm
        view.autoResizeDrawable = false
        view.drawableSize = CGSize(width: 480, height: 640)
        view.contentMode = .scaleAspectFit
        view.backgroundColor = .black
        view.layer.magnificationFilter = .trilinear
        camera.metalView = view
        return view
    }
    
    func updateUIView(_ view: MTKView, context: Context) {}
}

struct ContentView: View {
    @StateObject private var camera = CameraManager()
    @State private var fadeOpacity = 0.0
    @State private var iconRotation = 0.0
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            
            VStack(spacing: 20) {
                ZStack {
                    MetalPreviewView(camera: camera)
                    
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
                
                HStack(spacing: 8) {
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
                    
                    Button { camera.ditherEnabled.toggle() } label: {
                        Image(systemName: "circle.grid.3x3")
                            .font(.system(size: 18))
                            .frame(width: 44, height: 44)
                            .background(camera.ditherEnabled ? Color.white : Color.white.opacity(0.2))
                            .foregroundColor(camera.ditherEnabled ? .black : .white)
                            .cornerRadius(8)
                    }
                }
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
        if delta == -180 { delta = 180 }
        withAnimation(.easeInOut(duration: abs(delta) > 135 ? 0.4 : 0.3)) {
            iconRotation += delta
        }
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
               let data = imageDataWithMetadata(cgImage, bits: camera.bitsPerComponent, dither: camera.ditherEnabled) {
                PHPhotoLibrary.shared().performChanges {
                    let request = PHAssetCreationRequest.forAsset()
                    request.addResource(with: .photo, data: data, options: nil)
                }
            }
        }
    }
}

private func imageDataWithMetadata(_ cgImage: CGImage, bits: Int, dither: Bool) -> Data? {
    let data = NSMutableData()
    let format = "public.png"
    guard let dest = CGImageDestinationCreateWithData(data, format as CFString, 1, nil) else { return nil }
    
    let ditherStr = dither ? " dithered" : " quantized"
    let metadata: [String: Any] = [
        kCGImagePropertyExifDictionary as String: [
            kCGImagePropertyExifLensModel as String: "Smolcam \(bits)-bit\(ditherStr)"
        ]
    ]
    
    CGImageDestinationAddImage(dest, cgImage, metadata as CFDictionary)
    CGImageDestinationFinalize(dest)
    return data as Data
}

#Preview {
    ContentView()
}
