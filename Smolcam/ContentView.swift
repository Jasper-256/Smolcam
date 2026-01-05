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
    @State private var baseZoom: CGFloat = 1.0
    @State private var lastMag: CGFloat = 0
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            
            VStack(spacing: 20) {
                ZStack(alignment: .bottom) {
                    MetalPreviewView(camera: camera)
                    
                    Color.black
                        .opacity(fadeOpacity)
                        .allowsHitTesting(false)
                    
                    if camera.displayZoom != 1.0 {
                        Text(String(format: "%.1fx", floor(camera.displayZoom * 10) / 10))
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundColor(.white)
                            .frame(width: 35, height: 35)
                            .background(Circle().fill(.black.opacity(0.5)))
                            .rotationEffect(.degrees(iconRotation))
                            .padding(.bottom, 10)
                            .onTapGesture(perform: resetZoom)
                            .transition(.opacity)
                    }
                }
                .aspectRatio(480.0/640.0, contentMode: .fit)
                .simultaneousGesture(TapGesture(count: 2).onEnded { flip() })
                .gesture(
                    MagnifyGesture(minimumScaleDelta: 0)
                        .onChanged { value in
                            if lastMag == 0 {
                                lastMag = value.magnification
                                return
                            }
                            let deltaMag = value.magnification / lastMag
                            let relZoom = camera.zoomLevel / camera.baseZoomFactor
                            let speed = max(1.5, 1.0 + 1.442695 * log(relZoom))
                            camera.setZoom(camera.zoomLevel * pow(deltaMag, speed))
                            lastMag = value.magnification
                        }
                        .onEnded { _ in
                            baseZoom = camera.zoomLevel
                            lastMag = 0
                        }
                )
                
                ZStack {
                    Button(action: capture) {
                        Circle()
                            .stroke(Color.white, lineWidth: 4)
                            .frame(width: 70, height: 70)
                    }
                    
                    HStack {
                        Button { openPhotos() } label: {
                            Image(systemName: "photo")
                                .font(.system(size: 25))
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
                .padding(.horizontal, 20)
                
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
                            .background(camera.ditherEnabled ? Color.white : Color.clear)
                            .foregroundColor(camera.ditherEnabled ? .black : .white)
                    }
                    .background(Color.white.opacity(0.2))
                    .cornerRadius(8)
                }
                .padding(.horizontal, 20)
                
                Text("\(1 << (camera.bitsPerComponent * 3)) colors")
                    .foregroundColor(.gray)
                    .font(.system(size: 14))
            }
        }
        .statusBarHidden(true)
        .onAppear {
            camera.start()
            baseZoom = camera.baseZoomFactor
        }
        .onDisappear { camera.stop() }
        .onChange(of: camera.deviceOrientation) { updateIconRotation() }
        .onChange(of: camera.isFront) { baseZoom = camera.zoomLevel }
    }
    
    private func flip() {
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
        camera.flipCamera()
    }
    
    private func resetZoom() {
        camera.animateZoom(to: camera.baseZoomFactor)
        baseZoom = camera.baseZoomFactor
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
