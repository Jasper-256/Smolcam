import SwiftUI
import Photos
import MetalKit

private func colorFormatName(_ bitsPerPixel: Int) -> String {
    let names: [Int: String] = [8: "RGB332", 16: "RGB565"]
    if let name = names[bitsPerPixel] { return name }
    let b = bitsPerPixel / 3
    return "RGB\(b)\(b)\(b)"
}

struct MetalPreviewView: UIViewRepresentable {
    let camera: CameraManager
    
    func makeUIView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: camera.device)
        view.delegate = camera
        view.isPaused = true
        view.enableSetNeedsDisplay = false
        view.framebufferOnly = true
        view.colorPixelFormat = .bgra8Unorm
        view.autoResizeDrawable = true
        view.contentMode = .scaleAspectFit
        view.backgroundColor = .black
        camera.metalView = view
        return view
    }
    
    func updateUIView(_ view: MTKView, context: Context) {}
}

struct ContentView: View {
    @StateObject private var camera = CameraManager()
    @Environment(\.scenePhase) private var scenePhase
    @State private var fadeOpacity = 0.0
    @State private var iconRotation = 0.0
    @State private var baseZoom: CGFloat = 1.0
    @State private var lastMag: CGFloat = 0
    @State private var pendingSavePixelBits: Int?
    @State private var pendingSaveDither: Bool?
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            
            VStack(spacing: 20) {
                ZStack(alignment: .bottom) {
                    MetalPreviewView(camera: camera)
                    
                    if scenePhase == .background, let snapshot = camera.backgroundSnapshot {
                        Image(uiImage: snapshot)
                            .interpolation(.none)
                            .resizable()
                    }
                    
                    Color.black
                        .opacity(fadeOpacity)
                        .allowsHitTesting(false)
                    
                    if camera.displayZoom < 1.0 || camera.displayZoom > 1.02 {
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
                .simultaneousGesture(TapGesture(count: 2).onEnded(flip))
                .gesture(magnificationGesture)
                
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
                            Image(systemName: "arrow.triangle.2.circlepath")
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
                        ForEach([3, 6, 8, 12, 16, 24], id: \.self) { n in
                            Button { camera.bitsPerPixel = n } label: {
                                Text("\(n)")
                                    .font(.system(size: 16, weight: camera.bitsPerPixel == n ? .bold : .regular))
                                    .frame(maxWidth: .infinity, minHeight: 44)
                                    .background(camera.bitsPerPixel == n ? Color.white : Color.clear)
                                    .foregroundColor(camera.bitsPerPixel == n ? .black : .white)
                            }
                        }
                    }
                    .background(Color.white.opacity(0.2))
                    .cornerRadius(8)
                    
                    Button { camera.ditherEnabled.toggle() } label: {
                        Image(systemName: "checkerboard.rectangle")
                            .rotationEffect(.degrees(90))
                            .font(.system(size: 16))
                            .frame(width: 44, height: 44)
                            .background(camera.ditherEnabled && camera.bitsPerPixel != 24 ? Color.white : Color.clear)
                            .foregroundColor(camera.bitsPerPixel == 24 ? .gray : (camera.ditherEnabled ? .black : .white))
                    }
                    .disabled(camera.bitsPerPixel == 24)
                    .background(Color.white.opacity(0.2))
                    .cornerRadius(8)
                }
                .padding(.horizontal, 20)
                
                HStack {
                    Text("\(1 << camera.bitsPerPixel) colors")
                        .foregroundColor(.gray)
                        .font(.system(size: 14))
                        .frame(maxWidth: .infinity, alignment: .leading)
                    
                    Text(colorFormatName(camera.bitsPerPixel))
                        .foregroundColor(.gray)
                        .font(.system(size: 14))
                    
                    Text(camera.ditherEnabled && camera.bitsPerPixel != 24 ? "dither on" : "dither off")
                        .foregroundColor(.gray)
                        .font(.system(size: 14))
                        .frame(maxWidth: .infinity, alignment: .trailing)
                }
                .padding(.horizontal, 20)
                .padding(.bottom, hasHomeButton ? 20 : 0)
            }
        }
        .statusBarHidden(true)
        .onAppear {
            camera.start()
            baseZoom = camera.baseZoomFactor
            updateIconRotation(animated: false)
        }
        .onDisappear { camera.stop() }
        .onChange(of: camera.deviceOrientation) { _ in updateIconRotation() }
        .onChange(of: camera.isFront) { _ in baseZoom = camera.zoomLevel }
        .onChange(of: camera.capturedImage) { newImage in
            guard let img = newImage,
                  let cgImage = img.cgImage,
                  let pixelBits = pendingSavePixelBits,
                  let dither = pendingSaveDither else { return }
            pendingSavePixelBits = nil
            pendingSaveDither = nil
            DispatchQueue.global(qos: .userInitiated).async {
                guard let data = imageDataWithMetadata(cgImage, pixelBits: pixelBits, dither: dither) else { return }
                Task {
                    try? await PHPhotoLibrary.shared().performChanges {
                        let request = PHAssetCreationRequest.forAsset()
                        request.addResource(with: .photo, data: data, options: nil)
                    }
                }
            }
        }
    }
    
    private var magnificationGesture: some Gesture {
        if #available(iOS 17.0, *) {
            return MagnifyGesture(minimumScaleDelta: 0)
                .onChanged { value in
                    handleMagnificationChange(value.magnification)
                }
                .onEnded { _ in
                    handleMagnificationEnd()
                }
        } else {
            return MagnificationGesture(minimumScaleDelta: 0)
                .onChanged { value in
                    handleMagnificationChange(value)
                }
                .onEnded { _ in
                    handleMagnificationEnd()
                }
        }
    }
    
    private func handleMagnificationChange(_ magnification: CGFloat) {
        if lastMag == 0 {
            lastMag = magnification
            return
        }
        let deltaMag = magnification / lastMag
        let relZoom = camera.zoomLevel / camera.baseZoomFactor
        let speed = max(1.5, 1.0 + 1.442695 * log(relZoom))
        camera.setZoom(camera.zoomLevel * pow(deltaMag, speed))
        lastMag = magnification
    }
    
    private func handleMagnificationEnd() {
        baseZoom = camera.zoomLevel
        lastMag = 0
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
    
    private func updateIconRotation(animated: Bool = true) {
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
        if animated {
            withAnimation(.easeInOut(duration: abs(delta) > 135 ? 0.4 : 0.3)) {
                iconRotation += delta
            }
        } else {
            iconRotation += delta
        }
    }
    
    private func capture() {
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        pendingSavePixelBits = camera.bitsPerPixel
        pendingSaveDither = camera.ditherEnabled && camera.bitsPerPixel != 24
        camera.capture()
        
        withAnimation(.easeInOut(duration: 0.2)) {
            fadeOpacity = 1.0
        }
        
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) {
            withAnimation(.easeInOut(duration: 0.2)) {
                fadeOpacity = 0.0
            }
        }
    }
}

#Preview {
    ContentView()
}
