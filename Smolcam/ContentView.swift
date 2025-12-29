import SwiftUI

struct ContentView: View {
    @StateObject private var camera = CameraManager()
    @State private var fadeOpacity = 0.0
    
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
                
                Button(action: capture) {
                    Circle()
                        .stroke(Color.white, lineWidth: 4)
                        .frame(width: 70, height: 70)
                }
                .padding(.bottom, 30)
            }
        }
        .statusBarHidden(true)
        .onAppear { camera.start() }
        .onDisappear { camera.stop() }
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
            if let img = camera.capturedImage {
                UIImageWriteToSavedPhotosAlbum(img, nil, nil, nil)
            }
        }
    }
}

#Preview {
    ContentView()
}
