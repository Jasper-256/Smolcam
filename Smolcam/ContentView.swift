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
                .onTapGesture(count: 2) { flip() }
                
                ZStack {
                    Button(action: capture) {
                        Circle()
                            .stroke(Color.white, lineWidth: 4)
                            .frame(width: 70, height: 70)
                    }
                    
                    HStack {
                        Spacer()
                        Button(action: flip) {
                            Image(systemName: "arrow.trianglehead.2.clockwise.rotate.90")
                                .font(.system(size: 28))
                                .foregroundColor(.white)
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
    }
    
    private func flip() {
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
        camera.flipCamera()
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
