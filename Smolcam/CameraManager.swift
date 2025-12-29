import AVFoundation
import UIKit
import Combine

class CameraManager: NSObject, ObservableObject {
    @Published var previewImage: UIImage?
    @Published var capturedImage: UIImage?
    
    private let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "camera")
    private var shouldCapture = false
    
    override init() {
        super.init()
        setupSession()
    }
    
    private func setupSession() {
        session.sessionPreset = .vga640x480
        
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device) else { return }
        
        if session.canAddInput(input) { session.addInput(input) }
        
        output.setSampleBufferDelegate(self, queue: queue)
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        if session.canAddOutput(output) { session.addOutput(output) }
        
        if let connection = output.connection(with: .video) {
            connection.videoRotationAngle = 90
        }
    }
    
    func start() {
        queue.async { self.session.startRunning() }
    }
    
    func stop() {
        queue.async { self.session.stopRunning() }
    }
    
    func capture() {
        shouldCapture = true
    }
    
    private func process(_ image: UIImage) -> UIImage {
        let size = CGSize(width: 480, height: 640)
        UIGraphicsBeginImageContext(size)
        image.draw(in: CGRect(origin: .zero, size: size))
        let resized = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()
        
        guard let cgImage = resized.cgImage else { return resized }
        
        let width = cgImage.width
        let height = cgImage.height
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        
        var pixelData = [UInt8](repeating: 0, count: height * bytesPerRow)
        
        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return resized }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        // 3-bit color: 1 bit per channel (8 colors total)
        for i in stride(from: 0, to: pixelData.count, by: 4) {
            pixelData[i] = pixelData[i] < 128 ? 0 : 255     // R
            pixelData[i+1] = pixelData[i+1] < 128 ? 0 : 255 // G
            pixelData[i+2] = pixelData[i+2] < 128 ? 0 : 255 // B
        }
        
        guard let outputContext = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ),
        let outputImage = outputContext.makeImage() else { return resized }
        
        return UIImage(cgImage: outputImage)
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return }
        let uiImage = UIImage(cgImage: cgImage)
        
        DispatchQueue.main.async {
            self.previewImage = self.process(uiImage)
            
            if self.shouldCapture {
                self.shouldCapture = false
                self.capturedImage = self.previewImage
            }
        }
    }
}

