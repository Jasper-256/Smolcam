import AVFoundation
import UIKit
import Combine
import CoreMotion

class CameraManager: NSObject, ObservableObject {
    @Published var previewImage: UIImage?
    @Published var capturedImage: UIImage?
    
    private let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "camera")
    private let motionManager = CMMotionManager()
    private var shouldCapture = false
    private var captureOrientation: UIImage.Orientation = .up
    
    override init() {
        super.init()
        setupSession()
        startMotionUpdates()
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
    
    private func startMotionUpdates() {
        motionManager.accelerometerUpdateInterval = 0.1
        motionManager.startAccelerometerUpdates()
    }
    
    private func currentOrientation() -> UIImage.Orientation {
        guard let data = motionManager.accelerometerData else { return .up }
        let x = data.acceleration.x
        let y = data.acceleration.y
        
        if abs(y) > abs(x) {
            return y < 0 ? .up : .down
        } else {
            return x < 0 ? .left : .right
        }
    }
    
    func start() {
        queue.async { self.session.startRunning() }
    }
    
    func stop() {
        queue.async { self.session.stopRunning() }
    }
    
    func capture() {
        captureOrientation = currentOrientation()
        shouldCapture = true
    }
    
    private func process(_ image: UIImage, forSave: Bool = false) -> UIImage {
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
        
        for i in stride(from: 0, to: pixelData.count, by: 4) {
            pixelData[i] = pixelData[i] < 128 ? 0 : 255
            pixelData[i+1] = pixelData[i+1] < 128 ? 0 : 255
            pixelData[i+2] = pixelData[i+2] < 128 ? 0 : 255
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
        
        let result = UIImage(cgImage: outputImage)
        
        if forSave {
            return rotateForSave(result)
        }
        return result
    }
    
    private func rotateForSave(_ image: UIImage) -> UIImage {
        guard let cgImage = image.cgImage else { return image }
        
        let orientation = captureOrientation
        if orientation == .up { return image }
        
        let width = cgImage.width
        let height = cgImage.height
        
        var transform = CGAffineTransform.identity
        var newSize = CGSize(width: width, height: height)
        
        switch orientation {
        case .down:
            transform = transform.translatedBy(x: CGFloat(width), y: CGFloat(height)).rotated(by: .pi)
        case .left:
            newSize = CGSize(width: height, height: width)
            transform = transform.translatedBy(x: CGFloat(height), y: 0).rotated(by: .pi / 2)
        case .right:
            newSize = CGSize(width: height, height: width)
            transform = transform.translatedBy(x: 0, y: CGFloat(width)).rotated(by: -.pi / 2)
        default:
            return image
        }
        
        guard let colorSpace = cgImage.colorSpace,
              let ctx = CGContext(
                data: nil,
                width: Int(newSize.width),
                height: Int(newSize.height),
                bitsPerComponent: cgImage.bitsPerComponent,
                bytesPerRow: 0,
                space: colorSpace,
                bitmapInfo: cgImage.bitmapInfo.rawValue
              ) else { return image }
        
        ctx.concatenate(transform)
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        guard let rotated = ctx.makeImage() else { return image }
        return UIImage(cgImage: rotated)
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
                self.capturedImage = self.process(uiImage, forSave: true)
            }
        }
    }
}
