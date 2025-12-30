import AVFoundation
import UIKit
import Combine
import CoreMotion
import Accelerate

class CameraManager: NSObject, ObservableObject {
    @Published var previewImage: UIImage?
    @Published var capturedImage: UIImage?
    @Published var isFront = false
    @Published var bitsPerComponent = 4
    @Published var deviceOrientation: UIImage.Orientation = .up
    
    private let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "camera")
    private let processQueue = DispatchQueue(label: "process", qos: .userInteractive)
    private let motionManager = CMMotionManager()
    private var shouldCapture = false
    private var captureOrientation: UIImage.Orientation = .up
    private var lastOrientation: UIImage.Orientation = .up
    
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    private var isProcessing = false
    
    // Reusable buffers
    private var pixelBuffer = [UInt8](repeating: 0, count: 480 * 640 * 4)
    private var floatBuffer = [Float](repeating: 0, count: 480 * 640 * 4)
    
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
        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        if session.canAddOutput(output) { session.addOutput(output) }
        
        if let connection = output.connection(with: .video) {
            connection.videoRotationAngle = 90
        }
    }
    
    func flipCamera() {
        queue.async {
            self.session.beginConfiguration()
            
            if let current = self.session.inputs.first as? AVCaptureDeviceInput {
                self.session.removeInput(current)
            }
            
            let newPosition: AVCaptureDevice.Position = self.isFront ? .back : .front
            guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: newPosition),
                  let input = try? AVCaptureDeviceInput(device: device),
                  self.session.canAddInput(input) else {
                self.session.commitConfiguration()
                return
            }
            
            self.session.addInput(input)
            
            if let connection = self.output.connection(with: .video) {
                connection.videoRotationAngle = 90
                connection.isVideoMirrored = newPosition == .front
            }
            
            self.session.commitConfiguration()
            
            DispatchQueue.main.async { self.isFront = !self.isFront }
        }
    }
    
    private func startMotionUpdates() {
        motionManager.accelerometerUpdateInterval = 0.1
        motionManager.startAccelerometerUpdates(to: .main) { [weak self] data, _ in
            guard let self, let data else { return }
            let x = data.acceleration.x
            let y = data.acceleration.y
            guard max(abs(x), abs(y)) > 0.5 else { return }
            let orientation: UIImage.Orientation = abs(y) > abs(x) ? (y < 0 ? .up : .down) : (x < 0 ? .left : .right)
            self.lastOrientation = orientation
            self.deviceOrientation = orientation
        }
    }
    
    private func currentOrientation() -> UIImage.Orientation { lastOrientation }
    
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
    
    private func process(_ cgImage: CGImage, bits: Int, forSave: Bool = false) -> UIImage? {
        let width = 480, height = 640
        let bytesPerRow = width * 4
        let totalBytes = height * bytesPerRow
        
        // Draw directly into our reusable buffer
        guard let context = CGContext(
            data: &pixelBuffer,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }
        
        context.interpolationQuality = .low
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        // Quantize using Accelerate (SIMD)
        let levels = Float(1 << bits)
        let scale = 255.0 / (levels - 1)
        let divisor = 256.0 / levels
        
        vDSP_vfltu8(pixelBuffer, 1, &floatBuffer, 1, vDSP_Length(totalBytes))
        var div = divisor
        vDSP_vsdiv(floatBuffer, 1, &div, &floatBuffer, 1, vDSP_Length(totalBytes))
        var n = Int32(totalBytes)
        vvfloorf(&floatBuffer, floatBuffer, &n)
        var sc = scale
        vDSP_vsmul(floatBuffer, 1, &sc, &floatBuffer, 1, vDSP_Length(totalBytes))
        vDSP_vfixu8(floatBuffer, 1, &pixelBuffer, 1, vDSP_Length(totalBytes))
        
        guard let outContext = CGContext(
            data: &pixelBuffer,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ),
        let outputImage = outContext.makeImage() else { return nil }
        
        let result = UIImage(cgImage: outputImage)
        return forSave ? rotateForSave(result) : result
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
        let now = CACurrentMediaTime()
        let capture = shouldCapture
        
        // Skip if still processing previous frame (unless capturing)
        guard capture || !isProcessing else { return }
        isProcessing = true
        
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            isProcessing = false
            return
        }
        
        let ciImage = CIImage(cvPixelBuffer: pb)
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else {
            isProcessing = false
            return
        }
        
        let bits = bitsPerComponent
        
        processQueue.async { [weak self] in
            guard let self else { return }
            
            let preview = self.process(cgImage, bits: bits, forSave: false)
            var captured: UIImage?
            
            if capture {
                self.shouldCapture = false
                captured = self.process(cgImage, bits: bits, forSave: true)
            }
            
            DispatchQueue.main.async {
                self.previewImage = preview
                if let img = captured { self.capturedImage = img }
                self.isProcessing = false
            }
        }
    }
}
