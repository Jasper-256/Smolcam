import AVFoundation
import UIKit
import Combine
import CoreMotion
import Metal
import MetalKit

class CameraManager: NSObject, ObservableObject {
    @Published var previewImage: UIImage?
    @Published var capturedImage: UIImage?
    @Published var isFront = false
    @Published var bitsPerComponent = 4
    @Published var deviceOrientation: UIImage.Orientation = .up
    @Published var ditherEnabled = false
    
    private let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "camera")
    private let motionManager = CMMotionManager()
    private var shouldCapture = false
    private var captureOrientation: UIImage.Orientation = .up
    private var lastOrientation: UIImage.Orientation = .up
    private var isProcessing = false
    
    // Metal
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipelineState: MTLComputePipelineState
    private let textureCache: CVMetalTextureCache
    private var outTexture: MTLTexture?
    
    override init() {
        // Setup Metal
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue(),
              let library = device.makeDefaultLibrary(),
              let kernel = library.makeFunction(name: "ditherQuantize"),
              let pipelineState = try? device.makeComputePipelineState(function: kernel) else {
            fatalError("Metal init failed")
        }
        
        var cache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
        guard let textureCache = cache else { fatalError("Texture cache failed") }
        
        self.device = device
        self.commandQueue = commandQueue
        self.pipelineState = pipelineState
        self.textureCache = textureCache
        
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
            let x = data.acceleration.x, y = data.acceleration.y
            guard max(abs(x), abs(y)) > 0.5 else { return }
            let orientation: UIImage.Orientation = abs(y) > abs(x) ? (y < 0 ? .up : .down) : (x < 0 ? .left : .right)
            self.lastOrientation = orientation
            self.deviceOrientation = orientation
        }
    }
    
    func start() { queue.async { self.session.startRunning() } }
    func stop() { queue.async { self.session.stopRunning() } }
    
    func capture() {
        captureOrientation = lastOrientation
        shouldCapture = true
    }
    
    private func process(_ pixelBuffer: CVPixelBuffer, bits: Int, dither: Bool, forSave: Bool) -> UIImage? {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        
        // Create input texture from pixel buffer
        var cvTexture: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(nil, textureCache, pixelBuffer, nil, .bgra8Unorm, width, height, 0, &cvTexture)
        guard let cvTex = cvTexture, let inTexture = CVMetalTextureGetTexture(cvTex) else { return nil }
        
        // Create/reuse output texture
        if outTexture == nil || outTexture!.width != width || outTexture!.height != height {
            let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
            desc.usage = [.shaderRead, .shaderWrite]
            outTexture = device.makeTexture(descriptor: desc)
        }
        guard let outTex = outTexture else { return nil }
        
        // Encode compute
        guard let cmdBuffer = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuffer.makeComputeCommandEncoder() else { return nil }
        
        var bitsVal = Int32(bits)
        var ditherVal: Int32 = dither ? 1 : 0
        
        encoder.setComputePipelineState(pipelineState)
        encoder.setTexture(inTexture, index: 0)
        encoder.setTexture(outTex, index: 1)
        encoder.setBytes(&bitsVal, length: MemoryLayout<Int32>.size, index: 0)
        encoder.setBytes(&ditherVal, length: MemoryLayout<Int32>.size, index: 1)
        
        let threadGroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let threadGroups = MTLSize(
            width: (width + 15) / 16,
            height: (height + 15) / 16,
            depth: 1
        )
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()
        
        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()
        
        // Read back
        let bytesPerRow = width * 4
        var pixels = [UInt8](repeating: 0, count: height * bytesPerRow)
        outTex.getBytes(&pixels, bytesPerRow: bytesPerRow, from: MTLRegion(origin: .init(), size: .init(width: width, height: height, depth: 1)), mipmapLevel: 0)
        
        // Create CGImage (BGRA -> RGBA swap not needed for display, but fix byte order)
        guard let context = CGContext(
            data: &pixels,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
        ), let cgImage = context.makeImage() else { return nil }
        
        let result = UIImage(cgImage: cgImage)
        return forSave ? rotateForSave(result) : result
    }
    
    private func rotateForSave(_ image: UIImage) -> UIImage {
        guard let cgImage = image.cgImage, captureOrientation != .up else { return image }
        
        let width = cgImage.width, height = cgImage.height
        var transform = CGAffineTransform.identity
        var newSize = CGSize(width: width, height: height)
        
        switch captureOrientation {
        case .down:
            transform = transform.translatedBy(x: CGFloat(width), y: CGFloat(height)).rotated(by: .pi)
        case .left:
            newSize = CGSize(width: height, height: width)
            transform = transform.translatedBy(x: CGFloat(height), y: 0).rotated(by: .pi / 2)
        case .right:
            newSize = CGSize(width: height, height: width)
            transform = transform.translatedBy(x: 0, y: CGFloat(width)).rotated(by: -.pi / 2)
        default: return image
        }
        
        guard let colorSpace = cgImage.colorSpace,
              let ctx = CGContext(data: nil, width: Int(newSize.width), height: Int(newSize.height),
                                  bitsPerComponent: cgImage.bitsPerComponent, bytesPerRow: 0,
                                  space: colorSpace, bitmapInfo: cgImage.bitmapInfo.rawValue) else { return image }
        
        ctx.concatenate(transform)
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let rotated = ctx.makeImage() else { return image }
        return UIImage(cgImage: rotated)
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        let capture = shouldCapture
        guard capture || !isProcessing else { return }
        isProcessing = true
        
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            isProcessing = false
            return
        }
        
        let bits = bitsPerComponent
        let dither = ditherEnabled
        
        let preview = process(pb, bits: bits, dither: dither, forSave: false)
        var captured: UIImage?
        
        if capture {
            shouldCapture = false
            captured = process(pb, bits: bits, dither: dither, forSave: true)
        }
        
        DispatchQueue.main.async {
            self.previewImage = preview
            if let img = captured { self.capturedImage = img }
            self.isProcessing = false
        }
    }
}
