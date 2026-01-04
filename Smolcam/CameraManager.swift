import AVFoundation
import UIKit
import Combine
import CoreMotion
import Metal
import MetalKit

class CameraManager: NSObject, ObservableObject {
    @Published var capturedImage: UIImage?
    @Published var isFront = false
    @Published var bitsPerComponent = 4
    @Published var deviceOrientation: UIImage.Orientation = .up
    @Published var ditherEnabled = true
    
    private let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "camera")
    private let motionManager = CMMotionManager()
    private var shouldCapture = false
    private var captureOrientation: UIImage.Orientation = .up
    private var lastOrientation: UIImage.Orientation = .up
    
    // Metal
    let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let computePipeline: MTLComputePipelineState
    private let renderPipeline: MTLRenderPipelineState
    private let textureCache: CVMetalTextureCache
    private var processedTexture: MTLTexture?
    
    // For MTKView rendering
    weak var metalView: MTKView?
    private var currentTexture: MTLTexture?
    private let textureLock = NSLock()
    
    override init() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue(),
              let library = device.makeDefaultLibrary(),
              let computeKernel = library.makeFunction(name: "ditherQuantize"),
              let computePipeline = try? device.makeComputePipelineState(function: computeKernel),
              let vertexFunc = library.makeFunction(name: "vertexPassthrough"),
              let fragFunc = library.makeFunction(name: "fragmentPassthrough") else {
            fatalError("Metal init failed")
        }
        
        let renderDesc = MTLRenderPipelineDescriptor()
        renderDesc.vertexFunction = vertexFunc
        renderDesc.fragmentFunction = fragFunc
        renderDesc.colorAttachments[0].pixelFormat = .bgra8Unorm
        guard let renderPipeline = try? device.makeRenderPipelineState(descriptor: renderDesc) else {
            fatalError("Render pipeline failed")
        }
        
        var cache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
        guard let textureCache = cache else { fatalError("Texture cache failed") }
        
        self.device = device
        self.commandQueue = commandQueue
        self.computePipeline = computePipeline
        self.renderPipeline = renderPipeline
        self.textureCache = textureCache
        
        super.init()
        setupSession()
        startMotionUpdates()
    }
    
    private func setupSession() {
        session.sessionPreset = .vga640x480
        
        guard let camDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: camDevice) else { return }
        
        // Lock frame duration for consistency
        try? camDevice.lockForConfiguration()
        camDevice.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 30)
        camDevice.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 30)
        camDevice.unlockForConfiguration()
        
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
            guard let camDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: newPosition),
                  let input = try? AVCaptureDeviceInput(device: camDevice),
                  self.session.canAddInput(input) else {
                self.session.commitConfiguration()
                return
            }
            
            try? camDevice.lockForConfiguration()
            camDevice.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 30)
            camDevice.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 30)
            camDevice.unlockForConfiguration()
            
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
    
    private func processToTexture(_ pixelBuffer: CVPixelBuffer, bits: Int, dither: Bool) -> MTLTexture? {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        
        var cvTexture: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(nil, textureCache, pixelBuffer, nil, .bgra8Unorm, width, height, 0, &cvTexture)
        guard let cvTex = cvTexture, let inTexture = CVMetalTextureGetTexture(cvTex) else { return nil }
        
        if processedTexture == nil || processedTexture!.width != width || processedTexture!.height != height {
            let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
            desc.usage = [.shaderRead, .shaderWrite]
            processedTexture = device.makeTexture(descriptor: desc)
        }
        guard let outTex = processedTexture else { return nil }
        
        guard let cmdBuffer = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuffer.makeComputeCommandEncoder() else { return nil }
        
        var bitsVal = Int32(bits)
        var ditherVal: Int32 = dither ? 1 : 0
        
        encoder.setComputePipelineState(computePipeline)
        encoder.setTexture(inTexture, index: 0)
        encoder.setTexture(outTex, index: 1)
        encoder.setBytes(&bitsVal, length: 4, index: 0)
        encoder.setBytes(&ditherVal, length: 4, index: 1)
        
        let tgSize = MTLSize(width: 16, height: 16, depth: 1)
        let tgCount = MTLSize(width: (width + 15) / 16, height: (height + 15) / 16, depth: 1)
        encoder.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()
        
        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()
        
        return outTex
    }
    
    private func textureToImage(_ texture: MTLTexture) -> UIImage? {
        let width = texture.width, height = texture.height
        let bytesPerRow = width * 4
        var pixels = [UInt8](repeating: 0, count: height * bytesPerRow)
        texture.getBytes(&pixels, bytesPerRow: bytesPerRow, from: MTLRegion(origin: .init(), size: .init(width: width, height: height, depth: 1)), mipmapLevel: 0)
        
        guard let context = CGContext(
            data: &pixels, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
        ), let cgImage = context.makeImage() else { return nil }
        
        return rotateForSave(UIImage(cgImage: cgImage))
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

extension CameraManager: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
    
    func draw(in view: MTKView) {
        textureLock.lock()
        let tex = currentTexture
        textureLock.unlock()
        
        guard let texture = tex,
              let drawable = view.currentDrawable,
              let cmdBuffer = commandQueue.makeCommandBuffer() else { return }
        
        let passDesc = MTLRenderPassDescriptor()
        passDesc.colorAttachments[0].texture = drawable.texture
        passDesc.colorAttachments[0].loadAction = .clear
        passDesc.colorAttachments[0].storeAction = .store
        passDesc.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        
        guard let encoder = cmdBuffer.makeRenderCommandEncoder(descriptor: passDesc) else { return }
        encoder.setRenderPipelineState(renderPipeline)
        encoder.setFragmentTexture(texture, index: 0)
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        encoder.endEncoding()
        
        cmdBuffer.present(drawable)
        cmdBuffer.commit()
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        let bits = bitsPerComponent
        let dither = ditherEnabled
        let capture = shouldCapture
        
        guard let texture = processToTexture(pb, bits: bits, dither: dither) else { return }
        
        textureLock.lock()
        currentTexture = texture
        textureLock.unlock()
        
        DispatchQueue.main.async {
            self.metalView?.draw()
        }
        
        if capture {
            shouldCapture = false
            if let img = textureToImage(texture) {
                DispatchQueue.main.async { self.capturedImage = img }
            }
        }
    }
}
