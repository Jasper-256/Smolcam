import AVFoundation
import UIKit
import Combine
import Metal
import MetalKit

class CameraManager: NSObject, ObservableObject {
    @Published var capturedImage: UIImage?
    @Published var isFront = false
    @Published var bitsPerComponent = 2
    @Published var deviceOrientation: UIImage.Orientation = .up
    @Published var ditherEnabled = true
    @Published var zoomLevel: CGFloat = 1.0
    
    var backgroundWidth = 3
    var backgroundHeight = 4
    var backgroundSnapshot: UIImage?
    
    private var currentDevice: AVCaptureDevice?
    private(set) var baseZoomFactor: CGFloat = 1.0
    private var backCameraMaxOpticalZoom: CGFloat = 1.0
    
    var displayZoom: CGFloat { zoomLevel / baseZoomFactor }
    var minDisplayZoom: CGFloat { 1.0 / baseZoomFactor }
    
    private let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "camera")
    private var shouldCapture = false
    private var captureOrientation: UIImage.Orientation = .up
    
    // Metal
    let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let computePipeline: MTLComputePipelineState
    private let downsamplePipeline: MTLComputePipelineState
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
              let downsampleKernel = library.makeFunction(name: "downsampleQuantize"),
              let downsamplePipeline = try? device.makeComputePipelineState(function: downsampleKernel),
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
        self.downsamplePipeline = downsamplePipeline
        self.renderPipeline = renderPipeline
        self.textureCache = textureCache
        
        super.init()
        setupSession()
        startOrientationUpdates()
    }
    
    private func setupSession() {
        session.beginConfiguration()
        session.sessionPreset = .vga640x480
        
        guard let camDevice = bestCamera(for: .back),
              let input = try? AVCaptureDeviceInput(device: camDevice) else {
            session.commitConfiguration()
            return
        }
        
        currentDevice = camDevice
        baseZoomFactor = camDevice.virtualDeviceSwitchOverVideoZoomFactors.first?.doubleValue ?? 1.0
        backCameraMaxOpticalZoom = camDevice.virtualDeviceSwitchOverVideoZoomFactors.last?.doubleValue ?? 1.0
        zoomLevel = baseZoomFactor
        
        if session.canAddInput(input) { session.addInput(input) }
        
        try? camDevice.lockForConfiguration()
        camDevice.videoZoomFactor = baseZoomFactor
        camDevice.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 30)
        camDevice.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 30)
        camDevice.unlockForConfiguration()
        
        output.setSampleBufferDelegate(self, queue: queue)
        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        if session.canAddOutput(output) { session.addOutput(output) }
        
        if let connection = output.connection(with: .video) {
            connection.videoRotationAngle = 90
        }
        
        session.commitConfiguration()
    }
    
    private func bestCamera(for position: AVCaptureDevice.Position) -> AVCaptureDevice? {
        let types: [AVCaptureDevice.DeviceType] = position == .back
            ? [.builtInTripleCamera, .builtInDualWideCamera, .builtInDualCamera, .builtInWideAngleCamera]
            : [.builtInWideAngleCamera]
        return AVCaptureDevice.DiscoverySession(deviceTypes: types, mediaType: .video, position: position).devices.first
    }
    
    private var maxZoomFactor: CGFloat {
        guard let device = currentDevice else { return 1.0 }
        let limit = isFront ? 5.0 : 5.0 * backCameraMaxOpticalZoom
        return min(device.maxAvailableVideoZoomFactor, limit)
    }
    
    func setZoom(_ factor: CGFloat) {
        guard let device = currentDevice else { return }
        let clamped = max(device.minAvailableVideoZoomFactor, min(factor, maxZoomFactor))
        try? device.lockForConfiguration()
        device.videoZoomFactor = clamped
        device.unlockForConfiguration()
        DispatchQueue.main.async { self.zoomLevel = clamped }
    }
    
    func animateZoom(to factor: CGFloat) {
        guard let device = currentDevice else { return }
        let clamped = max(device.minAvailableVideoZoomFactor, min(factor, maxZoomFactor))
        try? device.lockForConfiguration()
        device.ramp(toVideoZoomFactor: clamped, withRate: 32.0)
        device.unlockForConfiguration()
        DispatchQueue.main.async { self.zoomLevel = clamped }
    }
    
    func flipCamera() {
        queue.async {
            self.session.beginConfiguration()
            
            if let current = self.session.inputs.first as? AVCaptureDeviceInput {
                self.session.removeInput(current)
            }
            
            let newPosition: AVCaptureDevice.Position = self.isFront ? .back : .front
            guard let camDevice = self.bestCamera(for: newPosition),
                  let input = try? AVCaptureDeviceInput(device: camDevice),
                  self.session.canAddInput(input) else {
                self.session.commitConfiguration()
                return
            }
            
            self.currentDevice = camDevice
            let newBase = camDevice.virtualDeviceSwitchOverVideoZoomFactors.first?.doubleValue ?? 1.0
            self.baseZoomFactor = newBase
            
            self.session.addInput(input)
            
            try? camDevice.lockForConfiguration()
            camDevice.videoZoomFactor = newBase
            camDevice.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 30)
            camDevice.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 30)
            camDevice.unlockForConfiguration()
            
            if let connection = self.output.connection(with: .video) {
                connection.videoRotationAngle = 90
                connection.isVideoMirrored = newPosition == .front
            }
            
            self.session.commitConfiguration()
            DispatchQueue.main.async {
                self.isFront = !self.isFront
                self.zoomLevel = newBase
            }
        }
    }
    
    private func startOrientationUpdates() {
        UIDevice.current.beginGeneratingDeviceOrientationNotifications()
        updateOrientationFromDevice()
        NotificationCenter.default.addObserver(forName: UIDevice.orientationDidChangeNotification, object: nil, queue: .main) { [weak self] _ in
            self?.updateOrientationFromDevice()
        }
    }
    
    private func updateOrientationFromDevice() {
        switch UIDevice.current.orientation {
            case .portrait: deviceOrientation = .up
            case .portraitUpsideDown: deviceOrientation = .down
            case .landscapeLeft: deviceOrientation = .left
            case .landscapeRight: deviceOrientation = .right
            default: break
        }
    }
    
    func start() { queue.async { self.session.startRunning() } }
    func stop() { queue.async { self.session.stopRunning() } }
    
    func capture() {
        captureOrientation = deviceOrientation
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
    
    private func smallTextureToImage(_ texture: MTLTexture) -> UIImage? {
        let w = texture.width, h = texture.height
        var pixels = [UInt8](repeating: 0, count: h * w * 4)
        texture.getBytes(&pixels, bytesPerRow: w * 4, from: MTLRegion(origin: .init(), size: .init(width: w, height: h, depth: 1)), mipmapLevel: 0)
        guard let ctx = CGContext(data: &pixels, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue),
        let cgImage = ctx.makeImage() else { return nil }
        return UIImage(cgImage: cgImage)
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
        
        // Update background snapshot
        let w = backgroundWidth, h = backgroundHeight
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: w, height: h, mipmapped: false)
        desc.usage = [.shaderRead, .shaderWrite]
        if let dst = device.makeTexture(descriptor: desc),
            let cmdBuffer = commandQueue.makeCommandBuffer(),
            let encoder = cmdBuffer.makeComputeCommandEncoder() {
            var b = Int32(bits)
            encoder.setComputePipelineState(downsamplePipeline)
            encoder.setTexture(texture, index: 0)
            encoder.setTexture(dst, index: 1)
            encoder.setBytes(&b, length: 4, index: 0)
            encoder.dispatchThreads(MTLSize(width: w, height: h, depth: 1), threadsPerThreadgroup: MTLSize(width: w, height: h, depth: 1))
            encoder.endEncoding()
            cmdBuffer.commit()
            cmdBuffer.waitUntilCompleted()
            backgroundSnapshot = smallTextureToImage(dst)
        }
        
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
