import AVFoundation
import UIKit
import Combine
import Metal
import MetalKit

let hasHomeButton: Bool = {
    guard let scene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
          let window = scene.windows.first else { return false }
    return window.safeAreaInsets.bottom == 0
}()

class CameraManager: NSObject, ObservableObject {
    @Published var capturedImage: UIImage?
    @Published var isFront = false
    @Published var bitsPerPixel = 8
    @Published var deviceOrientation: UIImage.Orientation = .up
    @Published var ditherEnabled = false
    @Published var zoomLevel: CGFloat = 1.0
    @Published var adaptivePaletteEnabled = false
    
    var backgroundWidth = 12
    var backgroundHeight = 16
    var backgroundSnapshot: UIImage?
    
    private var currentDevice: AVCaptureDevice?
    private(set) var baseZoomFactor: CGFloat = 1.0
    private var backCameraMaxOpticalZoom: CGFloat = 1.0
    
    var displayZoom: CGFloat { zoomLevel / baseZoomFactor }
    
    private let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "camera")
    private let sessionQueue = DispatchQueue(label: "camera.session", qos: .userInitiated)
    private var isFlipping = false
    private var shouldCapture = false
    private var captureOrientation: UIImage.Orientation = .up
    private var captureIsFront = false
    
    // Metal
    let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let computePipeline: MTLComputePipelineState
    private let downsamplePipeline: MTLComputePipelineState
    private let renderPipeline: MTLRenderPipelineState
    private let textureCache: CVMetalTextureCache
    private var processedTexture: MTLTexture?
    
    // Adaptive palette pipelines
    private let downsampleForPalettePipeline: MTLComputePipelineState
    private let clearHistogramPipeline: MTLComputePipelineState
    private let buildHistogramPipeline: MTLComputePipelineState
    private let prefixSumCopyPipeline: MTLComputePipelineState
    private let prefixSumPassRPipeline: MTLComputePipelineState
    private let prefixSumPassGPipeline: MTLComputePipelineState
    private let prefixSumPassBPipeline: MTLComputePipelineState
    private let initColorBoxPipeline: MTLComputePipelineState
    private let medianCutCompletePipeline: MTLComputePipelineState
    private let computePaletteColorsPipeline: MTLComputePipelineState
    private let buildPaletteLUTPipeline: MTLComputePipelineState
    private let applyAdaptivePalettePipeline: MTLComputePipelineState
    
    // Adaptive palette buffers and textures
    private var downsampledTexture: MTLTexture?  // 1/4 size texture for palette computation
    private var histogramBuffer: MTLBuffer?
    private var prefixSumBuffer: MTLBuffer?  // 3D prefix sum for O(1) box queries
    private var colorBoxBuffer: MTLBuffer?
    private var paletteBuffer: MTLBuffer?
    private var paletteLUTBuffer: MTLBuffer?  // 32x32x32 LUT for fast palette lookup
    private let histogramSize = 32 * 32 * 32
    private let maxPaletteColors = 256
    
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
        
        // Adaptive palette pipelines
        guard let downsampleForPaletteKernel = library.makeFunction(name: "downsampleForPalette"),
              let downsampleForPalettePipeline = try? device.makeComputePipelineState(function: downsampleForPaletteKernel),
              let clearHistogramKernel = library.makeFunction(name: "clearHistogram"),
              let clearHistogramPipeline = try? device.makeComputePipelineState(function: clearHistogramKernel),
              let buildHistogramKernel = library.makeFunction(name: "buildHistogram"),
              let buildHistogramPipeline = try? device.makeComputePipelineState(function: buildHistogramKernel),
              let prefixSumCopyKernel = library.makeFunction(name: "prefixSumCopy"),
              let prefixSumCopyPipeline = try? device.makeComputePipelineState(function: prefixSumCopyKernel),
              let prefixSumPassRKernel = library.makeFunction(name: "prefixSumPassR"),
              let prefixSumPassRPipeline = try? device.makeComputePipelineState(function: prefixSumPassRKernel),
              let prefixSumPassGKernel = library.makeFunction(name: "prefixSumPassG"),
              let prefixSumPassGPipeline = try? device.makeComputePipelineState(function: prefixSumPassGKernel),
              let prefixSumPassBKernel = library.makeFunction(name: "prefixSumPassB"),
              let prefixSumPassBPipeline = try? device.makeComputePipelineState(function: prefixSumPassBKernel),
              let initColorBoxKernel = library.makeFunction(name: "initColorBox"),
              let initColorBoxPipeline = try? device.makeComputePipelineState(function: initColorBoxKernel),
              let medianCutCompleteKernel = library.makeFunction(name: "medianCutComplete"),
              let medianCutCompletePipeline = try? device.makeComputePipelineState(function: medianCutCompleteKernel),
              let computePaletteColorsKernel = library.makeFunction(name: "computePaletteColors"),
              let computePaletteColorsPipeline = try? device.makeComputePipelineState(function: computePaletteColorsKernel),
              let buildPaletteLUTKernel = library.makeFunction(name: "buildPaletteLUT"),
              let buildPaletteLUTPipeline = try? device.makeComputePipelineState(function: buildPaletteLUTKernel),
              let applyAdaptivePaletteKernel = library.makeFunction(name: "applyAdaptivePalette"),
              let applyAdaptivePalettePipeline = try? device.makeComputePipelineState(function: applyAdaptivePaletteKernel) else {
            fatalError("Adaptive palette pipeline init failed")
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
        
        // Adaptive palette pipelines
        self.downsampleForPalettePipeline = downsampleForPalettePipeline
        self.clearHistogramPipeline = clearHistogramPipeline
        self.buildHistogramPipeline = buildHistogramPipeline
        self.prefixSumCopyPipeline = prefixSumCopyPipeline
        self.prefixSumPassRPipeline = prefixSumPassRPipeline
        self.prefixSumPassGPipeline = prefixSumPassGPipeline
        self.prefixSumPassBPipeline = prefixSumPassBPipeline
        self.initColorBoxPipeline = initColorBoxPipeline
        self.medianCutCompletePipeline = medianCutCompletePipeline
        self.computePaletteColorsPipeline = computePaletteColorsPipeline
        self.buildPaletteLUTPipeline = buildPaletteLUTPipeline
        self.applyAdaptivePalettePipeline = applyAdaptivePalettePipeline
        
        super.init()
        setupSession()
        startOrientationUpdates()
        setupAdaptivePaletteBuffers()
    }
    
    private func setupAdaptivePaletteBuffers() {
        // Histogram buffer: 32x32x32 uint values
        histogramBuffer = device.makeBuffer(length: histogramSize * MemoryLayout<UInt32>.size, options: .storageModeShared)
        
        // 3D prefix sum buffer: same size as histogram (32x32x32 uint values)
        prefixSumBuffer = device.makeBuffer(length: histogramSize * MemoryLayout<UInt32>.size, options: .storageModeShared)
        
        // Color box buffer: up to 256 boxes (for 256 colors max)
        // Each ColorBox is 32 bytes (uint3 minC, uint3 maxC, uint count, float priority)
        colorBoxBuffer = device.makeBuffer(length: maxPaletteColors * 32, options: .storageModeShared)
        
        // Palette buffer: up to 256 float3 colors (each float3 is 16 bytes aligned)
        paletteBuffer = device.makeBuffer(length: maxPaletteColors * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)
        
        // LUT buffer: 32x32x32 entries, each containing 8 float3 (8 nearest colors)
        // PaletteLUTEntry is 128 bytes (8 float3, each padded to 16 bytes in Metal)
        paletteLUTBuffer = device.makeBuffer(length: histogramSize * 128, options: .storageModeShared)
    }
    
    private func setupSession() {
        session.beginConfiguration()
        session.sessionPreset = .vga640x480
        
        let position: AVCaptureDevice.Position = isFront ? .front : .back
        guard let camDevice = bestCamera(for: position),
              let input = try? AVCaptureDeviceInput(device: camDevice) else {
            session.commitConfiguration()
            return
        }
        
        currentDevice = camDevice
        baseZoomFactor = calculateBaseZoomFactor(for: camDevice)
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
            connection.videoOrientation = .portrait
            connection.isVideoMirrored = isFront
        }
        
        session.commitConfiguration()
    }
    
    private func bestCamera(for position: AVCaptureDevice.Position) -> AVCaptureDevice? {
        let types: [AVCaptureDevice.DeviceType] = position == .back
            ? [.builtInTripleCamera, .builtInDualWideCamera, .builtInDualCamera, .builtInWideAngleCamera]
            : [.builtInWideAngleCamera]
        return AVCaptureDevice.DiscoverySession(deviceTypes: types, mediaType: .video, position: position).devices.first
    }
    
    private func hasUltraWideCamera(_ device: AVCaptureDevice) -> Bool {
        switch device.deviceType {
        case .builtInTripleCamera, .builtInDualWideCamera:
            return true
        default:
            return false
        }
    }
    
    private func calculateBaseZoomFactor(for device: AVCaptureDevice) -> CGFloat {
        if hasUltraWideCamera(device) {
            return device.virtualDeviceSwitchOverVideoZoomFactors.first?.doubleValue ?? 1.0
        } else {
            return 1.0
        }
    }
    
    private var maxZoomFactor: CGFloat {
        guard let device = currentDevice else { return 1.0 }
        let limit = isFront ? 5.0 : Double((5.0 * backCameraMaxOpticalZoom).rounded())
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
        guard !isFlipping else { return }
        isFlipping = true
        
        sessionQueue.async {
            self.session.beginConfiguration()
            
            if let current = self.session.inputs.first as? AVCaptureDeviceInput {
                self.session.removeInput(current)
            }
            
            let newPosition: AVCaptureDevice.Position = self.isFront ? .back : .front
            guard let camDevice = self.bestCamera(for: newPosition),
                  let input = try? AVCaptureDeviceInput(device: camDevice),
                  self.session.canAddInput(input) else {
                self.session.commitConfiguration()
                self.isFlipping = false
                return
            }
            
            self.currentDevice = camDevice
            let newBase = self.calculateBaseZoomFactor(for: camDevice)
            self.baseZoomFactor = newBase
            
            self.session.addInput(input)
            
            try? camDevice.lockForConfiguration()
            camDevice.videoZoomFactor = newBase
            camDevice.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 30)
            camDevice.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 30)
            camDevice.unlockForConfiguration()
            
            if let connection = self.output.connection(with: .video) {
                connection.videoOrientation = .portrait
                connection.isVideoMirrored = newPosition == .front
            }
            
            self.session.commitConfiguration()
            self.isFlipping = false
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
    
    func start() { sessionQueue.async { self.session.startRunning() } }
    func stop() { sessionQueue.async { self.session.stopRunning() } }
    
    func capture() {
        captureOrientation = deviceOrientation
        captureIsFront = isFront
        shouldCapture = true
    }
    
    private func processToTexture(_ pixelBuffer: CVPixelBuffer, bits: Int, dither: Bool, adaptivePalette: Bool) -> MTLTexture? {
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
        
        // Use adaptive palette for bits <= 8, otherwise use standard quantization
        if adaptivePalette && bits <= 8 {
            return processWithAdaptivePalette(inTexture: inTexture, outTexture: outTex, bits: bits, dither: dither, width: width, height: height)
        }
        
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
    
    private func processWithAdaptivePalette(inTexture: MTLTexture, outTexture: MTLTexture, bits: Int, dither: Bool, width: Int, height: Int) -> MTLTexture? {
        guard let histogramBuffer = histogramBuffer,
              let prefixSumBuffer = prefixSumBuffer,
              let colorBoxBuffer = colorBoxBuffer,
              let paletteBuffer = paletteBuffer,
              let paletteLUTBuffer = paletteLUTBuffer else { return nil }
        
        let paletteSize = 1 << bits
        let tgSize = MTLSize(width: 16, height: 16, depth: 1)
        let tgCount = MTLSize(width: (width + 15) / 16, height: (height + 15) / 16, depth: 1)
        
        // Downsampled dimensions (2x smaller in each dimension = 4x fewer pixels)
        let dsWidth = width / 2
        let dsHeight = height / 2
        let dsTgCount = MTLSize(width: (dsWidth + 15) / 16, height: (dsHeight + 15) / 16, depth: 1)
        
        // Create/resize downsampled texture if needed
        if downsampledTexture == nil || downsampledTexture!.width != dsWidth || downsampledTexture!.height != dsHeight {
            let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: dsWidth, height: dsHeight, mipmapped: false)
            desc.usage = [.shaderRead, .shaderWrite]
            downsampledTexture = device.makeTexture(descriptor: desc)
        }
        guard let dsTexture = downsampledTexture else { return nil }
        
        var paletteSizeVal = Int32(paletteSize)
        var ditherVal: Int32 = dither ? 1 : 0
        
        // ========== BATCH A: Downsample + Clear + Build Histogram ==========
        if let cmdBuffer = commandQueue.makeCommandBuffer() {
            // Downsample input for palette computation (4x reduction)
            if let enc = cmdBuffer.makeComputeCommandEncoder() {
                enc.setComputePipelineState(downsampleForPalettePipeline)
                enc.setTexture(inTexture, index: 0)
                enc.setTexture(dsTexture, index: 1)
                enc.dispatchThreadgroups(dsTgCount, threadsPerThreadgroup: tgSize)
                enc.endEncoding()
            }
            
            // Clear histogram using blit encoder (faster than compute)
            if let blitEnc = cmdBuffer.makeBlitCommandEncoder() {
                blitEnc.fill(buffer: histogramBuffer, range: 0..<histogramBuffer.length, value: 0)
                blitEnc.endEncoding()
            }
            
            // Build histogram from downsampled texture
            if let enc = cmdBuffer.makeComputeCommandEncoder() {
                enc.setComputePipelineState(buildHistogramPipeline)
                enc.setTexture(dsTexture, index: 0)
                enc.setBuffer(histogramBuffer, offset: 0, index: 0)
                enc.dispatchThreadgroups(dsTgCount, threadsPerThreadgroup: tgSize)
                enc.endEncoding()
            }
            
            cmdBuffer.commit()
            cmdBuffer.waitUntilCompleted()
        }
        
        // ========== BATCH B: Prefix Sum (4 passes with memory barriers) ==========
        if let cmdBuffer = commandQueue.makeCommandBuffer() {
            // Pass 1: Copy histogram to prefix buffer
            if let enc = cmdBuffer.makeComputeCommandEncoder() {
                enc.setComputePipelineState(prefixSumCopyPipeline)
                enc.setBuffer(histogramBuffer, offset: 0, index: 0)
                enc.setBuffer(prefixSumBuffer, offset: 0, index: 1)
                enc.dispatchThreads(MTLSize(width: histogramSize, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                enc.memoryBarrier(scope: .buffers)
                enc.endEncoding()
            }
            
            // Pass 2: Prefix sum along R axis
            if let enc = cmdBuffer.makeComputeCommandEncoder() {
                enc.setComputePipelineState(prefixSumPassRPipeline)
                enc.setBuffer(prefixSumBuffer, offset: 0, index: 0)
                enc.dispatchThreads(MTLSize(width: 1024, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                enc.memoryBarrier(scope: .buffers)
                enc.endEncoding()
            }
            
            // Pass 3: Prefix sum along G axis
            if let enc = cmdBuffer.makeComputeCommandEncoder() {
                enc.setComputePipelineState(prefixSumPassGPipeline)
                enc.setBuffer(prefixSumBuffer, offset: 0, index: 0)
                enc.dispatchThreads(MTLSize(width: 1024, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                enc.memoryBarrier(scope: .buffers)
                enc.endEncoding()
            }
            
            // Pass 4: Prefix sum along B axis
            if let enc = cmdBuffer.makeComputeCommandEncoder() {
                enc.setComputePipelineState(prefixSumPassBPipeline)
                enc.setBuffer(prefixSumBuffer, offset: 0, index: 0)
                enc.dispatchThreads(MTLSize(width: 1024, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                enc.endEncoding()
            }
            
            cmdBuffer.commit()
            cmdBuffer.waitUntilCompleted()
        }
        
        // ========== BATCH C: Init Color Box + Complete Median Cut (single GPU dispatch) ==========
        if let cmdBuffer = commandQueue.makeCommandBuffer() {
            // Initialize first color box
            if let enc = cmdBuffer.makeComputeCommandEncoder() {
                enc.setComputePipelineState(initColorBoxPipeline)
                enc.setBuffer(colorBoxBuffer, offset: 0, index: 0)
                enc.setBuffer(histogramBuffer, offset: 0, index: 1)
                enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                enc.memoryBarrier(scope: .buffers)
                enc.endEncoding()
            }
            
            // Complete median cut on GPU with parallel reduction
            // Runs entire loop on GPU, eliminating all CPU-GPU synchronization
            if let enc = cmdBuffer.makeComputeCommandEncoder() {
                enc.setComputePipelineState(medianCutCompletePipeline)
                enc.setBuffer(colorBoxBuffer, offset: 0, index: 0)
                enc.setBuffer(prefixSumBuffer, offset: 0, index: 1)
                enc.setBytes(&paletteSizeVal, length: 4, index: 2)
                // Dispatch single threadgroup of 256 threads for parallel reduction
                enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                enc.endEncoding()
            }
            
            cmdBuffer.commit()
            cmdBuffer.waitUntilCompleted()
        }
        
        // ========== BATCH D: Compute Palette + Build LUT + Apply ==========
        if let cmdBuffer = commandQueue.makeCommandBuffer() {
            // Compute palette colors
            if let enc = cmdBuffer.makeComputeCommandEncoder() {
                enc.setComputePipelineState(computePaletteColorsPipeline)
                enc.setBuffer(colorBoxBuffer, offset: 0, index: 0)
                enc.setBuffer(histogramBuffer, offset: 0, index: 1)
                enc.setBuffer(paletteBuffer, offset: 0, index: 2)
                enc.setBytes(&paletteSizeVal, length: 4, index: 3)
                enc.dispatchThreads(MTLSize(width: paletteSize, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: min(paletteSize, 256), height: 1, depth: 1))
                enc.memoryBarrier(scope: .buffers)
                enc.endEncoding()
            }
            
            // Build LUT
            if let enc = cmdBuffer.makeComputeCommandEncoder() {
                enc.setComputePipelineState(buildPaletteLUTPipeline)
                enc.setBuffer(paletteLUTBuffer, offset: 0, index: 0)
                enc.setBuffer(paletteBuffer, offset: 0, index: 1)
                enc.setBytes(&paletteSizeVal, length: 4, index: 2)
                enc.dispatchThreads(MTLSize(width: histogramSize, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                enc.memoryBarrier(scope: .buffers)
                enc.endEncoding()
            }
            
            // Apply palette using LUT lookup
            if let enc = cmdBuffer.makeComputeCommandEncoder() {
                enc.setComputePipelineState(applyAdaptivePalettePipeline)
                enc.setTexture(inTexture, index: 0)
                enc.setTexture(outTexture, index: 1)
                enc.setBuffer(paletteLUTBuffer, offset: 0, index: 0)
                enc.setBytes(&ditherVal, length: 4, index: 1)
                enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
                enc.endEncoding()
            }
            
            cmdBuffer.commit()
            cmdBuffer.waitUntilCompleted()
        }
        
        return outTexture
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
        guard let cgImage = image.cgImage else { return image }
        if captureOrientation == .up && !captureIsFront { return image }
        
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
        default: break
        }
        
        if captureIsFront {
            transform = transform.translatedBy(x: captureOrientation == .left || captureOrientation == .right ? 0 : CGFloat(width), y: captureOrientation == .left || captureOrientation == .right ? CGFloat(height) : 0)
            transform = transform.scaledBy(x: captureOrientation == .left || captureOrientation == .right ? 1 : -1, y: captureOrientation == .left || captureOrientation == .right ? -1 : 1)
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
        // Skip processing during camera flip to avoid queue backup
        guard !isFlipping else { return }
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        let bits = bitsPerPixel
        let dither = ditherEnabled && bits < 24
        let capture = shouldCapture
        let adaptivePalette = adaptivePaletteEnabled && bits <= 8
        
        guard let texture = processToTexture(pb, bits: bits, dither: dither, adaptivePalette: adaptivePalette) else { return }
        
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
