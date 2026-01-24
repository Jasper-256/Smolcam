import Foundation
import ImageIO
import Compression
import CoreGraphics

nonisolated func imageDataWithMetadata(_ cgImage: CGImage, pixelBits: Int, ditherMode: Int) -> Data? {
    let exifText = "Smolcam | \(pixelBits)-bit | \(ditherModeNames[ditherMode])"
    
    // Use indexed PNG for 8 bits or less (256 colors max)
    if pixelBits <= 8 {
        return createIndexedPNG(cgImage, bitDepth: pixelBits <= 4 ? 4 : 8, exifText: exifText)
    }
    
    // Use standard PNG for higher bit depths (strip alpha)
    let w = cgImage.width, h = cgImage.height
    guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue),
        let rgbImage = (ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h)), ctx.makeImage()).1 else { return nil }
    
    let data = NSMutableData()
    guard let dest = CGImageDestinationCreateWithData(data, "public.png" as CFString, 1, nil) else { return nil }
    let metadata: [String: Any] = [
        kCGImagePropertyExifDictionary as String: [kCGImagePropertyExifLensModel as String: exifText]
    ]
    CGImageDestinationAddImage(dest, rgbImage, metadata as CFDictionary)
    CGImageDestinationFinalize(dest)
    return data as Data
}

nonisolated private func createIndexedPNG(_ cgImage: CGImage, bitDepth: Int, exifText: String) -> Data? {
    let w = cgImage.width, h = cgImage.height
    guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                               space: CGColorSpaceCreateDeviceRGB(),
                               bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue),
          let ptr = ctx.data else { return nil }
    ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))
    let pixels = ptr.assumingMemoryBound(to: UInt8.self)
    
    // Build palette from unique colors
    var colorToIndex = [UInt32: UInt8]()
    var palette = [UInt8]()
    for i in 0..<(w * h) {
        let rgb = UInt32(pixels[i * 4]) << 16 | UInt32(pixels[i * 4 + 1]) << 8 | UInt32(pixels[i * 4 + 2])
        if colorToIndex[rgb] == nil {
            colorToIndex[rgb] = UInt8(palette.count / 3)
            palette += [pixels[i * 4], pixels[i * 4 + 1], pixels[i * 4 + 2]]
        }
    }
    
    // Create indexed pixel data with filter byte per row
    var raw = Data(capacity: h * (1 + w))
    for y in 0..<h {
        raw.append(0)  // filter: none
        if bitDepth == 8 {
            for x in 0..<w {
                let rgb = UInt32(pixels[(y * w + x) * 4]) << 16 | UInt32(pixels[(y * w + x) * 4 + 1]) << 8 | UInt32(pixels[(y * w + x) * 4 + 2])
                raw.append(colorToIndex[rgb]!)
            }
        } else {
            var byte: UInt8 = 0, bits = 0
            for x in 0..<w {
                let rgb = UInt32(pixels[(y * w + x) * 4]) << 16 | UInt32(pixels[(y * w + x) * 4 + 1]) << 8 | UInt32(pixels[(y * w + x) * 4 + 2])
                let idx = colorToIndex[rgb]!
                byte = (byte << bitDepth) | (idx & UInt8((1 << bitDepth) - 1))
                bits += bitDepth
                if bits == 8 { raw.append(byte); byte = 0; bits = 0 }
            }
            if bits > 0 { raw.append(byte << (8 - bits)) }
        }
    }
    
    // Compress with zlib
    guard let compressed = deflate(raw) else { return nil }
    
    // Build PNG
    var png = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])  // signature
    png.append(pngChunk("IHDR", data: Data([
        UInt8(w >> 24 & 0xFF), UInt8(w >> 16 & 0xFF), UInt8(w >> 8 & 0xFF), UInt8(w & 0xFF),
        UInt8(h >> 24 & 0xFF), UInt8(h >> 16 & 0xFF), UInt8(h >> 8 & 0xFF), UInt8(h & 0xFF),
        UInt8(bitDepth), 3, 0, 0, 0  // bitDepth, colorType=indexed, compression, filter, interlace
    ])))
    png.append(pngChunk("PLTE", data: Data(palette)))
    png.append(pngChunk("eXIf", data: buildExifWithLensModel(exifText)))
    png.append(pngChunk("IDAT", data: compressed))
    png.append(pngChunk("IEND", data: Data()))
    return png
}

nonisolated private func buildExifWithLensModel(_ lensModel: String) -> Data {
    let str = Data(lensModel.utf8) + Data([0])  // null-terminated
    let strLen = UInt32(str.count)
    // TIFF header (big-endian) + IFD0 (1 entry: ExifIFDPointer) + Exif IFD (1 entry: LensModel) + string
    let exifIFDOffset: UInt32 = 8 + 2 + 12 + 4  // after TIFF header + IFD0
    let strOffset: UInt32 = exifIFDOffset + 2 + 12 + 4  // after Exif IFD
    var exif = Data()
    exif.append(contentsOf: [0x4D, 0x4D, 0x00, 0x2A])  // "MM" + magic (big-endian)
    exif.append(contentsOf: beUInt32(8))  // offset to IFD0
    // IFD0: 1 entry
    exif.append(contentsOf: beUInt16(1))  // entry count
    exif.append(contentsOf: beUInt16(0x8769))  // ExifIFDPointer tag
    exif.append(contentsOf: beUInt16(4))  // type: LONG
    exif.append(contentsOf: beUInt32(1))  // count
    exif.append(contentsOf: beUInt32(exifIFDOffset))  // value: offset to Exif IFD
    exif.append(contentsOf: beUInt32(0))  // next IFD offset (none)
    // Exif IFD: 1 entry
    exif.append(contentsOf: beUInt16(1))  // entry count
    exif.append(contentsOf: beUInt16(0xA434))  // LensModel tag
    exif.append(contentsOf: beUInt16(2))  // type: ASCII
    exif.append(contentsOf: beUInt32(strLen))  // count
    if strLen <= 4 {
        var val = str; while val.count < 4 { val.append(0) }
        exif.append(val)
    } else {
        exif.append(contentsOf: beUInt32(strOffset))
    }
    exif.append(contentsOf: beUInt32(0))  // next IFD offset (none)
    if strLen > 4 { exif.append(str) }
    return exif
}

nonisolated private func beUInt16(_ v: UInt16) -> [UInt8] { [UInt8(v >> 8 & 0xFF), UInt8(v & 0xFF)] }
nonisolated private func beUInt32(_ v: UInt32) -> [UInt8] { [UInt8(v >> 24 & 0xFF), UInt8(v >> 16 & 0xFF), UInt8(v >> 8 & 0xFF), UInt8(v & 0xFF)] }

nonisolated private func pngChunk(_ type: String, data: Data) -> Data {
    var chunk = Data()
    let len = UInt32(data.count)
    chunk.append(contentsOf: [UInt8(len >> 24 & 0xFF), UInt8(len >> 16 & 0xFF), UInt8(len >> 8 & 0xFF), UInt8(len & 0xFF)])
    let typeData = Data(type.utf8)
    chunk.append(typeData)
    chunk.append(data)
    let crc = crc32(typeData + data)
    chunk.append(contentsOf: [UInt8(crc >> 24 & 0xFF), UInt8(crc >> 16 & 0xFF), UInt8(crc >> 8 & 0xFF), UInt8(crc & 0xFF)])
    return chunk
}

nonisolated private func crc32(_ data: Data) -> UInt32 {
    var crc: UInt32 = 0xFFFFFFFF
    for byte in data {
        crc ^= UInt32(byte)
        for _ in 0..<8 { crc = (crc >> 1) ^ (crc & 1 == 1 ? 0xEDB88320 : 0) }
    }
    return ~crc
}

nonisolated private func deflate(_ data: Data) -> Data? {
    let destSize = data.count + 1024
    var dest = Data(count: destSize)
    let result = dest.withUnsafeMutableBytes { destPtr in
        data.withUnsafeBytes { srcPtr in
            compression_encode_buffer(destPtr.bindMemory(to: UInt8.self).baseAddress!, destSize,
                                      srcPtr.bindMemory(to: UInt8.self).baseAddress!, data.count,
                                      nil, COMPRESSION_ZLIB)
        }
    }
    guard result > 0 else { return nil }
    // Wrap in zlib format: header + data + adler32
    var zlib = Data([0x78, 0x9C])  // zlib header (default compression)
    zlib.append(dest.prefix(result))
    let adler = adler32(data)
    zlib.append(contentsOf: [UInt8(adler >> 24 & 0xFF), UInt8(adler >> 16 & 0xFF), UInt8(adler >> 8 & 0xFF), UInt8(adler & 0xFF)])
    return zlib
}

nonisolated private func adler32(_ data: Data) -> UInt32 {
    var a: UInt32 = 1, b: UInt32 = 0
    for byte in data { a = (a + UInt32(byte)) % 65521; b = (b + a) % 65521 }
    return (b << 16) | a
}
