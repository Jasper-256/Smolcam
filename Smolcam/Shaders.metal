#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
};

vertex VertexOut vertexPassthrough(uint vid [[vertex_id]]) {
    const float2 positions[4] = { {-1,-1}, {1,-1}, {-1,1}, {1,1} };
    const float2 texCoords[4] = { {0,1}, {1,1}, {0,0}, {1,0} };
    VertexOut out;
    out.position = float4(positions[vid], 0, 1);
    out.texCoord = texCoords[vid];
    return out;
}

fragment float4 fragmentPassthrough(VertexOut in [[stage_in]],
        texture2d<float> tex [[texture(0)]]) {
    constexpr sampler s(filter::linear);
    return tex.sample(s, in.texCoord);
}

inline float3 getMaxLevels(int bitsPerPixel) {
    if (bitsPerPixel == 8) return float3(7.0, 7.0, 3.0);
    if (bitsPerPixel == 16) return float3(31.0, 63.0, 31.0);
    return float3((1 << (bitsPerPixel / 3)) - 1);
}

constant float bayer16x16[256] = {
      0,128, 32,160,  8,136, 40,168,  2,130, 34,162, 10,138, 42,170,
    192, 64,224, 96,200, 72,232,104,194, 66,226, 98,202, 74,234,106,
     48,176, 16,144, 56,184, 24,152, 50,178, 18,146, 58,186, 26,154,
    240,112,208, 80,248,120,216, 88,242,114,210, 82,250,122,218, 90,
     12,140, 44,172,  4,132, 36,164, 14,142, 46,174,  6,134, 38,166,
    204, 76,236,108,196, 68,228,100,206, 78,238,110,198, 70,230,102,
     60,188, 28,156, 52,180, 20,148, 62,190, 30,158, 54,182, 22,150,
    252,124,220, 92,244,116,212, 84,254,126,222, 94,246,118,214, 86,
      3,131, 35,163, 11,139, 43,171,  1,129, 33,161,  9,137, 41,169,
    195, 67,227, 99,203, 75,235,107,193, 65,225, 97,201, 73,233,105,
     51,179, 19,147, 59,187, 27,155, 49,177, 17,145, 57,185, 25,153,
    243,115,211, 83,251,123,219, 91,241,113,209, 81,249,121,217, 89,
     15,143, 47,175,  7,135, 39,167, 13,141, 45,173,  5,133, 37,165,
    207, 79,239,111,199, 71,231,103,205, 77,237,109,197, 69,229,101,
     63,191, 31,159, 55,183, 23,151, 61,189, 29,157, 53,181, 21,149,
    255,127,223, 95,247,119,215, 87,253,125,221, 93,245,117,213, 85,
};

// sRGB <-> Linear conversion functions for perceptually correct dithering
inline float srgbToLinear(float c) {
    return (c <= 0.04045) ? (c / 12.92) : pow((c + 0.055) / 1.055, 2.4);
}

inline float linearToSrgb(float c) {
    return (c <= 0.0031308) ? (c * 12.92) : (1.055 * pow(c, 1.0/2.4) - 0.055);
}

inline float3 srgbToLinear3(float3 c) {
    return float3(srgbToLinear(c.r), srgbToLinear(c.g), srgbToLinear(c.b));
}

inline float3 linearToSrgb3(float3 c) {
    return float3(linearToSrgb(c.r), linearToSrgb(c.g), linearToSrgb(c.b));
}

kernel void ditherQuantize(
    texture2d<float, access::read> inTex [[texture(0)]],
    texture2d<float, access::write> outTex [[texture(1)]],
    constant int &bitsPerPixel [[buffer(0)]],
    constant int &dither [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= inTex.get_width() || gid.y >= inTex.get_height()) return;
    
    float4 color = inTex.read(gid);
    float3 maxLevels = getMaxLevels(bitsPerPixel);
    
    if (dither != 0) {
        uint idx = (gid.y % 16) * 16 + (gid.x % 16);
        float threshold = bayer16x16[idx] / 256.0;
        color.rgb = floor(color.rgb * maxLevels + threshold * 0.9) / maxLevels;
    } else {
        color.rgb = floor(color.rgb * maxLevels + 0.5) / maxLevels;
    }
    
    outTex.write(float4(saturate(color.rgb), color.a), gid);
}

kernel void downsampleQuantize(
    texture2d<float, access::read> inTex [[texture(0)]],
    texture2d<float, access::write> outTex [[texture(1)]],
    constant int &bitsPerPixel [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint outW = outTex.get_width(), outH = outTex.get_height();
    if (gid.x >= outW || gid.y >= outH) return;
    
    uint inW = inTex.get_width(), inH = inTex.get_height();
    uint x0 = (gid.x * inW) / outW, x1 = ((gid.x + 1) * inW) / outW;
    uint y0 = (gid.y * inH) / outH, y1 = ((gid.y + 1) * inH) / outH;
    
    float3 sum = float3(0);
    for (uint y = y0; y < y1; y++)
        for (uint x = x0; x < x1; x++)
            sum += inTex.read(uint2(x, y)).rgb;
    
    float3 avg = sum / float((x1 - x0) * (y1 - y0));
    float3 maxLevels = getMaxLevels(bitsPerPixel);
    
    uint idx = (gid.y % 16) * 16 + (gid.x % 16);
    float threshold = bayer16x16[idx] / 256.0;
    float3 quantized = floor(avg * maxLevels + threshold) / maxLevels;

    outTex.write(float4(quantized, 1.0), gid);
}

// MARK: - Adaptive Palette Shaders

constant int HIST_SIZE = 32;
constant int HIST_TOTAL = HIST_SIZE * HIST_SIZE * HIST_SIZE;

// Downsample image by 2x in each dimension for faster palette computation
kernel void downsampleForPalette(
    texture2d<float, access::read> inTex [[texture(0)]],
    texture2d<float, access::write> outTex [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= outTex.get_width() || gid.y >= outTex.get_height()) return;
    
    // Average 2x2 block from input
    uint2 base = gid * 2;
    float3 sum = inTex.read(base).rgb
               + inTex.read(base + uint2(1, 0)).rgb
               + inTex.read(base + uint2(0, 1)).rgb
               + inTex.read(base + uint2(1, 1)).rgb;
    outTex.write(float4(sum / 4.0, 1.0), gid);
}

kernel void clearHistogram(device atomic_uint *histogram [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    if (tid < uint(HIST_TOTAL)) atomic_store_explicit(&histogram[tid], 0, memory_order_relaxed);
}

// Build histogram with saturation weighting
// Saturated colors get more weight so they're better represented in the palette,
// even if they occupy a smaller portion of the image
kernel void buildHistogram(texture2d<float, access::read> inTex [[texture(0)]], device atomic_uint *histogram [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= inTex.get_width() || gid.y >= inTex.get_height()) return;
    float4 c = inTex.read(gid);
    
    // Compute saturation (HSV model) for weighting
    float maxC = max(c.r, max(c.g, c.b));
    float minC = min(c.r, min(c.g, c.b));
    float saturation = (maxC > 0.001) ? (maxC - minC) / maxC : 0.0;
    
    // Weight: base 1 + up to 3x boost for saturated colors
    // Fully saturated colors get 4x weight, grays get 1x
    uint weight = uint(1.0 + saturation * 3.0);
    
    uint3 b = uint3(saturate(c.rgb) * float(HIST_SIZE - 1) + 0.5);
    atomic_fetch_add_explicit(&histogram[min(b.b, 31u) * 1024 + min(b.g, 31u) * 32 + min(b.r, 31u)], weight, memory_order_relaxed);
}

struct ColorBox { uint3 minC, maxC; uint count, pad; };

// Initialize first box covering all colors
kernel void initColorBox(device ColorBox *boxes [[buffer(0)]], device const uint *hist [[buffer(1)]]) {
    uint3 lo = uint3(63), hi = uint3(0);
    uint total = 0;
    for (uint i = 0; i < HIST_TOTAL; i++) {
        uint c = hist[i];
        if (c > 0) {
            uint3 p = uint3(i % 32, (i / 32) % 32, i / 1024);
            lo = min(lo, p); hi = max(hi, p);
            total += c;
        }
    }
    boxes[0] = ColorBox{lo, hi, total, 0};
}

// Parallel 3D prefix sum - split into 4 kernels for parallelization
// Each pass runs 1024 threads in parallel instead of single-threaded

// Step 1: Parallel copy histogram to prefix buffer (32K threads)
kernel void prefixSumCopy(
    device const uint *hist [[buffer(0)]],
    device uint *prefix [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < uint(HIST_TOTAL)) {
        prefix[tid] = hist[tid];
    }
}

// Step 2: Prefix sum along R axis - 1024 threads, each handles one (b,g) row
kernel void prefixSumPassR(
    device uint *prefix [[buffer(0)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= 1024) return;
    uint b = tid / 32;
    uint g = tid % 32;
    uint base = b * 1024 + g * 32;
    for (uint r = 1; r < 32; r++) {
        prefix[base + r] += prefix[base + r - 1];
    }
}

// Step 3: Prefix sum along G axis - 1024 threads, each handles one (b,r) column
kernel void prefixSumPassG(
    device uint *prefix [[buffer(0)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= 1024) return;
    uint b = tid / 32;
    uint r = tid % 32;
    for (uint g = 1; g < 32; g++) {
        prefix[b * 1024 + g * 32 + r] += prefix[b * 1024 + (g - 1) * 32 + r];
    }
}

// Step 4: Prefix sum along B axis - 1024 threads, each handles one (g,r) depth
kernel void prefixSumPassB(
    device uint *prefix [[buffer(0)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= 1024) return;
    uint g = tid / 32;
    uint r = tid % 32;
    for (uint b = 1; b < 32; b++) {
        prefix[b * 1024 + g * 32 + r] += prefix[(b - 1) * 1024 + g * 32 + r];
    }
}

// Helper: query prefix sum, returns 0 for out-of-bounds (negative indices)
inline uint queryPrefix(device const uint *prefix, int r, int g, int b) {
    if (r < 0 || g < 0 || b < 0) return 0;
    return prefix[b * 1024 + g * 32 + r];
}

// Query sum within a 3D box using inclusion-exclusion principle
inline uint boxSum(device const uint *prefix, uint3 lo, uint3 hi) {
    int r0 = int(lo.r) - 1, g0 = int(lo.g) - 1, b0 = int(lo.b) - 1;
    int r1 = int(hi.r), g1 = int(hi.g), b1 = int(hi.b);
    
    return queryPrefix(prefix, r1, g1, b1)
         - queryPrefix(prefix, r0, g1, b1)
         - queryPrefix(prefix, r1, g0, b1)
         - queryPrefix(prefix, r1, g1, b0)
         + queryPrefix(prefix, r0, g0, b1)
         + queryPrefix(prefix, r0, g1, b0)
         + queryPrefix(prefix, r1, g0, b0)
         - queryPrefix(prefix, r0, g0, b0);
}

// Parallel median cut using 3D prefix sum for O(1) box queries
kernel void medianCutSplitParallel(
    device ColorBox *boxes [[buffer(0)]],
    device const uint *prefix [[buffer(1)]],  // Now uses prefix sum instead of histogram
    constant int &levelOffset [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= uint(levelOffset)) return;
    
    ColorBox box = boxes[tid];
    uint3 range = box.maxC - box.minC;
    
    // Find longest axis
    int axis = (range.g > range.r && range.g >= range.b) ? 1 : (range.b > range.r && range.b > range.g) ? 2 : 0;
    uint axisMin = (axis == 0) ? box.minC.r : (axis == 1) ? box.minC.g : box.minC.b;
    uint axisMax = (axis == 0) ? box.maxC.r : (axis == 1) ? box.maxC.g : box.maxC.b;
    
    // Can't split if range is 0
    if (axisMin >= axisMax) {
        boxes[tid + levelOffset] = box;
        return;
    }
    
    // Find median split point using prefix sum queries
    uint halfCount = box.count / 2;
    uint splitVal = axisMin + 1;
    
    // Binary search for the split point
    uint lo = axisMin, hi = axisMax;
    while (lo < hi) {
        uint mid = (lo + hi) / 2;
        
        // Count pixels in box with axis value <= mid
        uint3 queryLo = box.minC;
        uint3 queryHi = box.maxC;
        if (axis == 0) queryHi.r = mid;
        else if (axis == 1) queryHi.g = mid;
        else queryHi.b = mid;
        
        uint count = boxSum(prefix, queryLo, queryHi);
        
        if (count < halfCount) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    splitVal = max(lo, axisMin + 1);
    if (splitVal > axisMax) splitVal = axisMax;
    
    // Create child boxes
    ColorBox box1 = box, box2 = box;
    if (axis == 0) { box1.maxC.r = splitVal - 1; box2.minC.r = splitVal; }
    else if (axis == 1) { box1.maxC.g = splitVal - 1; box2.minC.g = splitVal; }
    else { box1.maxC.b = splitVal - 1; box2.minC.b = splitVal; }
    
    // Count pixels in each child using prefix sum (O(1) each)
    uint c1 = boxSum(prefix, box1.minC, box1.maxC);
    uint c2 = boxSum(prefix, box2.minC, box2.maxC);
    
    box1.count = c1;
    box2.count = c2;
    boxes[tid] = box1;
    boxes[tid + levelOffset] = box2;
}

// Compute palette color for each box
kernel void computePaletteColors(
    device const ColorBox *boxes [[buffer(0)]],
    device const uint *hist [[buffer(1)]],
    device float3 *palette [[buffer(2)]],
    constant int &numColors [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= uint(numColors)) return;
    ColorBox box = boxes[tid];
    
    float3 sum = float3(0);
    uint total = 0;
    for (uint b = box.minC.b; b <= box.maxC.b; b++) {
        for (uint g = box.minC.g; g <= box.maxC.g; g++) {
            for (uint r = box.minC.r; r <= box.maxC.r; r++) {
                uint c = hist[b * 1024 + g * 32 + r];
                if (c > 0) { sum += float3(r, g, b) * float(c); total += c; }
            }
        }
    }
    palette[tid] = (total > 0) ? sum / float(total * 31) : float3(box.minC + box.maxC) / 62.0;
}

// LUT entry: stores up to 8 nearest palette colors for smooth dithering
// Terminator value (-1, -1, -1) marks the end when fewer than 8 colors available
constant int LUT_CANDIDATES = 8;
constant float3 LUT_TERMINATOR = float3(-1.0, -1.0, -1.0);

struct PaletteLUTEntry {
    float3 colors[8];  // Up to 8 nearest palette colors, sorted by distance
};

// Check if a color is the terminator value
inline bool isTerminator(float3 color) {
    return color.r < 0.0;
}

// Build 32x32x32 LUT mapping each quantized RGB to up to 8 nearest palette colors
// Distance calculations done in linear space for perceptually correct results
// If palette has fewer than 8 colors, remaining slots are filled with terminator
kernel void buildPaletteLUT(
    device PaletteLUTEntry *lut [[buffer(0)]],
    device const float3 *palette [[buffer(1)]],
    constant int &paletteSize [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= uint(HIST_TOTAL)) return;
    
    // Convert linear index to RGB coordinates (same layout as histogram)
    float3 rgb = float3(tid % 32, (tid / 32) % 32, tid / 1024) / 31.0;
    
    // Convert to linear space for distance calculations
    float3 rgbLinear = srgbToLinear3(rgb);
    
    // How many candidates can we actually store?
    int numCandidates = min(paletteSize / 2, 8);
    
    // Find up to 8 nearest palette colors by distance in linear space
    int indices[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float distances[8] = {1e10, 1e10, 1e10, 1e10, 1e10, 1e10, 1e10, 1e10};
    
    for (int i = 0; i < paletteSize; i++) {
        // Convert palette color to linear for comparison
        float3 paletteLinear = srgbToLinear3(palette[i]);
        float3 diff = rgbLinear - paletteLinear;
        float d = dot(diff, diff);
        
        // Insert into sorted list
        int lastIdx = numCandidates - 1;
        if (d < distances[lastIdx]) {
            // Find insertion point
            int insertPos = lastIdx;
            while (insertPos > 0 && d < distances[insertPos - 1]) {
                insertPos--;
            }
            // Shift elements down
            for (int j = lastIdx; j > insertPos; j--) {
                distances[j] = distances[j - 1];
                indices[j] = indices[j - 1];
            }
            distances[insertPos] = d;
            indices[insertPos] = i;
        }
    }
    
    // Store valid colors in sRGB space, fill rest with terminator
    for (int i = 0; i < 8; i++) {
        if (i < numCandidates) {
            lut[tid].colors[i] = palette[indices[i]];
        } else {
            lut[tid].colors[i] = LUT_TERMINATOR;
        }
    }
}

// Helper: get LUT index from coordinates, clamped to valid range
inline uint getLutIdx(uint3 coord) {
    coord = min(coord, uint3(31));
    return coord.b * 1024 + coord.g * 32 + coord.r;
}

// Apply palette with ordered dithering using LUT lookup
// Uses up to 8 nearest palette colors with inverse-distance weighting for smooth dithering.
// Stops at terminator value if fewer than 8 colors available.
// All dithering calculations done in linear space for perceptually correct results.
kernel void applyAdaptivePalette(
    texture2d<float, access::read> inTex [[texture(0)]],
    texture2d<float, access::write> outTex [[texture(1)]],
    device const PaletteLUTEntry *lut [[buffer(0)]],
    constant int &dither [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= inTex.get_width() || gid.y >= inTex.get_height()) return;
    
    float3 rgb = inTex.read(gid).rgb;
    
    // Quantize to 32x32x32 LUT coordinates
    uint3 lutCoord = uint3(saturate(rgb) * 31.0 + 0.5);
    lutCoord = min(lutCoord, uint3(31));
    uint lutIdx = lutCoord.b * 1024 + lutCoord.g * 32 + lutCoord.r;
    
    // Lookup colors from LUT
    PaletteLUTEntry entry = lut[lutIdx];
    
    // No dithering: just use nearest
    if (dither == 0) {
        outTex.write(float4(entry.colors[0], 1.0), gid);
        return;
    }
    
    // Count valid colors (stop at terminator)
    int numColors = 0;
    for (int i = 0; i < 8; i++) {
        if (isTerminator(entry.colors[i])) break;
        numColors++;
    }
    
    // Fallback: if only 1 color or somehow 0, just use nearest
    if (numColors <= 1) {
        outTex.write(float4(entry.colors[0], 1.0), gid);
        return;
    }
    
    // Convert input to linear space for distance calculations
    float3 rgbLinear = srgbToLinear3(rgb);
    
    // Compute distances and weights only for valid candidates
    float weights[8];
    float totalWeight = 0.0;
    float softening = 0.01;  // Prevents division by zero and controls sharpness
    
    for (int i = 0; i < numColors; i++) {
        float3 candLinear = srgbToLinear3(entry.colors[i]);
        float3 diff = rgbLinear - candLinear;
        float d = length(diff) + softening;
        weights[i] = 1.0 / (d * d);  // Inverse square weighting
        totalWeight += weights[i];
    }
    
    // Normalize weights to sum to 1
    for (int i = 0; i < numColors; i++) {
        weights[i] /= totalWeight;
    }
    
    // Build cumulative distribution for threshold-based selection
    float cumulative[8];
    cumulative[0] = weights[0];
    for (int i = 1; i < numColors - 1; i++) {
        cumulative[i] = cumulative[i - 1] + weights[i];
    }
    cumulative[numColors - 1] = 1.0;  // Ensure last valid entry is exactly 1.0
    
    // Get Bayer threshold for this pixel
    float threshold = bayer16x16[(gid.y % 16) * 16 + (gid.x % 16)] / 256.0;
    
    // Select color based on where threshold falls in cumulative distribution
    int selected = 0;
    for (int i = 0; i < numColors; i++) {
        if (threshold >= cumulative[i]) {
            selected = i + 1;
        }
    }
    selected = min(selected, numColors - 1);
    
    outTex.write(float4(entry.colors[selected], 1.0), gid);
}
