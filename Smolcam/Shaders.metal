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

constant float blueNoise16x16[256] = {
    124, 56,199, 17,142, 83,231, 44,168, 99,215, 31,152, 71,188, 12,
     76,240,101,178, 63,196,  8,117,249, 58,137,180, 22,241,105, 67,
    202, 33,155, 89,223,  1,147, 85,193, 28,217,  6,161, 48,197,145,
      5,184,130, 52,164,112,237, 42,175,108,159, 87,236,119, 81,228,
    247, 72,219, 19,243,  9, 70,211, 15,234, 54,207, 35,167,  3,141,
     97,151,  0,188,135,182, 57,154, 94,133,170, 77,143,252, 61,109,
    179, 43,121,238, 84, 29,201,  4,248,  2, 46,224, 18,201, 91,174,
     26,209,166, 68,146,107,226,123, 78,189,115,  9,128,  0,220, 37,
    139, 93,244, 14,191,251, 51,163,  0,241, 65,163,253, 55,153,131,
    225, 58,113,172, 41,  7,139, 87,213, 39,191,100, 37,186,  9,255,
     82,183, 21,233,122, 79,203, 25,148,120,  7,245,134, 76,204, 64,
    157,140,  0, 69,212,158,  0,178,243, 66,169, 53,211,  0,118,229,
    106,248, 50,195, 31,246,111, 55, 13,104,221, 20,156,247, 46,172,
     29,  0,147, 86,171,  0, 90,232,186, 33,141,  0, 94,177, 0, 95,
    218,129, 73,239,125,  0,149, 68,  0,251, 84,196,126,  0,134,250,
     11,192, 38,  0, 53,200, 27,165,103,  0,175, 59,244, 70,212, 17,
};

kernel void ditherQuantize(
    texture2d<float, access::read> inTex [[texture(0)]],
    texture2d<float, access::write> outTex [[texture(1)]],
    constant int &bitsPerPixel [[buffer(0)]],
    constant int &dither [[buffer(1)]],
    constant int &ditherType [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= inTex.get_width() || gid.y >= inTex.get_height()) return;
    
    float4 color = inTex.read(gid);
    float3 maxLevels = getMaxLevels(bitsPerPixel);
    
    if (dither != 0) {
        uint idx = (gid.y % 16) * 16 + (gid.x % 16);
        float threshold = (ditherType == 0 ? bayer16x16[idx] : blueNoise16x16[idx]) / 256.0;
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
