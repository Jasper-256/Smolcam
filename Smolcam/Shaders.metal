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

kernel void ditherQuantize(
    texture2d<float, access::read> inTex [[texture(0)]],
    texture2d<float, access::write> outTex [[texture(1)]],
    constant int &bits [[buffer(0)]],
    constant int &dither [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= inTex.get_width() || gid.y >= inTex.get_height()) return;
    
    float4 color = inTex.read(gid);
    float maxLevel = float((1 << bits) - 1);
    
    if (dither != 0) {
        uint idx = (gid.y % 16) * 16 + (gid.x % 16);
        float threshold = bayer16x16[idx] / 256.0;
        color.rgb = floor(color.rgb * maxLevel + threshold * 0.9) / maxLevel;
    } else {
        color.rgb = floor(color.rgb * maxLevel + 0.5) / maxLevel;
    }
    
    outTex.write(float4(saturate(color.rgb), color.a), gid);
}

kernel void downsampleQuantize(
    texture2d<float, access::read> inTex [[texture(0)]],
    texture2d<float, access::write> outTex [[texture(1)]],
    constant int &bits [[buffer(0)]],
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
    float maxLevel = float((1 << bits) - 1);
    
    uint idx = (gid.y % 16) * 16 + (gid.x % 16);
    float threshold = bayer16x16[idx] / 256.0;
    float3 quantized = floor(avg * maxLevel + threshold) / maxLevel;

    outTex.write(float4(quantized, 1.0), gid);
}
