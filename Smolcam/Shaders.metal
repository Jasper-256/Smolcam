#include <metal_stdlib>
using namespace metal;

constant float bayer8x8[64] = {
     0,32, 8,40, 2,34,10,42,
    48,16,56,24,50,18,58,26,
    12,44, 4,36,14,46, 6,38,
    60,28,52,20,62,30,54,22,
     3,35,11,43, 1,33, 9,41,
    51,19,59,27,49,17,57,25,
    15,47, 7,39,13,45, 5,37,
    63,31,55,23,61,29,53,21
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
    float levels = float(1 << bits);
    float scale = 1.0 / (levels - 1.0);
    
    if (dither != 0) {
        // Bayer 8x8 ordered dithering
        uint idx = (gid.y % 8) * 8 + (gid.x % 8);
        float threshold = (bayer8x8[idx] / 64.0 - 0.5) / levels;
        color.rgb = floor((color.rgb + threshold) * levels) * scale;
    } else {
        color.rgb = floor(color.rgb * levels) * scale;
    }
    
    outTex.write(float4(saturate(color.rgb), color.a), gid);
}

