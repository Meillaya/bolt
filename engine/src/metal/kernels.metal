#include <metal_stdlib>
using namespace metal;

kernel void copy_f32(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &element_count [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= element_count) return;
    output[id] = input[id];
}

kernel void add_f32(
    device const float *left [[buffer(0)]],
    device const float *right [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &element_count [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= element_count) return;
    output[id] = left[id] + right[id];
}
