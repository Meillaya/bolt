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

kernel void matmul_f32(
    device const float *left [[buffer(0)]],
    device const float *right [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &cols [[buffer(4)]],
    constant uint &inner [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    const uint element_count = rows * cols;
    if (id >= element_count) return;

    const uint row = id / cols;
    const uint col = id % cols;
    float sum = 0.0f;
    for (uint k = 0; k < inner; k += 1) {
        sum += left[row * inner + k] * right[k * cols + col];
    }
    output[id] = sum;
}

kernel void bias_add_f32(
    device const float *input [[buffer(0)]],
    device const float *bias [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &cols [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    const uint element_count = rows * cols;
    if (id >= element_count) return;
    output[id] = input[id] + bias[id % cols];
}

kernel void relu_f32(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &element_count [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= element_count) return;
    output[id] = max(input[id], 0.0f);
}

kernel void reduce_sum_f32(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &element_count [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id > 0 || element_count == 0) return;
    float sum = 0.0f;
    for (uint i = 0; i < element_count; i += 1) {
        sum += input[i];
    }
    output[0] = sum;
}

kernel void softmax_f32(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &element_count [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id > 0 || element_count == 0) return;

    float max_value = input[0];
    for (uint i = 1; i < element_count; i += 1) {
        max_value = max(max_value, input[i]);
    }

    float sum = 0.0f;
    for (uint i = 0; i < element_count; i += 1) {
        const float shifted = exp(input[i] - max_value);
        output[i] = shifted;
        sum += shifted;
    }

    for (uint i = 0; i < element_count; i += 1) {
        output[i] = output[i] / sum;
    }
}
