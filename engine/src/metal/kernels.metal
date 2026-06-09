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

// Reference-compatible Q1_0_g128 matrix-vector multiply foundation.
// Matches the buffer/dimension contract used by reference nnmetal qmv.
struct QMVDims {
    uint M;
    uint K;
    uint group_size;
};

kernel void qmv(
    device const uint8_t* packed_bits [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const float* input [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant QMVDims& dims [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_index_in_threadgroup]])
{
    const uint simdgroup_idx = tid_in_tg / 32;
    const uint lane = tid_in_tg % 32;
    const uint row = tgid * 2 + simdgroup_idx;
    if (row >= dims.M) return;

    const uint K = dims.K;
    const uint group_size = dims.group_size;
    const uint groups_per_row = (K + group_size - 1) / group_size;
    const uint cols_per_lane = K / 32;
    const uint col_start = lane * cols_per_lane;
    const uint row_bit_offset = row * (K / 8);
    const uint row_scale_offset = row * groups_per_row;

    float accum = 0.0f;
    uint cur_group = col_start / group_size;
    float set_accum = 0.0f;
    float group_sum = 0.0f;

    for (uint c = 0; c < cols_per_lane; c++) {
        const uint col = col_start + c;
        const uint group_idx = col / group_size;
        if (group_idx != cur_group) {
            const float scale = float(scales[row_scale_offset + cur_group]);
            accum += scale * (2.0f * set_accum - group_sum);
            set_accum = 0.0f;
            group_sum = 0.0f;
            cur_group = group_idx;
        }

        const uint byte_idx = row_bit_offset + col / 8;
        const uint bit_pos = col % 8;
        const bool bit_set = (packed_bits[byte_idx] >> bit_pos) & 1;
        const float x_val = input[col];
        group_sum += x_val;
        set_accum += select(0.0f, x_val, bit_set);
    }

    const float scale = float(scales[row_scale_offset + cur_group]);
    accum += scale * (2.0f * set_accum - group_sum);

    float row_sum = simd_sum(accum);
    if (lane == 0) output[row] = row_sum;
}

// Generic MLX Q4 affine matrix-vector multiply foundation.
// Layout matches Qwen/Q4 safetensors: two unsigned 4-bit weights per byte
// (low nibble first), plus BF16/F16-compatible half scale and bias per group.
struct Q4MVDims {
    uint M;
    uint K;
    uint group_size;
};

inline float bolt_bf16_to_f32(ushort raw) {
    return as_type<float>(uint(raw) << 16);
}

kernel void q4mv_f32(
    device const uint8_t* packed_nibbles [[buffer(0)]],
    device const ushort* scales [[buffer(1)]],
    device const ushort* biases [[buffer(2)]],
    device const float* input [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant Q4MVDims& dims [[buffer(5)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= dims.M) return;

    const uint groups_per_row = dims.K / dims.group_size;
    const uint packed_bytes_per_row = dims.K / 2;
    const uint row_byte_offset = row * packed_bytes_per_row;
    const uint row_group_offset = row * groups_per_row;

    float sum = 0.0f;
    for (uint group = 0; group < groups_per_row; group += 1) {
        const float scale = bolt_bf16_to_f32(scales[row_group_offset + group]);
        const float bias = bolt_bf16_to_f32(biases[row_group_offset + group]);
        const uint col_begin = group * dims.group_size;
        const uint col_end = col_begin + dims.group_size;
        for (uint col = col_begin; col < col_end; col += 1) {
            const uint byte_value = packed_nibbles[row_byte_offset + col / 2];
            const uint nibble = (col & 1u) == 0u ? (byte_value & 0xFu) : ((byte_value >> 4) & 0xFu);
            sum += (scale * float(nibble) + bias) * input[col];
        }
    }

    output[row] = sum;
}
