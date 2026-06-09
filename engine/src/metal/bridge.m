#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

@interface BoltMetalRuntime : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> commandQueue;
@property(nonatomic, strong) id<MTLLibrary> library;
@property(nonatomic, strong) id<MTLComputePipelineState> vectorCopyState;
@property(nonatomic, strong) id<MTLComputePipelineState> vectorAddState;
@property(nonatomic, strong) id<MTLComputePipelineState> matrixMultiplyState;
@property(nonatomic, strong) id<MTLComputePipelineState> biasAddState;
@property(nonatomic, strong) id<MTLComputePipelineState> vectorReluState;
@property(nonatomic, strong) id<MTLComputePipelineState> reduceSumState;
@property(nonatomic, strong) id<MTLComputePipelineState> softmaxState;
@end

@implementation BoltMetalRuntime
@end

@interface BoltMetalBuffer : NSObject
@property(nonatomic, strong) id<MTLBuffer> buffer;
@property(nonatomic, assign) NSUInteger elementCount;
@property(nonatomic, assign) NSUInteger byteLength;
@end

@implementation BoltMetalBuffer
@end

static char *bolt_copy_error_message(NSString *message) {
    const char *utf8 = message != nil ? message.UTF8String : "unknown Metal error";
    return utf8 != NULL ? strdup(utf8) : strdup("unknown Metal error");
}

static void bolt_set_error(char **error_out, NSString *message) {
    if (error_out != NULL) {
        *error_out = bolt_copy_error_message(message);
    }
}

static BOOL bolt_validate_buffer_count(BoltMetalBuffer *buffer, size_t count, NSString *label, char **error_out) {
    if (buffer == nil) {
        bolt_set_error(error_out, [NSString stringWithFormat:@"missing %@ Metal buffer", label]);
        return NO;
    }
    if (count > buffer.elementCount) {
        bolt_set_error(
            error_out,
            [NSString stringWithFormat:@"%@ Metal buffer too small for %zu elements", label, count]
        );
        return NO;
    }
    return YES;
}

static BOOL bolt_validate_buffer_bytes(BoltMetalBuffer *buffer, size_t byte_count, NSString *label, char **error_out) {
    if (buffer == nil) {
        bolt_set_error(error_out, [NSString stringWithFormat:@"missing %@ Metal buffer", label]);
        return NO;
    }
    if (byte_count > buffer.byteLength) {
        bolt_set_error(
            error_out,
            [NSString stringWithFormat:@"%@ Metal buffer too small for %zu bytes", label, byte_count]
        );
        return NO;
    }
    return YES;
}

static id<MTLComputePipelineState> bolt_make_pipeline(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    NSString *name,
    char **error_out
) {
    NSError *error = nil;
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (function == nil) {
        bolt_set_error(error_out, [NSString stringWithFormat:@"missing Metal kernel %@", name]);
        return nil;
    }

    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                                 error:&error];
    if (pipeline == nil) {
        NSString *description = error.localizedDescription ?: [NSString stringWithFormat:@"failed to create %@ pipeline", name];
        bolt_set_error(error_out, description);
        return nil;
    }

    return pipeline;
}

static BOOL bolt_validate_u32(size_t value, NSString *label, char **error_out) {
    if (value > UINT32_MAX) {
        bolt_set_error(error_out, [NSString stringWithFormat:@"%@ exceeds Metal uint limit", label]);
        return NO;
    }
    return YES;
}

static BOOL bolt_encode_buffer_pipeline(
    BoltMetalRuntime *runtime,
    id<MTLComputePipelineState> pipeline,
    BoltMetalBuffer *left_buffer,
    BoltMetalBuffer *right_buffer,
    size_t count,
    BoltMetalBuffer *output_buffer,
    char **error_out
) {
    if (count == 0) {
        return YES;
    }
    if (runtime == nil) {
        bolt_set_error(error_out, @"missing Metal runtime");
        return NO;
    }
    if (pipeline == nil) {
        bolt_set_error(error_out, @"missing Metal compute pipeline");
        return NO;
    }
    if (!bolt_validate_buffer_count(left_buffer, count, @"left", error_out)) {
        return NO;
    }
    if (right_buffer != nil && !bolt_validate_buffer_count(right_buffer, count, @"right", error_out)) {
        return NO;
    }
    if (!bolt_validate_buffer_count(output_buffer, count, @"output", error_out)) {
        return NO;
    }

    id<MTLCommandBuffer> command_buffer = [runtime.commandQueue commandBuffer];
    if (command_buffer == nil) {
        bolt_set_error(error_out, @"failed to create Metal command buffer");
        return NO;
    }

    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        bolt_set_error(error_out, @"failed to create Metal compute encoder");
        return NO;
    }

    const uint32_t element_count = (uint32_t)count;
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:left_buffer.buffer offset:0 atIndex:0];
    if (right_buffer != nil) {
        [encoder setBuffer:right_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer.buffer offset:0 atIndex:2];
        [encoder setBytes:&element_count length:sizeof(element_count) atIndex:3];
    } else {
        [encoder setBuffer:output_buffer.buffer offset:0 atIndex:1];
        [encoder setBytes:&element_count length:sizeof(element_count) atIndex:2];
    }

    const NSUInteger thread_width = MAX((NSUInteger)1, MIN((NSUInteger)count, pipeline.maxTotalThreadsPerThreadgroup));
    const MTLSize grid_size = MTLSizeMake(count, 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(thread_width, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];

    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if (command_buffer.error != nil) {
        NSString *description = command_buffer.error.localizedDescription ?: @"Metal command buffer failed";
        bolt_set_error(error_out, description);
        return NO;
    }

    return YES;
}

static BOOL bolt_encode_single_thread_pipeline(
    BoltMetalRuntime *runtime,
    id<MTLComputePipelineState> pipeline,
    BoltMetalBuffer *input_buffer,
    size_t count,
    BoltMetalBuffer *output_buffer,
    size_t output_count,
    char **error_out
) {
    if (count == 0) {
        return YES;
    }
    if (runtime == nil) {
        bolt_set_error(error_out, @"missing Metal runtime");
        return NO;
    }
    if (pipeline == nil) {
        bolt_set_error(error_out, @"missing Metal compute pipeline");
        return NO;
    }
    if (!bolt_validate_u32(count, @"element count", error_out)) {
        return NO;
    }
    if (!bolt_validate_buffer_count(input_buffer, count, @"input", error_out)) {
        return NO;
    }
    if (!bolt_validate_buffer_count(output_buffer, output_count, @"output", error_out)) {
        return NO;
    }

    id<MTLCommandBuffer> command_buffer = [runtime.commandQueue commandBuffer];
    if (command_buffer == nil) {
        bolt_set_error(error_out, @"failed to create Metal command buffer");
        return NO;
    }

    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        bolt_set_error(error_out, @"failed to create Metal compute encoder");
        return NO;
    }

    const uint32_t element_count = (uint32_t)count;
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input_buffer.buffer offset:0 atIndex:0];
    [encoder setBuffer:output_buffer.buffer offset:0 atIndex:1];
    [encoder setBytes:&element_count length:sizeof(element_count) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];

    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if (command_buffer.error != nil) {
        NSString *description = command_buffer.error.localizedDescription ?: @"Metal command buffer failed";
        bolt_set_error(error_out, description);
        return NO;
    }

    return YES;
}

bool bolt_metal_context_is_available(void) {
    @autoreleasepool {
        return MTLCreateSystemDefaultDevice() != nil;
    }
}

bool bolt_metal_validate_dispatch_dimensions(
    size_t grid_x,
    size_t grid_y,
    size_t grid_z,
    size_t threads_x,
    size_t threads_y,
    size_t threads_z,
    size_t max_threads_per_threadgroup,
    char **error_out
) {
    if (grid_x == 0 || grid_y == 0 || grid_z == 0) {
        bolt_set_error(error_out, @"Metal dispatch grid dimensions must be non-zero");
        return false;
    }
    if (threads_x == 0 || threads_y == 0 || threads_z == 0) {
        bolt_set_error(error_out, @"Metal threadgroup dimensions must be non-zero");
        return false;
    }
    if (threads_x > SIZE_MAX / threads_y || threads_x * threads_y > SIZE_MAX / threads_z) {
        bolt_set_error(error_out, @"Metal threadgroup dimensions overflow");
        return false;
    }
    const size_t thread_count = threads_x * threads_y * threads_z;
    if (thread_count > max_threads_per_threadgroup) {
        bolt_set_error(error_out, @"Metal threadgroup dimensions exceed pipeline maximum");
        return false;
    }
    return true;
}

bool bolt_metal_context_has_kernel(void *context_handle, const char *name, size_t name_len) {
    @autoreleasepool {
        BoltMetalRuntime *runtime = (__bridge BoltMetalRuntime *)context_handle;
        if (runtime == nil || name == NULL || name_len == 0) {
            return false;
        }
        NSString *kernel_name = [[NSString alloc] initWithBytes:name
                                                         length:name_len
                                                       encoding:NSUTF8StringEncoding];
        if (kernel_name == nil) {
            return false;
        }
        return [runtime.library newFunctionWithName:kernel_name] != nil;
    }
}

void bolt_metal_string_destroy(char *message) {
    if (message != NULL) {
        free(message);
    }
}

void *bolt_metal_context_create(const char *source, size_t source_len, char **error_out) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            bolt_set_error(error_out, @"Metal is unavailable on this machine");
            return NULL;
        }

        NSString *source_string = [[NSString alloc] initWithBytes:source
                                                           length:source_len
                                                         encoding:NSUTF8StringEncoding];
        if (source_string == nil) {
            bolt_set_error(error_out, @"failed to decode embedded Metal source");
            return NULL;
        }

        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source_string
                                                      options:nil
                                                        error:&error];
        if (library == nil) {
            NSString *description = error.localizedDescription ?: @"failed to compile Metal source";
            bolt_set_error(error_out, description);
            return NULL;
        }

        id<MTLCommandQueue> command_queue = [device newCommandQueue];
        if (command_queue == nil) {
            bolt_set_error(error_out, @"failed to create Metal command queue");
            return NULL;
        }

        id<MTLComputePipelineState> copy_pipeline = bolt_make_pipeline(device, library, @"copy_f32", error_out);
        if (copy_pipeline == nil) return NULL;
        id<MTLComputePipelineState> add_pipeline = bolt_make_pipeline(device, library, @"add_f32", error_out);
        if (add_pipeline == nil) return NULL;
        id<MTLComputePipelineState> matmul_pipeline = bolt_make_pipeline(device, library, @"matmul_f32", error_out);
        if (matmul_pipeline == nil) return NULL;
        id<MTLComputePipelineState> bias_add_pipeline = bolt_make_pipeline(device, library, @"bias_add_f32", error_out);
        if (bias_add_pipeline == nil) return NULL;
        id<MTLComputePipelineState> relu_pipeline = bolt_make_pipeline(device, library, @"relu_f32", error_out);
        if (relu_pipeline == nil) return NULL;
        id<MTLComputePipelineState> reduce_sum_pipeline = bolt_make_pipeline(device, library, @"reduce_sum_f32", error_out);
        if (reduce_sum_pipeline == nil) return NULL;
        id<MTLComputePipelineState> softmax_pipeline = bolt_make_pipeline(device, library, @"softmax_f32", error_out);
        if (softmax_pipeline == nil) return NULL;

        BoltMetalRuntime *runtime = [[BoltMetalRuntime alloc] init];
        runtime.device = device;
        runtime.commandQueue = command_queue;
        runtime.library = library;
        runtime.vectorCopyState = copy_pipeline;
        runtime.vectorAddState = add_pipeline;
        runtime.matrixMultiplyState = matmul_pipeline;
        runtime.biasAddState = bias_add_pipeline;
        runtime.vectorReluState = relu_pipeline;
        runtime.reduceSumState = reduce_sum_pipeline;
        runtime.softmaxState = softmax_pipeline;
        return (__bridge_retained void *)runtime;
    }
}

void bolt_metal_context_destroy(void *handle) {
    if (handle == NULL) return;
    @autoreleasepool {
        __unused BoltMetalRuntime *runtime = (__bridge_transfer BoltMetalRuntime *)handle;
    }
}

void *bolt_metal_buffer_create(void *context_handle, size_t count, char **error_out) {
    @autoreleasepool {
        BoltMetalRuntime *runtime = (__bridge BoltMetalRuntime *)context_handle;
        if (runtime == nil) {
            bolt_set_error(error_out, @"missing Metal runtime for buffer creation");
            return NULL;
        }
        if (count == 0) {
            bolt_set_error(error_out, @"cannot allocate a zero-length Metal buffer");
            return NULL;
        }

        const NSUInteger byte_count = count * sizeof(float);
        id<MTLBuffer> metal_buffer = [runtime.device newBufferWithLength:byte_count
                                                                 options:MTLResourceStorageModeShared];
        if (metal_buffer == nil) {
            bolt_set_error(error_out, @"failed to allocate Metal shared buffer");
            return NULL;
        }

        memset(metal_buffer.contents, 0, byte_count);

        BoltMetalBuffer *buffer = [[BoltMetalBuffer alloc] init];
        buffer.buffer = metal_buffer;
        buffer.elementCount = count;
        buffer.byteLength = byte_count;
        return (__bridge_retained void *)buffer;
    }
}

void *bolt_metal_buffer_create_bytes(void *context_handle, size_t byte_count, char **error_out) {
    @autoreleasepool {
        BoltMetalRuntime *runtime = (__bridge BoltMetalRuntime *)context_handle;
        if (runtime == nil) {
            bolt_set_error(error_out, @"missing Metal runtime for byte-buffer creation");
            return NULL;
        }
        if (byte_count == 0) {
            bolt_set_error(error_out, @"cannot allocate a zero-length Metal byte buffer");
            return NULL;
        }

        id<MTLBuffer> metal_buffer = [runtime.device newBufferWithLength:byte_count
                                                                 options:MTLResourceStorageModeShared];
        if (metal_buffer == nil) {
            bolt_set_error(error_out, @"failed to allocate Metal shared byte buffer");
            return NULL;
        }

        memset(metal_buffer.contents, 0, byte_count);

        BoltMetalBuffer *buffer = [[BoltMetalBuffer alloc] init];
        buffer.buffer = metal_buffer;
        buffer.elementCount = byte_count / sizeof(float);
        buffer.byteLength = byte_count;
        return (__bridge_retained void *)buffer;
    }
}

void bolt_metal_buffer_destroy(void *handle) {
    if (handle == NULL) return;
    @autoreleasepool {
        __unused BoltMetalBuffer *buffer = (__bridge_transfer BoltMetalBuffer *)handle;
    }
}

float *bolt_metal_buffer_contents_f32(void *handle) {
    @autoreleasepool {
        BoltMetalBuffer *buffer = (__bridge BoltMetalBuffer *)handle;
        if (buffer == nil) {
            return NULL;
        }
        return (float *)buffer.buffer.contents;
    }
}

uint8_t *bolt_metal_buffer_contents_u8(void *handle) {
    @autoreleasepool {
        BoltMetalBuffer *buffer = (__bridge BoltMetalBuffer *)handle;
        if (buffer == nil) {
            return NULL;
        }
        return (uint8_t *)buffer.buffer.contents;
    }
}

size_t bolt_metal_buffer_byte_length(void *handle) {
    @autoreleasepool {
        BoltMetalBuffer *buffer = (__bridge BoltMetalBuffer *)handle;
        if (buffer == nil) {
            return 0;
        }
        return buffer.byteLength;
    }
}

bool bolt_metal_buffer_validate_bytes(void *handle, size_t byte_count, char **error_out) {
    @autoreleasepool {
        BoltMetalBuffer *buffer = (__bridge BoltMetalBuffer *)handle;
        return bolt_validate_buffer_bytes(buffer, byte_count, @"shared", error_out);
    }
}

bool bolt_metal_copy_buffer_f32(
    void *context_handle,
    void *input_handle,
    size_t count,
    void *output_handle,
    char **error_out
) {
    @autoreleasepool {
        BoltMetalRuntime *runtime = (__bridge BoltMetalRuntime *)context_handle;
        BoltMetalBuffer *input_buffer = (__bridge BoltMetalBuffer *)input_handle;
        BoltMetalBuffer *output_buffer = (__bridge BoltMetalBuffer *)output_handle;
        return bolt_encode_buffer_pipeline(
            runtime,
            runtime.vectorCopyState,
            input_buffer,
            nil,
            count,
            output_buffer,
            error_out
        );
    }
}

bool bolt_metal_add_buffer_f32(
    void *context_handle,
    void *left_handle,
    void *right_handle,
    size_t count,
    void *output_handle,
    char **error_out
) {
    @autoreleasepool {
        BoltMetalRuntime *runtime = (__bridge BoltMetalRuntime *)context_handle;
        BoltMetalBuffer *left_buffer = (__bridge BoltMetalBuffer *)left_handle;
        BoltMetalBuffer *right_buffer = (__bridge BoltMetalBuffer *)right_handle;
        BoltMetalBuffer *output_buffer = (__bridge BoltMetalBuffer *)output_handle;
        return bolt_encode_buffer_pipeline(
            runtime,
            runtime.vectorAddState,
            left_buffer,
            right_buffer,
            count,
            output_buffer,
            error_out
        );
    }
}

bool bolt_metal_matmul_buffer_f32(
    void *context_handle,
    void *left_handle,
    void *right_handle,
    size_t rows,
    size_t cols,
    size_t inner,
    void *output_handle,
    char **error_out
) {
    @autoreleasepool {
        BoltMetalRuntime *runtime = (__bridge BoltMetalRuntime *)context_handle;
        BoltMetalBuffer *left_buffer = (__bridge BoltMetalBuffer *)left_handle;
        BoltMetalBuffer *right_buffer = (__bridge BoltMetalBuffer *)right_handle;
        BoltMetalBuffer *output_buffer = (__bridge BoltMetalBuffer *)output_handle;
        if (rows == 0 || cols == 0 || inner == 0) {
            return YES;
        }
        if (runtime == nil) {
            bolt_set_error(error_out, @"missing Metal runtime");
            return NO;
        }
        if (runtime.matrixMultiplyState == nil) {
            bolt_set_error(error_out, @"missing matmul_f32 pipeline");
            return NO;
        }
        if (!bolt_validate_u32(rows, @"rows", error_out) ||
            !bolt_validate_u32(cols, @"cols", error_out) ||
            !bolt_validate_u32(inner, @"inner", error_out)) {
            return NO;
        }

        const size_t left_count = rows * inner;
        const size_t right_count = inner * cols;
        const size_t output_count = rows * cols;
        if (!bolt_validate_buffer_count(left_buffer, left_count, @"left", error_out) ||
            !bolt_validate_buffer_count(right_buffer, right_count, @"right", error_out) ||
            !bolt_validate_buffer_count(output_buffer, output_count, @"output", error_out)) {
            return NO;
        }

        id<MTLCommandBuffer> command_buffer = [runtime.commandQueue commandBuffer];
        if (command_buffer == nil) {
            bolt_set_error(error_out, @"failed to create Metal command buffer");
            return NO;
        }

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            bolt_set_error(error_out, @"failed to create Metal compute encoder");
            return NO;
        }

        const uint32_t row_count = (uint32_t)rows;
        const uint32_t col_count = (uint32_t)cols;
        const uint32_t inner_count = (uint32_t)inner;
        [encoder setComputePipelineState:runtime.matrixMultiplyState];
        [encoder setBuffer:left_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:right_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer.buffer offset:0 atIndex:2];
        [encoder setBytes:&row_count length:sizeof(row_count) atIndex:3];
        [encoder setBytes:&col_count length:sizeof(col_count) atIndex:4];
        [encoder setBytes:&inner_count length:sizeof(inner_count) atIndex:5];

        const NSUInteger thread_width = MAX((NSUInteger)1, MIN((NSUInteger)output_count, runtime.matrixMultiplyState.maxTotalThreadsPerThreadgroup));
        [encoder dispatchThreads:MTLSizeMake(output_count, 1, 1) threadsPerThreadgroup:MTLSizeMake(thread_width, 1, 1)];
        [encoder endEncoding];

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.error != nil) {
            NSString *description = command_buffer.error.localizedDescription ?: @"Metal command buffer failed";
            bolt_set_error(error_out, description);
            return NO;
        }

        return YES;
    }
}

typedef struct {
    uint32_t M;
    uint32_t K;
    uint32_t group_size;
} BoltMetalQMVDims;

bool bolt_metal_qmv_buffer_f32(
    void *context_handle,
    void *packed_bits_handle,
    void *scales_handle,
    void *input_handle,
    size_t rows,
    size_t cols,
    size_t group_size,
    void *output_handle,
    char **error_out
) {
    @autoreleasepool {
        BoltMetalRuntime *runtime = (__bridge BoltMetalRuntime *)context_handle;
        BoltMetalBuffer *packed_bits_buffer = (__bridge BoltMetalBuffer *)packed_bits_handle;
        BoltMetalBuffer *scales_buffer = (__bridge BoltMetalBuffer *)scales_handle;
        BoltMetalBuffer *input_buffer = (__bridge BoltMetalBuffer *)input_handle;
        BoltMetalBuffer *output_buffer = (__bridge BoltMetalBuffer *)output_handle;
        if (rows == 0 || cols == 0 || group_size == 0) {
            bolt_set_error(error_out, @"invalid empty qmv dimensions");
            return NO;
        }
        if (runtime == nil) {
            bolt_set_error(error_out, @"missing Metal runtime");
            return NO;
        }
        if (cols % group_size != 0 || cols % 32 != 0) {
            bolt_set_error(error_out, @"qmv requires cols divisible by group_size and 32");
            return NO;
        }
        if (!bolt_validate_u32(rows, @"qmv rows", error_out) ||
            !bolt_validate_u32(cols, @"qmv cols", error_out) ||
            !bolt_validate_u32(group_size, @"qmv group size", error_out)) {
            return NO;
        }
        const size_t packed_bytes = rows * (cols / 8);
        const size_t scale_count = rows * ((cols + group_size - 1) / group_size);
        if (!bolt_validate_buffer_bytes(packed_bits_buffer, packed_bytes, @"qmv packed_bits", error_out) ||
            !bolt_validate_buffer_bytes(scales_buffer, scale_count * sizeof(uint16_t), @"qmv scales", error_out) ||
            !bolt_validate_buffer_count(input_buffer, cols, @"qmv input", error_out) ||
            !bolt_validate_buffer_count(output_buffer, rows, @"qmv output", error_out)) {
            return NO;
        }

        id<MTLComputePipelineState> pipeline = bolt_make_pipeline(runtime.device, runtime.library, @"qmv", error_out);
        if (pipeline == nil) {
            return NO;
        }

        id<MTLCommandBuffer> command_buffer = [runtime.commandQueue commandBuffer];
        if (command_buffer == nil) {
            bolt_set_error(error_out, @"failed to create Metal command buffer");
            return NO;
        }

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            bolt_set_error(error_out, @"failed to create Metal compute encoder");
            return NO;
        }

        const BoltMetalQMVDims dims = {
            .M = (uint32_t)rows,
            .K = (uint32_t)cols,
            .group_size = (uint32_t)group_size,
        };
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:packed_bits_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:scales_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:2];
        [encoder setBuffer:output_buffer.buffer offset:0 atIndex:3];
        [encoder setBytes:&dims length:sizeof(dims) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake((rows + 1) / 2, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [encoder endEncoding];

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.error != nil) {
            NSString *description = command_buffer.error.localizedDescription ?: @"Metal qmv command buffer failed";
            bolt_set_error(error_out, description);
            return NO;
        }

        return YES;
    }
}

bool bolt_metal_q4mv_buffer_f32(
    void *context_handle,
    void *packed_nibbles_handle,
    void *scales_handle,
    void *biases_handle,
    void *input_handle,
    size_t rows,
    size_t cols,
    size_t group_size,
    void *output_handle,
    char **error_out
) {
    @autoreleasepool {
        BoltMetalRuntime *runtime = (__bridge BoltMetalRuntime *)context_handle;
        BoltMetalBuffer *packed_buffer = (__bridge BoltMetalBuffer *)packed_nibbles_handle;
        BoltMetalBuffer *scales_buffer = (__bridge BoltMetalBuffer *)scales_handle;
        BoltMetalBuffer *biases_buffer = (__bridge BoltMetalBuffer *)biases_handle;
        BoltMetalBuffer *input_buffer = (__bridge BoltMetalBuffer *)input_handle;
        BoltMetalBuffer *output_buffer = (__bridge BoltMetalBuffer *)output_handle;
        if (rows == 0 || cols == 0 || group_size == 0) {
            bolt_set_error(error_out, @"invalid empty q4mv dimensions");
            return NO;
        }
        if (runtime == nil) {
            bolt_set_error(error_out, @"missing Metal runtime");
            return NO;
        }
        if (cols % group_size != 0 || cols % 2 != 0) {
            bolt_set_error(error_out, @"q4mv requires cols divisible by group_size and 2");
            return NO;
        }
        if (!bolt_validate_u32(rows, @"q4mv rows", error_out) ||
            !bolt_validate_u32(cols, @"q4mv cols", error_out) ||
            !bolt_validate_u32(group_size, @"q4mv group size", error_out)) {
            return NO;
        }
        const size_t packed_bytes = rows * (cols / 2);
        const size_t group_count = rows * (cols / group_size);
        if (!bolt_validate_buffer_bytes(packed_buffer, packed_bytes, @"q4mv packed_nibbles", error_out) ||
            !bolt_validate_buffer_bytes(scales_buffer, group_count * sizeof(uint16_t), @"q4mv scales", error_out) ||
            !bolt_validate_buffer_bytes(biases_buffer, group_count * sizeof(uint16_t), @"q4mv biases", error_out) ||
            !bolt_validate_buffer_count(input_buffer, cols, @"q4mv input", error_out) ||
            !bolt_validate_buffer_count(output_buffer, rows, @"q4mv output", error_out)) {
            return NO;
        }

        id<MTLComputePipelineState> pipeline = bolt_make_pipeline(runtime.device, runtime.library, @"q4mv_f32", error_out);
        if (pipeline == nil) {
            return NO;
        }

        id<MTLCommandBuffer> command_buffer = [runtime.commandQueue commandBuffer];
        if (command_buffer == nil) {
            bolt_set_error(error_out, @"failed to create Metal command buffer");
            return NO;
        }

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            bolt_set_error(error_out, @"failed to create Metal compute encoder");
            return NO;
        }

        const BoltMetalQMVDims dims = {
            .M = (uint32_t)rows,
            .K = (uint32_t)cols,
            .group_size = (uint32_t)group_size,
        };
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:packed_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:scales_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:biases_buffer.buffer offset:0 atIndex:2];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:3];
        [encoder setBuffer:output_buffer.buffer offset:0 atIndex:4];
        [encoder setBytes:&dims length:sizeof(dims) atIndex:5];
        const NSUInteger thread_width = MAX((NSUInteger)1, MIN((NSUInteger)rows, pipeline.maxTotalThreadsPerThreadgroup));
        [encoder dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(thread_width, 1, 1)];
        [encoder endEncoding];

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.error != nil) {
            NSString *description = command_buffer.error.localizedDescription ?: @"Metal q4mv command buffer failed";
            bolt_set_error(error_out, description);
            return NO;
        }

        return YES;
    }
}

bool bolt_metal_bias_add_buffer_f32(
    void *context_handle,
    void *input_handle,
    void *bias_handle,
    size_t rows,
    size_t cols,
    void *output_handle,
    char **error_out
) {
    @autoreleasepool {
        BoltMetalRuntime *runtime = (__bridge BoltMetalRuntime *)context_handle;
        BoltMetalBuffer *input_buffer = (__bridge BoltMetalBuffer *)input_handle;
        BoltMetalBuffer *bias_buffer = (__bridge BoltMetalBuffer *)bias_handle;
        BoltMetalBuffer *output_buffer = (__bridge BoltMetalBuffer *)output_handle;
        if (rows == 0 || cols == 0) {
            return YES;
        }
        if (runtime == nil) {
            bolt_set_error(error_out, @"missing Metal runtime");
            return NO;
        }
        if (runtime.biasAddState == nil) {
            bolt_set_error(error_out, @"missing bias_add_f32 pipeline");
            return NO;
        }
        if (!bolt_validate_u32(rows, @"rows", error_out) ||
            !bolt_validate_u32(cols, @"cols", error_out)) {
            return NO;
        }

        const size_t element_count = rows * cols;
        if (!bolt_validate_buffer_count(input_buffer, element_count, @"input", error_out) ||
            !bolt_validate_buffer_count(bias_buffer, cols, @"bias", error_out) ||
            !bolt_validate_buffer_count(output_buffer, element_count, @"output", error_out)) {
            return NO;
        }

        id<MTLCommandBuffer> command_buffer = [runtime.commandQueue commandBuffer];
        if (command_buffer == nil) {
            bolt_set_error(error_out, @"failed to create Metal command buffer");
            return NO;
        }

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            bolt_set_error(error_out, @"failed to create Metal compute encoder");
            return NO;
        }

        const uint32_t row_count = (uint32_t)rows;
        const uint32_t col_count = (uint32_t)cols;
        [encoder setComputePipelineState:runtime.biasAddState];
        [encoder setBuffer:input_buffer.buffer offset:0 atIndex:0];
        [encoder setBuffer:bias_buffer.buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer.buffer offset:0 atIndex:2];
        [encoder setBytes:&row_count length:sizeof(row_count) atIndex:3];
        [encoder setBytes:&col_count length:sizeof(col_count) atIndex:4];

        const NSUInteger thread_width = MAX((NSUInteger)1, MIN((NSUInteger)element_count, runtime.biasAddState.maxTotalThreadsPerThreadgroup));
        [encoder dispatchThreads:MTLSizeMake(element_count, 1, 1) threadsPerThreadgroup:MTLSizeMake(thread_width, 1, 1)];
        [encoder endEncoding];

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.error != nil) {
            NSString *description = command_buffer.error.localizedDescription ?: @"Metal command buffer failed";
            bolt_set_error(error_out, description);
            return NO;
        }

        return YES;
    }
}

bool bolt_metal_relu_buffer_f32(
    void *context_handle,
    void *input_handle,
    size_t count,
    void *output_handle,
    char **error_out
) {
    @autoreleasepool {
        BoltMetalRuntime *runtime = (__bridge BoltMetalRuntime *)context_handle;
        BoltMetalBuffer *input_buffer = (__bridge BoltMetalBuffer *)input_handle;
        BoltMetalBuffer *output_buffer = (__bridge BoltMetalBuffer *)output_handle;
        return bolt_encode_buffer_pipeline(
            runtime,
            runtime.vectorReluState,
            input_buffer,
            nil,
            count,
            output_buffer,
            error_out
        );
    }
}

bool bolt_metal_reduce_sum_buffer_f32(
    void *context_handle,
    void *input_handle,
    size_t count,
    void *output_handle,
    char **error_out
) {
    @autoreleasepool {
        BoltMetalRuntime *runtime = (__bridge BoltMetalRuntime *)context_handle;
        BoltMetalBuffer *input_buffer = (__bridge BoltMetalBuffer *)input_handle;
        BoltMetalBuffer *output_buffer = (__bridge BoltMetalBuffer *)output_handle;
        return bolt_encode_single_thread_pipeline(
            runtime,
            runtime.reduceSumState,
            input_buffer,
            count,
            output_buffer,
            1,
            error_out
        );
    }
}

bool bolt_metal_softmax_buffer_f32(
    void *context_handle,
    void *input_handle,
    size_t count,
    void *output_handle,
    char **error_out
) {
    @autoreleasepool {
        BoltMetalRuntime *runtime = (__bridge BoltMetalRuntime *)context_handle;
        BoltMetalBuffer *input_buffer = (__bridge BoltMetalBuffer *)input_handle;
        BoltMetalBuffer *output_buffer = (__bridge BoltMetalBuffer *)output_handle;
        return bolt_encode_single_thread_pipeline(
            runtime,
            runtime.softmaxState,
            input_buffer,
            count,
            output_buffer,
            count,
            error_out
        );
    }
}
