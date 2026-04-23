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
@end

@implementation BoltMetalRuntime
@end

@interface BoltMetalBuffer : NSObject
@property(nonatomic, strong) id<MTLBuffer> buffer;
@property(nonatomic, assign) NSUInteger elementCount;
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

bool bolt_metal_context_is_available(void) {
    @autoreleasepool {
        return MTLCreateSystemDefaultDevice() != nil;
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

        id<MTLFunction> copy_function = [library newFunctionWithName:@"copy_f32"];
        if (copy_function == nil) {
            bolt_set_error(error_out, @"missing Metal kernel copy_f32");
            return NULL;
        }

        id<MTLFunction> add_function = [library newFunctionWithName:@"add_f32"];
        if (add_function == nil) {
            bolt_set_error(error_out, @"missing Metal kernel add_f32");
            return NULL;
        }

        id<MTLComputePipelineState> copy_pipeline = [device newComputePipelineStateWithFunction:copy_function
                                                                                           error:&error];
        if (copy_pipeline == nil) {
            NSString *description = error.localizedDescription ?: @"failed to create copy_f32 pipeline";
            bolt_set_error(error_out, description);
            return NULL;
        }

        id<MTLComputePipelineState> add_pipeline = [device newComputePipelineStateWithFunction:add_function
                                                                                          error:&error];
        if (add_pipeline == nil) {
            NSString *description = error.localizedDescription ?: @"failed to create add_f32 pipeline";
            bolt_set_error(error_out, description);
            return NULL;
        }

        BoltMetalRuntime *runtime = [[BoltMetalRuntime alloc] init];
        runtime.device = device;
        runtime.commandQueue = command_queue;
        runtime.library = library;
        runtime.vectorCopyState = copy_pipeline;
        runtime.vectorAddState = add_pipeline;
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
