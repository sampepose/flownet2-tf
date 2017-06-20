#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>
#include <iostream>

#include "flow_augmentation.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

__global__ void FillFlowAugmentationKernel(
    const int32 nthreads,
    const float* flow_ptr,
    const float* transform_ptr,
    const float* inv_transform_ptr,
    const int src_total_count, const int src_height, const int src_width,
    const int batch_size, const int out_height,
    const int out_width, float* output_ptr) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {
            const float x = (float)(index % out_width);
            const float y = (float)((index / out_width) % out_height);
            const int n = (index / out_width / out_height) % batch_size;

            const int transformIdx = n * 8;

            const float xpos1 = x * transform_ptr[transformIdx + 0]
                                + y * transform_ptr[transformIdx + 1]
                                + transform_ptr[transformIdx + 2];
            const float ypos1 = x * transform_ptr[transformIdx + 3]
                                + y * transform_ptr[transformIdx + 4]
                                + transform_ptr[transformIdx + 5];

            // Caffe, NKHW: ((n * K + k) * H + h) * W + w at point (n, k, h, w)
            // TF, NHWK: ((n * H + h) * W + w) * K + k at point (n, h, w, k)
            const int srcXIdx = ((n * src_height + (int)(y + 0.5)) * src_width + (int)(x + 0.5)) * 2 + 0;
            const int srcYIdx = ((n * src_height + (int)(y + 0.5)) * src_width + (int)(x + 0.5)) * 2 + 1;

            const float xpos2 = xpos1 + flow_ptr[min(srcXIdx, src_total_count)];
            const float ypos2 = ypos1 + flow_ptr[min(srcYIdx, src_total_count)];

            const float xpos3 = xpos2 * inv_transform_ptr[transformIdx + 0]
                                + ypos2 * inv_transform_ptr[transformIdx + 1]
                                + inv_transform_ptr[transformIdx + 2];
            const float ypos3 = xpos2 * inv_transform_ptr[transformIdx + 3]
                                + ypos2 * inv_transform_ptr[transformIdx + 4]
                                + inv_transform_ptr[transformIdx + 5];

            output_ptr[((n * out_height + (int)y) * out_width + (int)x) * 2 + 0] = xpos3 - x;
            output_ptr[((n * out_height + (int)y) * out_width + (int)x) * 2 + 1] = ypos3 - y;
        }
}

bool FillFlowAugmentation(const GPUDevice& device,
                          typename TTypes<float, 4>::Tensor output,
                          typename TTypes<float, 4>::ConstTensor flows,
                          typename TTypes<float, 2>::ConstTensor transforms_a,
                          typename TTypes<float, 2>::ConstTensor inv_transforms_b) {
    const int batch_size = output.dimension(0);
    const int out_height = output.dimension(1);
    const int out_width = output.dimension(2);
    const int depth = 2;
    const int total_count = batch_size * out_height * out_width * depth;
    const int src_total_count = flows.dimension(0) * flows.dimension(1) * flows.dimension(2)
                            * flows.dimension(3);

    CudaLaunchConfig config = GetCudaLaunchConfig(total_count / 2, device);
    FillFlowAugmentationKernel<<<config.block_count, config.thread_per_block, 0,
                        device.stream()>>>(
      total_count / 2, flows.data(), transforms_a.data(), inv_transforms_b.data(),
      src_total_count, flows.dimension(1), flows.dimension(2), batch_size,
      out_height, out_width, output.data());
    return device.ok();
}

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
