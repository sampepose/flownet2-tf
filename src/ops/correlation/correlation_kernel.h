#ifndef FLOWNET_CORRELATION_H_
#define FLOWNET_CORRELATION_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;

void Correlation(const GPUDevice& device,
                 typename TTypes<float, 4>::ConstTensor input_a,
                 typename TTypes<float, 4>::ConstTensor input_b,
                 int max_displacement,
                 int neighborhood_grid_radius,
                 int neighborhood_grid_width,
                 int kernel_radius,
                 int kernel_size,
                 int stride_1,
                 int stride_2,
                 typename TTypes<float, 4>::Tensor output);
} // end namespace tensorflow

#endif  // FLOWNET_CORRELATION_H_
