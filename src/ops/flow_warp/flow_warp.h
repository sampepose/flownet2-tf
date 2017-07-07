#ifndef FLOWNET_FLOWWARP_H_
#define FLOWNET_FLOWWARP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;

void FlowWarp(const GPUDevice& device,
              typename TTypes<float, 4>::ConstTensor input,
              typename TTypes<float, 4>::ConstTensor flow,
              typename TTypes<float, 4>::Tensor output);
} // end namespace tensorflow

#endif  // FLOWNET_FLOWWARP_H_
