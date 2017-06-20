#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "spatial_transform.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace functor {

// Explicit instantiation of the GPU functor.
typedef Eigen::GpuDevice GPUDevice;

template class FillSpatialTransform<GPUDevice, uint8>;
template class FillSpatialTransform<GPUDevice, int32>;
template class FillSpatialTransform<GPUDevice, int64>;
template class FillSpatialTransform<GPUDevice, float>;
template class FillSpatialTransform<GPUDevice, double>;

}  // end namespace functor

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
