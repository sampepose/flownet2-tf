#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "spatial_transform.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace functor {

// Explicit instantiation of the CPU functor.
typedef Eigen::ThreadPoolDevice CPUDevice;

template class FillSpatialTransform<CPUDevice, uint8>;
template class FillSpatialTransform<CPUDevice, int32>;
template class FillSpatialTransform<CPUDevice, int64>;
template class FillSpatialTransform<CPUDevice, float>;
template class FillSpatialTransform<CPUDevice, double>;

}  // end namespace functor

typedef Eigen::ThreadPoolDevice CPUDevice;

using functor::FillSpatialTransform;
using generator::ProjectiveGenerator;

template <typename Device, typename T>
class SpatialTransform : public OpKernel {
 public:
  explicit SpatialTransform(OpKernelConstruction* ctx) : OpKernel(ctx) {
    // Get the crop [height, width] tensor and verify its dimensions
    OP_REQUIRES_OK(ctx, ctx->GetAttr("crop", &crop_));
    OP_REQUIRES(ctx, crop_.size() == 2, errors::InvalidArgument("crop must be 2 dimensions"));
  }

  void Compute(OpKernelContext* ctx) override {
    // Get the input images and transforms and verify their dimensions
    const Tensor& images_t = ctx->input(0);
    const Tensor& transform_t = ctx->input(1);
    OP_REQUIRES(ctx, images_t.shape().dims() == 4,
                errors::InvalidArgument("Input images must have rank 4"));
    OP_REQUIRES(ctx, (TensorShapeUtils::IsMatrix(transform_t.shape()) &&
                      (transform_t.dim_size(0) == images_t.dim_size(0) ||
                       transform_t.dim_size(0) == 1) &&
                      transform_t.dim_size(1) ==
                          ProjectiveGenerator<Device, T>::kNumParameters),
                errors::InvalidArgument(
                    "Input transform should be num_images x 8 or 1 x 8"));

    // Allocate the memory for the output
    Tensor* output_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
        0, TensorShape({images_t.dim_size(0), crop_[0], crop_[1], images_t.dim_size(3)}), &output_t));

    // Perform spatial transformation
    auto images = images_t.tensor<T, 4>();
    auto transform = transform_t.matrix<float>();
    auto output = output_t->tensor<T, 4>();
    const FillSpatialTransform<Device, T> functor;
    functor(ctx->eigen_device<Device>(), &output, images, transform);
  }

  private:
    std::vector<int32> crop_;
};

#define REGISTER_SPATIAL_TRANSFORM(TYPE)                      \
  REGISTER_KERNEL_BUILDER(Name("SpatialTransform")            \
                              .Device(DEVICE_CPU)             \
                              .TypeConstraint<TYPE>("dtype"), \
                          SpatialTransform<CPUDevice, TYPE>)
TF_CALL_uint8(REGISTER_SPATIAL_TRANSFORM);
TF_CALL_int32(REGISTER_SPATIAL_TRANSFORM);
TF_CALL_int64(REGISTER_SPATIAL_TRANSFORM);
TF_CALL_float(REGISTER_SPATIAL_TRANSFORM);
TF_CALL_double(REGISTER_SPATIAL_TRANSFORM);
#undef REGISTER_SPATIAL_TRANSFORM

#if GOOGLE_CUDA
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

// NOTE(ringwalt): We get an undefined symbol error if we don't explicitly
// instantiate the operator() in GCC'd code.
#define DECLARE_FUNCTOR(TYPE)                                               \
  template <>                                                               \
  void FillSpatialTransform<GPUDevice, TYPE>::operator()(                   \
      const GPUDevice& device, OutputType* output, const InputType& images, \
      const TransformsType& transform) const;                               \
  extern template class FillSpatialTransform<GPUDevice, TYPE>

TF_CALL_uint8(DECLARE_FUNCTOR);
TF_CALL_int32(DECLARE_FUNCTOR);
TF_CALL_int64(DECLARE_FUNCTOR);
TF_CALL_float(DECLARE_FUNCTOR);
TF_CALL_double(DECLARE_FUNCTOR);

}  // end namespace functor

#define REGISTER_SPATIAL_TRANSFORM(TYPE)                      \
  REGISTER_KERNEL_BUILDER(Name("SpatialTransform")            \
                              .Device(DEVICE_GPU)             \
                              .TypeConstraint<TYPE>("dtype"), \
                          SpatialTransform<GPUDevice, TYPE>)
TF_CALL_uint8(REGISTER_SPATIAL_TRANSFORM);
TF_CALL_int32(REGISTER_SPATIAL_TRANSFORM);
TF_CALL_int64(REGISTER_SPATIAL_TRANSFORM);
TF_CALL_float(REGISTER_SPATIAL_TRANSFORM);
TF_CALL_double(REGISTER_SPATIAL_TRANSFORM);
#undef REGISTER_SPATIAL_TRANSFORM

#endif  // GOOGLE_CUDA

}  // end namespace tensorflow
