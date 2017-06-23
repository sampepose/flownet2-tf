#define EIGEN_USE_THREADS

#include "flow_augmentation.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <typename Device>
class FlowAugmentation : public OpKernel {
 public:
  explicit FlowAugmentation(OpKernelConstruction* ctx) : OpKernel(ctx) {
    // Get the crop [height, width] tensor and verify its dimensions
    OP_REQUIRES_OK(ctx, ctx->GetAttr("crop", &crop_));
    OP_REQUIRES(ctx, crop_.size() == 2, errors::InvalidArgument("crop must be 2 dimensions"));
  }

  void Compute(OpKernelContext* ctx) override {
    // Get the input images and transforms and verify their dimensions
    const Tensor& flows_t = ctx->input(0);
    const Tensor& inv_transforms_from_a_t = ctx->input(1);
    const Tensor& transforms_from_b_t = ctx->input(2);
    OP_REQUIRES(ctx, flows_t.dims() == 4,
                errors::InvalidArgument("Input images must have rank 4"));
    OP_REQUIRES(ctx, (TensorShapeUtils::IsMatrix(transforms_from_b_t.shape()) &&
                      transforms_from_b_t.dim_size(0) == flows_t.dim_size(0) &&
                      transforms_from_b_t.dim_size(1) == 6),
                errors::InvalidArgument(
                    "Input transforms_from_b should be num_images x 6"));
    OP_REQUIRES(ctx, (TensorShapeUtils::IsMatrix(inv_transforms_from_a_t.shape()) &&
                      inv_transforms_from_a_t.dim_size(0) == flows_t.dim_size(0) &&
                      inv_transforms_from_a_t.dim_size(1) == 6),
                errors::InvalidArgument(
                    "Input inv_transforms_from_a should be num_images x 6"));

    // Allocate the memory for the output
    Tensor* output_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
        0, TensorShape({flows_t.dim_size(0), crop_[0], crop_[1], flows_t.dim_size(3)}), &output_t));

    // Perform flow augmentation
    auto flows = flows_t.tensor<float, 4>();
    auto transforms_from_b = transforms_from_b_t.tensor<float, 2>();
    auto inv_transforms_from_a = inv_transforms_from_a_t.tensor<float, 2>();
    auto output = output_t->tensor<float, 4>();

    FillFlowAugmentation(ctx->eigen_gpu_device(),
                         output,
                         flows,
                         transforms_from_b,
                         inv_transforms_from_a);
  }

  private:
    std::vector<int32> crop_;
};

REGISTER_KERNEL_BUILDER(Name("FlowAugmentation")
                          .Device(DEVICE_GPU),
                      FlowAugmentation<GPUDevice>)
}  // end namespace tensorflow
