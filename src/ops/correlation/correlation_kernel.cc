#define EIGEN_USE_THREADS

#include "correlation_kernel.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;

template<typename Device>
class CorrelationKernel : public OpKernel {
  public:
    explicit CorrelationKernel(OpKernelConstruction *ctx) : OpKernel(ctx) {
      // Get the attributes
      OP_REQUIRES_OK(ctx, ctx->GetAttr("kernel_size", &kernel_size));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("max_displacement", &max_displacement));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("stride_1", &stride_1));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("stride_2", &stride_2));

      OP_REQUIRES(ctx, kernel_size % 2 != 0, errors::InvalidArgument("kernel_size must be odd"));
    }

    void Compute(OpKernelContext *ctx) override {
      // Get the input images and transforms and verify their dimensions
      const Tensor& input_a_t = ctx->input(0);
      const Tensor& input_b_t = ctx->input(1);

      OP_REQUIRES(ctx, input_a_t.dims() == 4, errors::InvalidArgument("input_a must have rank 4"));
      OP_REQUIRES(ctx, input_b_t.dims() == 4, errors::InvalidArgument("input_b must have rank 4"));

      // Get dimensions of input
      int64 batch_size   = input_a_t.dim_size(0);
      int64 input_height = input_a_t.dim_size(1);
      int64 input_width  = input_a_t.dim_size(2);

      // The size of unreachable border region on each side
      int kernel_radius = (kernel_size - 1) / 2;
      int border_size   = max_displacement + kernel_radius;

      // Calculate the output dimensions
      int64 output_height = (int64)ceil((float)(input_height - border_size * 2) / (float)stride_1);
      int64 output_width  = (int64)ceil((float)(input_width - border_size * 2) / (float)stride_1);

      OP_REQUIRES(ctx, output_height >= 1,
                  errors::InvalidArgument("Neighborhood and kernel don't fit in input height."));
      OP_REQUIRES(ctx, output_width >= 1,
                  errors::InvalidArgument("Neighborhood and kernel don't fit in input width."));

      int   neighborhood_grid_radius = max_displacement / stride_2;
      int   neighborhood_grid_width  = neighborhood_grid_radius * 2 + 1;
      int64 output_channels          = neighborhood_grid_width * neighborhood_grid_width;

      // Allocate the memory for the output
      Tensor *output_t;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(
                       0,
                       TensorShape({ batch_size, output_height, output_width, output_channels }),
                       &output_t));

      // Perform cross correlation
      auto input_a = input_a_t.tensor<float, 4>();
      auto input_b = input_b_t.tensor<float, 4>();
      auto output  = output_t->tensor<float, 4>();

      Correlation(ctx->eigen_gpu_device(),
                  input_a,
                  input_b,
                  max_displacement,
                  neighborhood_grid_radius,
                  neighborhood_grid_width,
                  kernel_radius,
                  kernel_size,
                  stride_1,
                  stride_2,
                  output);
    }

  private:
    int kernel_size;
    int max_displacement;
    int stride_1;
    int stride_2;
};

REGISTER_KERNEL_BUILDER(Name("Correlation")
                        .Device(DEVICE_GPU),
                        CorrelationKernel<GPUDevice>)
} // end namespace tensorflow
