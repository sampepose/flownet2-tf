#define EIGEN_USE_THREADS

#include "data_augmentation.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;

template<typename Device>
class DataAugmentation : public OpKernel {
public:

  explicit DataAugmentation(OpKernelConstruction *ctx) : OpKernel(ctx) {
    // Get the crop [height, width] tensor and verify its dimensions
    OP_REQUIRES_OK(ctx, ctx->GetAttr("crop", &crop_));
    OP_REQUIRES(ctx, crop_.size() == 2,
                errors::InvalidArgument("crop must be 2 dimensions"));

    // TODO: Verify params are all the same length

    // Get the tensors for params_a and verify their dimensions
    OP_REQUIRES_OK(ctx, ctx->GetAttr("params_a_name", &params_a_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("params_a_rand_type", &params_a_rand_type_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("params_a_exp", &params_a_exp_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("params_a_mean", &params_a_mean_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("params_a_spread", &params_a_spread_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("params_a_prob", &params_a_prob_));

    // Get the tensors for params_b and verify their dimensions
    OP_REQUIRES_OK(ctx, ctx->GetAttr("params_b_name", &params_b_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("params_b_rand_type", &params_b_rand_type_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("params_b_exp", &params_b_exp_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("params_b_mean", &params_b_mean_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("params_b_spread", &params_b_spread_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("params_b_prob", &params_b_prob_));
  }

  void Compute(OpKernelContext *ctx) override {
    // Get the input images
    const Tensor& input_a_t = ctx->input(0);
    const Tensor& input_b_t = ctx->input(1);

    // Allocate the memory for the output images
    Tensor *output_a_t;
    Tensor *output_b_t;


    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                     0,
                     TensorShape({ input_a_t.dim_size(0), crop_[0], crop_[1],
                                   input_a_t.dim_size(3) }), &output_a_t));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                     1,
                     TensorShape({ input_b_t.dim_size(0), crop_[0], crop_[1],
                                   input_b_t.dim_size(3) }), &output_b_t));

    // Allocate the memory for the output spatial transforms
    Tensor *spat_transform_t;
    Tensor *inv_spat_transform_t;

    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                     2,
                     TensorShape({ input_a_t.dim_size(0), 6 }),
                     &spat_transform_t));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                     3,
                     TensorShape({ input_a_t.dim_size(0), 6 }),
                     &inv_spat_transform_t));

    // Perform data augmentation on image A
    auto input_a                    = input_a_t.tensor<float, 4>();
    auto output_a                   = output_a_t->tensor<float, 4>();
    AugmentationParams aug_params_a = AugmentationParams(crop_[0], crop_[1],
                                                         params_a_name_,
                                                         params_a_rand_type_,
                                                         params_a_exp_,
                                                         params_a_mean_,
                                                         params_a_spread_,
                                                         params_a_prob_);
    std::vector<AugmentationCoeffs> coeffs_a;
    ComputeDataAugmentation(ctx,
                            ctx->eigen_gpu_device(),
                            input_a,
                            aug_params_a,
                            output_a,
                            coeffs_a, // We have no incoming coefficients
                            coeffs_a);

    // Perform data augmentation on image B using coeffs generated above
    auto input_b                    = input_b_t.tensor<float, 4>();
    auto output_b                   = output_b_t->tensor<float, 4>();
    AugmentationParams aug_params_b = AugmentationParams(crop_[0], crop_[1],
                                                         params_b_name_,
                                                         params_b_rand_type_,
                                                         params_b_exp_,
                                                         params_b_mean_,
                                                         params_b_spread_,
                                                         params_b_prob_);
    std::vector<AugmentationCoeffs> coeffs_b;
    ComputeDataAugmentation(ctx,
                            ctx->eigen_gpu_device(),
                            input_b,
                            aug_params_b,
                            output_b,
                            coeffs_a,
                            coeffs_b);

    const int src_height       = input_a.dimension(1);
    const int src_width        = input_a.dimension(2);
    auto inv_transforms_from_a = spat_transform_t->tensor<float, 2>();
    auto transforms_from_b     = inv_spat_transform_t->tensor<float, 2>();

    for (int i = 0; i < coeffs_a.size(); i++) {
      auto coeffs = coeffs_a[i];
      coeffs.store_spatial_matrix(crop_[0],
                                  crop_[1],
                                  src_height,
                                  src_width,
                                  inv_transforms_from_a.data() + i * 6,
                                  true);
    }

    for (int i = 0; i < coeffs_b.size(); i++) {
      auto coeffs = coeffs_b[i];
      coeffs.store_spatial_matrix(crop_[0],
                                  crop_[1],
                                  src_height,
                                  src_width,
                                  transforms_from_b.data() + i * 6,
                                  false);
    }
  }

private:

  std::vector<int32>crop_;

  // Params A
  std::vector<string>params_a_name_;
  std::vector<string>params_a_rand_type_;
  std::vector<bool>params_a_exp_;
  std::vector<float>params_a_mean_;
  std::vector<float>params_a_spread_;
  std::vector<float>params_a_prob_;

  // Params B
  std::vector<string>params_b_name_;
  std::vector<string>params_b_rand_type_;
  std::vector<bool>params_b_exp_;
  std::vector<float>params_b_mean_;
  std::vector<float>params_b_spread_;
  std::vector<float>params_b_prob_;
};

REGISTER_KERNEL_BUILDER(Name("DataAugmentation")
                        .Device(DEVICE_GPU)
                        .HostMemory("inv_transforms_from_a")
                        .HostMemory("transforms_from_b"),
                        DataAugmentation<GPUDevice>)
} // end namespace tensorflow
