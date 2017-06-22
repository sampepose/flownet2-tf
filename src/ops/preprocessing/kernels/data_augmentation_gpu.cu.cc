#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>
#include <iostream>

#include "data_augmentation.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;

inline __device__ __host__ float clamp(float f, float a, float b) {
  return fmaxf(a, fminf(f, b));
}

__global__ void SpatialAugmentation(
  const int32  nthreads,
  const int    channels,
  const int    src_height,
  const int    src_width,
  const float *src_data,
  const int    src_count,
  const int    out_height,
  const int    out_width,
  float       *out_data,
  const float *transMats) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // TF, NHWK: ((n * H + h) * W + w) * K + k at point (n, h, w, k)
    int c = index % channels;
    int x = (index / channels) % out_width;
    int y = (index / channels / out_width) % out_height;
    int n = index / channels / out_width / out_height;

    const float *transMat = transMats + n * 6;
    float xpos            = x * transMat[0] + y * transMat[1] + transMat[2];
    float ypos            = x * transMat[3] + y * transMat[4] + transMat[5];

    xpos = clamp(xpos, 0.0f, (float)(src_width) - 1.05f);
    ypos = clamp(ypos, 0.0f, (float)(src_height) - 1.05f);

    float tlx = floor(xpos);
    float tly = floor(ypos);

    // Bilinear interpolation
    int srcTLIdx = ((n * src_height + tly) * src_width + tlx) * channels + c;
    int srcTRIdx = (int)fminf(
      (float)((n * src_height + tly) * src_width + tlx + 1) * channels + c,
      (float)src_count);
    int srcBLIdx = (int)fminf(
      (float)((n * src_height + tly + 1) * src_width + tlx) * channels + c,
      (float)src_count);
    int srcBRIdx = (int)fminf(
      (float)((n * src_height + tly + 1) * src_width + tlx + 1) * channels + c,
      (float)src_count);

    float xdist = xpos - tlx;
    float ydist = ypos - tly;

    float dest = (1 - xdist) * (1 - ydist) * src_data[srcTLIdx]
                 + (xdist) * (ydist) * src_data[srcBRIdx]
                 + (1 - xdist) * (ydist) * src_data[srcBLIdx]
                 + (xdist) * (1 - ydist) * src_data[srcTRIdx];

    out_data[index] = dest;
  }
}

float rng_generate(const AugmentationParam& param, const float default_value) {
  std::random_device rd;  // Will be used to obtain a seed for the random number
                          // engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()

  if (param.rand_type == "uniform_bernoulli") {
    float tmp1 = 0.0;
    bool  tmp2 = false;

    if (param.prob > 0.0) {
      std::bernoulli_distribution bernoulli(param.prob);
      tmp2 = bernoulli(gen);
    }

    if (!tmp2) {
      return default_value;
    }

    if (param.spread > 0.0) {
      std::uniform_real_distribution<> uniform(param.mean - param.spread,
                                               param.mean + param.spread);
      tmp1 = uniform(gen);
    } else {
      tmp1 = param.mean;
    }

    if (param.should_exp) {
      tmp1 = exp(tmp1);
    }

    return tmp1;
  } else if (param.rand_type == "gaussian_bernoulli") {
    float tmp1 = 0.0;
    bool  tmp2 = false;

    if (param.prob > 0.0) {
      std::bernoulli_distribution bernoulli(param.prob);
      tmp2 = bernoulli(gen);
    }

    if (!tmp2) {
      return default_value;
    }

    if (param.spread > 0.0) {
      std::normal_distribution<> normal(param.mean, param.spread);
      tmp1 = normal(gen);
    } else {
      tmp1 = param.mean;
    }

    if (param.should_exp) {
      tmp1 = exp(tmp1);
    }

    return tmp1;
  } else {
    throw "Unknown random type: " + param.rand_type;
  }
}

void generate_spatial_coeffs(const AugmentationParams& params,
                             AugmentationCoeffs      & coeff) {
  if (params.has_translate_x) {
    coeff.translate_x = rng_generate(params.translate_x, TRANSLATE_DEFAULT);
  }

  if (params.has_translate_y) {
    coeff.translate_y = rng_generate(params.translate_y, TRANSLATE_DEFAULT);
  }

  if (params.has_rotate) {
    coeff.rotate = rng_generate(params.rotate, ROTATE_DEFAULT);
  }

  if (params.has_zoom) {
    coeff.zoom_x = rng_generate(params.zoom, ZOOM_DEFAULT);
    coeff.zoom_y = coeff.zoom_x;
  }

  if (params.has_squeeze) {
    const float squeeze_coeff = rng_generate(params.squeeze, 1.0);
    coeff.zoom_x *= squeeze_coeff;
    coeff.zoom_y /= squeeze_coeff;
  }
}

AugmentationCoeffs generate_valid_spatial_coeffs(
  const AugmentationParams& params,
  AugmentationCoeffs      & coeff,
  const int                 src_height,
  const int                 src_width)
{
  int   good_params = 0;
  int   counter = 0;
  int   x, y;
  float x1, x2, y1, y2;
  AugmentationCoeffs new_coeff;

  while (good_params < 4 && counter < 50) {
    // Generate new coefficients
    new_coeff.clear();
    generate_spatial_coeffs(params, new_coeff);

    // Combine newly generated coeffs with the incoming coefficients
    new_coeff.combine_with(coeff);

    // Check if all 4 corners of the transformed image fit into the original
    // image
    good_params = 0;

    for (x = 0; x < params.crop_width; x += params.crop_width - 1) {
      for (y = 0; y < params.crop_height; y += params.crop_height - 1) {
        // Move the origin to the center wrt output size
        x1 = x - 0.5 * params.crop_width;
        y1 = y - 0.5 * params.crop_height;

        // Rotate
        const float cos_ = cos(new_coeff.rotate);
        const float sin_ = sin(new_coeff.rotate);
        x2 = cos_ * x1 - sin_ * y1;
        y2 = sin_ * x1 + cos_ * y1;

        // Translate
        x2 += new_coeff.translate_x * params.crop_width;
        y2 += new_coeff.translate_y * params.crop_height;

        // Zoom
        x2 /= new_coeff.zoom_x;
        y2 /= new_coeff.zoom_y;

        // Move the origin back wrt input size
        x2 += 0.5 * src_width;
        y2 += 0.5 * src_height;

        if ((floor(x2) > 0) && (floor(x2) < src_width - 2.0) && (floor(y2) > 0) &&
            (floor(y2) < src_height - 2.0)) {
          good_params++;
        }
      }
    }
    counter++;
  }

  if (counter >= 50) {
    printf("Exceeded maximum tries in finding suitable spatial coeffs.\n");

    return coeff;
  }

  return new_coeff;
}

void ComputeDataAugmentation(
  OpKernelContext *ctx,
  const GPUDevice& device,
  typename TTypes<float, 4>::ConstTensor input,
  AugmentationParams& params,
  typename TTypes<float, 4>::Tensor output,
  std::vector<AugmentationCoeffs>& in_coeffs,
  std::vector<AugmentationCoeffs>& out_coeffs)
{
  const int batch_size   = input.dimension(0);
  const int src_height   = input.dimension(1);
  const int src_width    = input.dimension(2);
  const int channels     = input.dimension(3);
  const int input_count  = batch_size * src_height * src_width * channels;
  const int output_count = output.dimension(0) * output.dimension(1) *
                           output.dimension(2) * output.dimension(3);

  // Create temporary Tensor to hold all spatial transformation matrices
  Tensor spatial_matrices_tensor;

  tensorflow::AllocatorAttributes pinned_allocator;
  pinned_allocator.set_on_host(true);
  pinned_allocator.set_gpu_compatible(true);

  OP_REQUIRES_OK(ctx,
                 ctx->allocate_temp(DataTypeToEnum<float>::value,
                                    TensorShape({ batch_size,
                                                  6 }),
                                    &spatial_matrices_tensor, pinned_allocator));
  auto spatial_matrices = spatial_matrices_tensor.tensor<float, 2>();

  // Generate coefficients for each augmentation type
  for (int i = 0; i < batch_size; i++) {
    if (params.do_spatial_transform()) {
      AugmentationCoeffs in_coeff;

      if (in_coeffs.size() == batch_size) {
        in_coeff = in_coeffs[i];
      } else {
        in_coeff = AugmentationCoeffs();
      }

      // Generate new spatial coefficients (combined with incoming ones)
      AugmentationCoeffs coeffs = generate_valid_spatial_coeffs(params,
                                                                in_coeff,
                                                                src_height,
                                                                src_width);

      coeffs.store_spatial_matrix(params.crop_height,
                                  params.crop_width,
                                  src_height,
                                  src_width, spatial_matrices.data() + i * 8);

      out_coeffs.push_back(coeffs);
    }
  }

  // Perform spatial transformation
  if (params.do_spatial_transform()) {
    CudaLaunchConfig config = GetCudaLaunchConfig(output_count, device);

    SpatialAugmentation << < config.block_count, config.thread_per_block, 0,
      device.stream() >> > (
      output_count, channels, src_height, src_width, input.data(),
      input_count, params.crop_height,
      params.crop_width, output.data(),
      spatial_matrices.data());
  }

  // TODO: Somehow return the coeffs
}
} // end namespace tensorflow

#endif  // GOOGLE_CUDA
