#ifndef FLOWNET_DATA_AUG_H_
#define FLOWNET_DATA_AUG_H_

#define EPSILON 0.0001

#define TRANSLATE_DEFAULT 0.0
#define ROTATE_DEFAULT 0.0
#define ZOOM_DEFAULT 1.0

// See docs in ../ops/image_ops.cc.
#include <random>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;

inline bool essentiallyEqual(float a, float b) {
  return fabs(a - b) <= ((fabs(a) > fabs(b) ? fabs(b) : fabs(a)) * EPSILON);
}

class TransMat {
public:

  float t0 = 1.0, t1 = 0.0, t2 = 0.0;
  float t3 = 0.0, t4 = 1.0, t5 = 0.0;

  void leftMultiply(float u0, float u1, float u2, float u3, float u4, float u5) {
    const float t0 = this->t0, t1 = this->t1, t2 = this->t2;
    const float t3 = this->t3, t4 = this->t4, t5 = this->t5;

    this->t0 = t0 * u0 + t3 * u1;
    this->t1 = t1 * u0 + t4 * u1;
    this->t2 = t2 * u0 + t5 * u1 + u2;

    this->t3 = t0 * u3 + t3 * u4;
    this->t4 = t1 * u3 + t4 * u4;
    this->t5 = t2 * u3 + t5 * u4 + u5;
  }

  void invert() {
    const float t0 = this->t0, t1 = this->t1, t2 = this->t2;
    const float t3 = this->t3, t4 = this->t4, t5 = this->t5;
    const float denom = t0 * t4 - t1 * t3;

    this->t0 = t4 / denom;
    this->t1 = t1 / -denom;
    this->t2 = (t2 * t4 - t1 * t5) / -denom;
    this->t3 = t3 / -denom;
    this->t4 = t0 / denom;
    this->t5 = (t2 * t3 - t0 * t5) / denom;
  }

  void toArray(float *arr) {
    arr[0] = this->t0;
    arr[1] = this->t1;
    arr[2] = this->t2;
    arr[3] = this->t3;
    arr[4] = this->t4;
    arr[5] = this->t5;
  }
};

typedef struct AugmentationParam {
  string rand_type;
  bool   should_exp;
  float  mean;
  float  spread;
  float  prob;
} AugmentationParam;

class AugmentationParams {
public:

  int crop_height;
  int crop_width;
  struct AugmentationParam translate;
  struct AugmentationParam rotate;
  struct AugmentationParam zoom;
  struct AugmentationParam squeeze;

  bool has_translate = false;
  bool has_rotate    = false;
  bool has_zoom      = false;
  bool has_squeeze   = false;

  inline AugmentationParams(int                crop_height,
                            int                crop_width,
                            std::vector<string>params_name,
                            std::vector<string>params_rand_type,
                            std::vector<bool>  params_exp,
                            std::vector<float> params_mean,
                            std::vector<float> params_spread,
                            std::vector<float> params_prob) {
    this->crop_height = crop_height;
    this->crop_width  = crop_width;

    for (int i = 0; i < params_name.size(); i++) {
      const string name       = params_name[i];
      const string rand_type  = params_rand_type[i];
      const bool   should_exp = params_exp[i];
      const float  mean       = params_mean[i];
      const float  spread     = params_spread[i];
      const float  prob       = params_prob[i];

      struct AugmentationParam param = { rand_type, should_exp, mean, spread, prob };

      if (name == "translate") {
        this->translate     = param;
        this->has_translate = true;
      } else if (name == "rotate") {
        this->rotate     = param;
        this->has_rotate = true;
      } else if (name == "zoom") {
        this->zoom     = param;
        this->has_zoom = true;
      }  else if (name == "squeeze") {
        this->squeeze     = param;
        this->has_squeeze = true;
      } else {
        std::cout << "Ignoring unknown augmentation parameter: " << name <<
          std::endl;
      }
    }
  }

  bool do_spatial_transform() {
    return has_translate || has_rotate || has_zoom || has_squeeze;
  }
};


class AugmentationCoeffs {
public:

  float translate_x = TRANSLATE_DEFAULT;
  float translate_y = TRANSLATE_DEFAULT;
  float rotate      = ROTATE_DEFAULT;
  float zoom_x      = ZOOM_DEFAULT;
  float zoom_y      = ZOOM_DEFAULT;

  void clear() {
    translate_x = TRANSLATE_DEFAULT;
    translate_y = TRANSLATE_DEFAULT;
    rotate      = ROTATE_DEFAULT;
    zoom_x      = ZOOM_DEFAULT;
    zoom_y      = ZOOM_DEFAULT;
  }

  void combine_with(const AugmentationCoeffs& coeffs) {
    if (!essentiallyEqual(coeffs.translate_x, TRANSLATE_DEFAULT)) {
      if (!essentiallyEqual(translate_x, TRANSLATE_DEFAULT)) {
        translate_x *= coeffs.translate_x;
      } else {
        translate_x = coeffs.translate_x;
      }
    }

    if (!essentiallyEqual(coeffs.translate_y, TRANSLATE_DEFAULT)) {
      if (!essentiallyEqual(translate_y, TRANSLATE_DEFAULT)) {
        translate_y *= coeffs.translate_y;
      } else {
        translate_y = coeffs.translate_y;
      }
    }

    if (!essentiallyEqual(coeffs.rotate, ROTATE_DEFAULT)) {
      if (!essentiallyEqual(rotate, ROTATE_DEFAULT)) {
        rotate *= coeffs.rotate;
      } else {
        rotate = coeffs.rotate;
      }
    }

    if (!essentiallyEqual(coeffs.zoom_x, ZOOM_DEFAULT)) {
      if (!essentiallyEqual(zoom_x, ZOOM_DEFAULT)) {
        zoom_x *= coeffs.zoom_x;
      } else {
        zoom_x = coeffs.zoom_x;
      }
    }

    if (!essentiallyEqual(coeffs.zoom_y, ZOOM_DEFAULT)) {
      if (!essentiallyEqual(zoom_y, ZOOM_DEFAULT)) {
        zoom_y *= coeffs.zoom_y;
      } else {
        zoom_y = coeffs.zoom_y;
      }
    }
  }

  bool do_spatial_transform() {
    return !essentiallyEqual(translate_x, TRANSLATE_DEFAULT) ||
           !essentiallyEqual(translate_y, TRANSLATE_DEFAULT) ||
           !essentiallyEqual(rotate, ROTATE_DEFAULT) ||
           !essentiallyEqual(zoom_x, ZOOM_DEFAULT) ||
           !essentiallyEqual(zoom_y, ZOOM_DEFAULT);
  }

  void store_spatial_matrix(const int  out_height,
                            const int  out_width,
                            const int  src_height,
                            const int  src_width,
                            float     *output,
                            const bool invert = false) {
    // TODO: Memoize this

    TransMat t = TransMat();

    t.leftMultiply(1, 0, -0.5 * out_width, 0, 1, -0.5 * out_height);

    if (!essentiallyEqual(rotate, ROTATE_DEFAULT)) {
      const float cos_ = cos(rotate);
      const float sin_ = sin(rotate);
      t.leftMultiply(cos_, -sin_, 0, sin_, cos_, 0);
    }

    if (!essentiallyEqual(translate_x, TRANSLATE_DEFAULT) || !essentiallyEqual(translate_y, TRANSLATE_DEFAULT)) {
      t.leftMultiply(1, 0, translate_x * out_width, 0, 1, translate_y * out_height);
    }

    if (!essentiallyEqual(zoom_x, ZOOM_DEFAULT) || !essentiallyEqual(zoom_y, ZOOM_DEFAULT)) {
      t.leftMultiply(1.0 / zoom_x, 0, 0, 0, 1.0 / zoom_y, 0);
    }

    t.leftMultiply(1, 0, 0.5 * src_width, 0, 1, 0.5 * src_height);

    if (invert) {
      t.invert();
    }

    t.toArray(output);
  }
};

float              rng_generate(const AugmentationParam& param,
                                const float              default_value);

void               generate_spatial_coeffs(const AugmentationParams& params,
                                           AugmentationCoeffs      & coeff);

AugmentationCoeffs generate_valid_spatial_coeffs(
  const AugmentationParams& params,
  AugmentationCoeffs      & coeff,
  const int                 src_height,
  const int                 src_width);

void ComputeDataAugmentation(
  OpKernelContext *ctx,
  const GPUDevice& device,
  typename TTypes<float, 4>::ConstTensor input,
  AugmentationParams& params,
  typename TTypes<float, 4>::Tensor output,
  std::vector<AugmentationCoeffs>& in_coeffs,
  std::vector<AugmentationCoeffs>& out_coeffs);
} // end namespace tensorflow

#endif  // FLOWNET_DATA_AUG_H_
