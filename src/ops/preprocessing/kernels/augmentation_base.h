#ifndef AUGMENTATION_LAYER_BASE_H_
#define AUGMENTATION_LAYER_BASE_H_

#include "tensorflow/core/framework/tensor_types.h"

#include <iostream>
#include <string>
#include <vector>

namespace tensorflow {
template<typename T>
class OptionalType {
  public:
    OptionalType(const T default_value) : default_value(default_value), has_value(false) {}

    operator bool() const {
      return has_value;
    }

    OptionalType& operator=(T val) {
      has_value = true;
      value     = val;
      return *this;
    }

    const T operator()() const {
      return has_value ? value : default_value;
    }

    void clear() {
      has_value = false;
    }

    const T get_default() {
      return default_value;
    }

  private:
    T value;
    bool has_value;
    const T default_value;
};

class AugmentationCoeff {
  public:
    // Spatial Types
    OptionalType<float>dx;
    OptionalType<float>dy;
    OptionalType<float>angle;
    OptionalType<float>zoom_x;
    OptionalType<float>zoom_y;

    // Effect Types
    OptionalType<float>noise;

    AugmentationCoeff() : dx(0.0), dy(0.0), angle(0.0), zoom_x(1.0), zoom_y(1.0), noise(0.0) {}

    AugmentationCoeff(const AugmentationCoeff& coeff) : AugmentationCoeff() {
      replace_with(coeff);
    }

    void clear();

    void combine_with(const AugmentationCoeff& coeff);

    void replace_with(const AugmentationCoeff& coeff);
};

typedef struct AugmentationParam {
  std::string rand_type;
  bool        should_exp;
  float       mean;
  float       spread;
  float       prob;
} AugmentationParam;

class AugmentationParams {
  public:
    int crop_height;
    int crop_width;

    // Spatial options
    OptionalType<struct AugmentationParam>translate;
    OptionalType<struct AugmentationParam>rotate;
    OptionalType<struct AugmentationParam>zoom;
    OptionalType<struct AugmentationParam>squeeze;

    // Effect options
    OptionalType<struct AugmentationParam>noise;

    inline AugmentationParams(int                     crop_height,
                              int                     crop_width,
                              std::vector<std::string>params_name,
                              std::vector<std::string>params_rand_type,
                              std::vector<bool>       params_exp,
                              std::vector<float>      params_mean,
                              std::vector<float>      params_spread,
                              std::vector<float>      params_prob) :
      crop_height(crop_height),
      crop_width(crop_width),
      translate(AugmentationParam()),
      rotate(AugmentationParam()),
      zoom(AugmentationParam()),
      squeeze(AugmentationParam()),
      noise(AugmentationParam()) {
      for (int i = 0; i < params_name.size(); i++) {
        const std::string name      = params_name[i];
        const std::string rand_type = params_rand_type[i];
        const bool  should_exp      = params_exp[i];
        const float mean            = params_mean[i];
        const float spread          = params_spread[i];
        const float prob            = params_prob[i];

        struct AugmentationParam param = { rand_type, should_exp, mean, spread, prob };

        if (name == "translate") {
          this->translate = param;
        } else if (name == "rotate") {
          this->rotate = param;
        } else if (name == "zoom") {
          this->zoom = param;
        }  else if (name == "squeeze") {
          this->squeeze = param;
        } else if (name == "noise") {
          this->noise = param;
        } else {
          std::cout << "Ignoring unknown augmentation parameter: " << name << std::endl;
        }
      }
    }

    bool should_do_spatial_transform() {
      return this->translate || this->rotate || this->zoom || this->squeeze;
    }

    bool should_do_effects_transform() {
      return this->noise;
    }
};

class AugmentationLayerBase {
  public:
    class TransMat {
      /**
       * Translation matrix class for spatial augmentation
       * | 0 1 2 |
       * | 3 4 5 |
       */

      public:
        float t0, t1, t2;
        float t3, t4, t5;


        void fromCoeff(AugmentationCoeff *coeff,
                       int                out_width,
                       int                out_height,
                       int                src_width,
                       int                src_height);

        void     fromTensor(const float *tensor_data);

        TransMat inverse();

        void     leftMultiply(float u0,
                              float u1,
                              float u2,
                              float u3,
                              float u4,
                              float u5);

        void toIdentity();
    };

    // TODO: Class ChromaticCoeffs

    static float rng_generate(const AugmentationParam& param,
                              const float              default_value);

    static void  clear_spatial_coeffs(AugmentationCoeff& coeff);
    static void  generate_spatial_coeffs(const AugmentationParams& aug,
                                         AugmentationCoeff       & coeff);
    static void  generate_effect_coeffs(const AugmentationParams& aug,
                                        AugmentationCoeff       & coeff);
    static void  generate_valid_spatial_coeffs(const AugmentationParams& aug,
                                               AugmentationCoeff       & coeff,
                                               int                       src_width,
                                               int                       src_height,
                                               int                       out_width,
                                               int                       out_height);

    static void copy_spatial_coeffs_to_tensor(const std::vector<AugmentationCoeff>& coeff_arr,
                                              const int out_width,
                                              const int out_height,
                                              const int src_width,
                                              const int src_height,
                                              typename TTypes<float, 2>::Tensor& out,
                                              const bool invert = false);
};
} // namespace tensorflow

#endif // AUGMENTATION_LAYER_BASE_H_
