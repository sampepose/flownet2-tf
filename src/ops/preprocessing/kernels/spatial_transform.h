#ifndef FLOWNET_SPATIAL_TRANS_H_
#define FLOWNET_SPATIAL_TRANS_H_

// See docs in ../ops/image_ops.cc.

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

#define my_min(a, b)        \
  ({ __typeof__(a)_a = (a); \
     __typeof__(b)_b = (b); \
     _a < _b ? _a : _b; })

namespace tensorflow {
namespace generator {
using Eigen::array;
using Eigen::DenseIndex;

template<typename Device, typename T>
class ProjectiveGenerator {
private:

  typename TTypes<T, 4>::ConstTensor input_;
  typename TTypes<float>::ConstMatrix transforms_;

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  float clamp(float f, float a, float b) const {
    return fmaxf(a, fminf(f, b));
  }

public:

  static const int kNumParameters = 8;

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  ProjectiveGenerator(typename TTypes<T, 4>::ConstTensor input,
                      typename TTypes<float>::ConstMatrix transforms)
    : input_(input), transforms_(transforms) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
  operator()(const array<DenseIndex, 4>& coords) const {
    const int64  output_y  = coords[1];
    const int64  output_x  = coords[2];
    const float *transform =
      transforms_.dimension(0) == 1
      ? transforms_.data()
      : &transforms_.data()[transforms_.dimension(1) * coords[0]];
    float projection = transform[6] * output_x + transform[7] * output_y + 1.f;
    float input_x    =
      (transform[0] * output_x + transform[1] * output_y + transform[2]) /
      projection;
    float input_y =
      (transform[3] * output_x + transform[4] * output_y + transform[5]) /
      projection;

    // Ensure that floor(xpos)+1 is still valid
    input_x = clamp(input_x, 0.0f, (float)input_.dimension(2) - 1.05f);
    input_y = clamp(input_y, 0.0f, (float)input_.dimension(1) - 1.05f);

    // Interpolate
    long tlx = long(floor(input_x));
    long tly = long(floor(input_y));
    long trx = my_min(tlx + 1, input_.dimension(2));
    long bly = my_min(tly + 1, input_.dimension(1));

    T sampleTL = input_(array<DenseIndex, 4>{ coords[0], tly, tlx, coords[3] });
    T sampleTR = input_(array<DenseIndex, 4>{ coords[0], tly, trx, coords[3] });
    T sampleBL = input_(array<DenseIndex, 4>{ coords[0], bly, tlx, coords[3] });
    T sampleBR = input_(array<DenseIndex, 4>{ coords[0], bly, trx, coords[3] });

    float xdist = input_x - tlx;
    float ydist = input_y - tly;

    return (1 - xdist) * (1 - ydist) * sampleTL
           + (xdist) * (ydist) * sampleBR
           + (1 - xdist) * (ydist) * sampleBL
           + (xdist) * (1 - ydist) * sampleTR;
  }
};
} // end namespace generator

// NOTE(ringwalt): We MUST wrap the generate() call in a functor and explicitly
// instantiate the functor in image_ops_gpu.cu.cc. Otherwise, we will be missing
// some Eigen device code.
namespace functor {
using generator::ProjectiveGenerator;

template<typename Device, typename T>
struct FillSpatialTransform {
  typedef typename TTypes<T, 4>::Tensor          OutputType;
  typedef typename TTypes<T, 4>::ConstTensor     InputType;
  typedef typename TTypes<float, 2>::ConstTensor TransformsType;

  FillSpatialTransform() {}

  EIGEN_ALWAYS_INLINE
  void operator()(const Device& device, OutputType *output,
                  const InputType& images,
                  const TransformsType& transform) const {
    output->device(device) = output->generate(
      ProjectiveGenerator<Device, T>(images, transform));
  }
};
} // end namespace functor
} // end namespace tensorflow

#endif  // FLOWNET_SPATIAL_TRANS_H_
