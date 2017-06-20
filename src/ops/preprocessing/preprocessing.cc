#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using shape_inference::DimensionHandle;

Status SetOutputToSizedImage(InferenceContext* c) {
  ShapeHandle input;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input));
  DimensionHandle batch = c->Dim(input, 0);
  DimensionHandle depth = c->Dim(input, 3);
  std::vector<int32> crop_;
  c->GetAttr("crop", &crop_);
  DimensionHandle height = c->MakeDim(crop_[0]);
  DimensionHandle width  = c->MakeDim(crop_[1]);
  c->set_output(0, c->MakeShape({batch, height, width, depth}));
  return Status::OK();
}

REGISTER_OP("SpatialTransform")
    .Input("images: dtype")
    .Input("transforms: float32")
    .Attr("crop: list(int) >= 2")
    .Attr("dtype: {uint8, int32, int64, float32, float64}")
    .Output("transformed_images: dtype")
    .SetShapeFn(SetOutputToSizedImage);

REGISTER_OP("FlowAugmentation")
    .Input("flows: float32")
    .Input("transforms_a: float32")
    .Input("transforms_b_inv: float32")
    .Attr("crop: list(int) >= 2")
    .Output("transformed_flows: float32")
    .SetShapeFn(SetOutputToSizedImage);

}  // namespace tensorflow
