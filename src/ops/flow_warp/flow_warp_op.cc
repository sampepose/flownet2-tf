#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
REGISTER_OP("FlowWarp")
.Input("image: float32")
.Input("flow: float32")
.Output("output: float32")
.SetShapeFn(::tensorflow::shape_inference::UnchangedShape);
} // namespace tensorflow
