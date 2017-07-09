This directory contains the scripts to convert weights from FlowNet2.0 in Caffe to TensorFlow variables.

Instructions:
1) Clone and compile FlowNet2.0 from here: https://github.com/lmb-freiburg/flownet2

2) Run `source set-env.sh` from the root directory of the flownet2 repository

3) Move the contents of this directory (scripts/caffe) to flownet2/scripts.

4) run the bash script: `sh ./convert_caffe_weights_to_tf.sh {C|S|CS|CSS|CSS-ft-sd|SD|2}`

The weights will be saved in the scripts directory alongside the bash script.
