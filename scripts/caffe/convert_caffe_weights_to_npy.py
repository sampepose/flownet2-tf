"""
Please read README.md for usage instructions.

Extracts Caffe parameters from a given caffemodel/prototxt to a dictionary of numpy arrays,
ready for conversion to TensorFlow variables. Writes the dictionary to a .npy file.
"""
import argparse
import caffe
import numpy as np
import os
import tempfile

FLAGS = None
ARCHS = {
    'C': {
        'CAFFEMODEL': '../models/FlowNet2-C/FlowNet2-C_weights.caffemodel',
        'DEPLOY_PROTOTXT': '../models/FlowNet2-C/FlowNet2-C_deploy.prototxt.template',
        # Mappings between Caffe parameter names and TensorFlow variable names
        'PARAMS': {
            'conv1': 'conv1',
            'conv2': 'conv2',
            'conv3': 'conv3',
            'conv_redir': 'conv_redir',
            'conv3_1': 'conv3_1',
            'conv4': 'conv4',
            'conv4_1': 'conv4_1',
            'conv5': 'conv5',
            'conv5_1': 'conv5_1',
            'conv6': 'conv6',
            'conv6_1': 'conv6_1',
            'Convolution1': 'predict_flow6',
            'deconv5': 'deconv5',
            'upsample_flow6to5': 'upsample_flow6to5',
            'Convolution2': 'predict_flow5',
            'deconv4': 'deconv4',
            'upsample_flow5to4': 'upsample_flow5to4',
            'Convolution3': 'predict_flow4',
            'deconv3': 'deconv3',
            'upsample_flow4to3': 'upsample_flow4to3',
            'Convolution4': 'predict_flow3',
            'deconv2': 'deconv2',
            'upsample_flow3to2': 'upsample_flow3to2',
            'Convolution5': 'predict_flow2',
            'scale_conv1': 'scale_conv1'
        }
    },
    'S': {
        'CAFFEMODEL': '../models/FlowNet2-S/FlowNet2-S_weights.caffemodel.h5',
        'DEPLOY_PROTOTXT': '../models/FlowNet2-S/FlowNet2-S_deploy.prototxt.template',
        # Mappings between Caffe parameter names and TensorFlow variable names
        'PARAMS': {
            'conv1': 'conv1',
            'conv2': 'conv2',
            'conv3': 'conv3',
            'conv3_1': 'conv3_1',
            'conv4': 'conv4',
            'conv4_1': 'conv4_1',
            'conv5': 'conv5',
            'conv5_1': 'conv5_1',
            'conv6': 'conv6',
            'conv6_1': 'conv6_1',
            'Convolution1': 'predict_flow6',
            'deconv5': 'deconv5',
            'upsample_flow6to5': 'upsample_flow6to5',
            'Convolution2': 'predict_flow5',
            'deconv4': 'deconv4',
            'upsample_flow5to4': 'upsample_flow5to4',
            'Convolution3': 'predict_flow4',
            'deconv3': 'deconv3',
            'upsample_flow4to3': 'upsample_flow4to3',
            'Convolution4': 'predict_flow3',
            'deconv2': 'deconv2',
            'upsample_flow3to2': 'upsample_flow3to2',
            'Convolution5': 'predict_flow2',
            'scale_conv1': 'scale_conv1'
        }
    },
    # TODO: 2
}
arch = None

# Setup variables to be injected into prototxt.template
# For now, use the dimensions of the Flying Chair Dataset
vars = {}
vars['TARGET_WIDTH'] = vars['ADAPTED_WIDTH'] = 512
vars['TARGET_HEIGHT'] = vars['ADAPTED_HEIGHT'] = 384
vars['SCALE_WIDTH'] = vars['SCALE_HEIGHT'] = 1.0

def main():
    # Create tempfile to hold prototxt
    tmp = tempfile.NamedTemporaryFile(mode='w', delete=True)

    # Parse prototxt and inject `vars`
    proto = open(arch['DEPLOY_PROTOTXT']).readlines()
    for line in proto:
        for key, value in vars.items():
            tag = "$%s$" % key
            line = line.replace(tag, str(value))
        tmp.write(line)
    tmp.flush()

    # Instantiate Caffe Model
    net = caffe.Net(tmp.name, arch['CAFFEMODEL'], caffe.TEST)

    out = {}
    for (caffe_param, tf_param) in arch['PARAMS'].items():
        # Caffe stores weights as (channels_out, channels_in, h, w)
        # but TF expects (h, w, channels_in, channels_out)
        out[tf_param+ '/weights'] = net.params[caffe_param][0].data.transpose((2, 3, 1, 0))
        out[tf_param+ '/biases'] = net.params[caffe_param][1].data

    np.save(FLAGS.out, out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help='Output file path, eg /foo/bar.npy'
    )
    parser.add_argument(
        '--arch',
        type=str,
        choices=['C', 'S', '2'],
        required=True,
        help='Name of the FlowNet arch: C or S or 2'
    )
    FLAGS = parser.parse_args()
    arch = ARCHS[FLAGS.arch]
    main()
