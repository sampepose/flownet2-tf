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
            'conv1': 'FlowNetC/conv1',
            'conv2': 'FlowNetC/conv2',
            'conv3': 'FlowNetC/conv3',
            'conv_redir': 'FlowNetC/conv_redir',
            'conv3_1': 'FlowNetC/conv3_1',
            'conv4': 'FlowNetC/conv4',
            'conv4_1': 'FlowNetC/conv4_1',
            'conv5': 'FlowNetC/conv5',
            'conv5_1': 'FlowNetC/conv5_1',
            'conv6': 'FlowNetC/conv6',
            'conv6_1': 'FlowNetC/conv6_1',
            'Convolution1': 'FlowNetC/predict_flow6',
            'deconv5': 'FlowNetC/deconv5',
            'upsample_flow6to5': 'FlowNetC/upsample_flow6to5',
            'Convolution2': 'FlowNetC/predict_flow5',
            'deconv4': 'FlowNetC/deconv4',
            'upsample_flow5to4': 'FlowNetC/upsample_flow5to4',
            'Convolution3': 'FlowNetC/predict_flow4',
            'deconv3': 'FlowNetC/deconv3',
            'upsample_flow4to3': 'FlowNetC/upsample_flow4to3',
            'Convolution4': 'FlowNetC/predict_flow3',
            'deconv2': 'FlowNetC/deconv2',
            'upsample_flow3to2': 'FlowNetC/upsample_flow3to2',
            'Convolution5': 'FlowNetC/predict_flow2',
        }
    },
    'S': {
        'CAFFEMODEL': '../models/FlowNet2-S/FlowNet2-S_weights.caffemodel.h5',
        'DEPLOY_PROTOTXT': '../models/FlowNet2-S/FlowNet2-S_deploy.prototxt.template',
        # Mappings between Caffe parameter names and TensorFlow variable names
        'PARAMS': {
            'conv1': 'FlowNetS/conv1',
            'conv2': 'FlowNetS/conv2',
            'conv3': 'FlowNetS/conv3',
            'conv3_1': 'FlowNetS/conv3_1',
            'conv4': 'FlowNetS/conv4',
            'conv4_1': 'FlowNetS/conv4_1',
            'conv5': 'FlowNetS/conv5',
            'conv5_1': 'FlowNetS/conv5_1',
            'conv6': 'FlowNetS/conv6',
            'conv6_1': 'FlowNetS/conv6_1',
            'Convolution1': 'FlowNetS/predict_flow6',
            'deconv5': 'FlowNetS/deconv5',
            'upsample_flow6to5': 'FlowNetS/upsample_flow6to5',
            'Convolution2': 'FlowNetS/predict_flow5',
            'deconv4': 'FlowNetS/deconv4',
            'upsample_flow5to4': 'FlowNetS/upsample_flow5to4',
            'Convolution3': 'FlowNetS/predict_flow4',
            'deconv3': 'FlowNetS/deconv3',
            'upsample_flow4to3': 'FlowNetS/upsample_flow4to3',
            'Convolution4': 'FlowNetS/predict_flow3',
            'deconv2': 'FlowNetS/deconv2',
            'upsample_flow3to2': 'FlowNetS/upsample_flow3to2',
            'Convolution5': 'FlowNetS/predict_flow2',
        }
    },
    'CS': {
        'CAFFEMODEL': '../models/FlowNet2-CS/FlowNet2-CS_weights.caffemodel',
        'DEPLOY_PROTOTXT': '../models/FlowNet2-CS/FlowNet2-CS_deploy.prototxt.template',
        # Mappings between Caffe parameter names and TensorFlow variable names
        'PARAMS': {
            # Net C
            'conv1': 'FlowNetCS/FlowNetC/conv1',
            'conv2': 'FlowNetCS/FlowNetC/conv2',
            'conv3': 'FlowNetCS/FlowNetC/conv3',
            'conv_redir': 'FlowNetCS/FlowNetC/conv_redir',
            'conv3_1': 'FlowNetCS/FlowNetC/conv3_1',
            'conv4': 'FlowNetCS/FlowNetC/conv4',
            'conv4_1': 'FlowNetCS/FlowNetC/conv4_1',
            'conv5': 'FlowNetCS/FlowNetC/conv5',
            'conv5_1': 'FlowNetCS/FlowNetC/conv5_1',
            'conv6': 'FlowNetCS/FlowNetC/conv6',
            'conv6_1': 'FlowNetCS/FlowNetC/conv6_1',
            'Convolution1': 'FlowNetCS/FlowNetC/predict_flow6',
            'deconv5': 'FlowNetCS/FlowNetC/deconv5',
            'upsample_flow6to5': 'FlowNetCS/FlowNetC/upsample_flow6to5',
            'Convolution2': 'FlowNetCS/FlowNetC/predict_flow5',
            'deconv4': 'FlowNetCS/FlowNetC/deconv4',
            'upsample_flow5to4': 'FlowNetCS/FlowNetC/upsample_flow5to4',
            'Convolution3': 'FlowNetCS/FlowNetC/predict_flow4',
            'deconv3': 'FlowNetCS/FlowNetC/deconv3',
            'upsample_flow4to3': 'FlowNetCS/FlowNetC/upsample_flow4to3',
            'Convolution4': 'FlowNetCS/FlowNetC/predict_flow3',
            'deconv2': 'FlowNetCS/FlowNetC/deconv2',
            'upsample_flow3to2': 'FlowNetCS/FlowNetC/upsample_flow3to2',
            'Convolution5': 'FlowNetCS/FlowNetC/predict_flow2',

            # Net S
            'net2_conv1': 'FlowNetCS/FlowNetS/conv1',
            'net2_conv2': 'FlowNetCS/FlowNetS/conv2',
            'net2_conv3': 'FlowNetCS/FlowNetS/conv3',
            'net2_conv3_1': 'FlowNetCS/FlowNetS/conv3_1',
            'net2_conv4': 'FlowNetCS/FlowNetS/conv4',
            'net2_conv4_1': 'FlowNetCS/FlowNetS/conv4_1',
            'net2_conv5': 'FlowNetCS/FlowNetS/conv5',
            'net2_conv5_1': 'FlowNetCS/FlowNetS/conv5_1',
            'net2_conv6': 'FlowNetCS/FlowNetS/conv6',
            'net2_conv6_1': 'FlowNetCS/FlowNetS/conv6_1',
            'net2_predict_conv6': 'FlowNetCS/FlowNetS/predict_flow6',
            'net2_deconv5': 'FlowNetCS/FlowNetS/deconv5',
            'net2_net2_upsample_flow6to5': 'FlowNetCS/FlowNetS/upsample_flow6to5',
            'net2_predict_conv5': 'FlowNetCS/FlowNetS/predict_flow5',
            'net2_deconv4': 'FlowNetCS/FlowNetS/deconv4',
            'net2_net2_upsample_flow5to4': 'FlowNetCS/FlowNetS/upsample_flow5to4',
            'net2_predict_conv4': 'FlowNetCS/FlowNetS/predict_flow4',
            'net2_deconv3': 'FlowNetCS/FlowNetS/deconv3',
            'net2_net2_upsample_flow4to3': 'FlowNetCS/FlowNetS/upsample_flow4to3',
            'net2_predict_conv3': 'FlowNetCS/FlowNetS/predict_flow3',
            'net2_deconv2': 'FlowNetCS/FlowNetS/deconv2',
            'net2_net2_upsample_flow3to2': 'FlowNetCS/FlowNetS/upsample_flow3to2',
            'net2_predict_conv2': 'FlowNetCS/FlowNetS/predict_flow2',
        }
    },
    'CSS': {
        'CAFFEMODEL': '../models/FlowNet2-CSS/FlowNet2-CSS_weights.caffemodel.h5',
        'DEPLOY_PROTOTXT': '../models/FlowNet2-CSS/FlowNet2-CSS_deploy.prototxt.template',
        # Mappings between Caffe parameter names and TensorFlow variable names
        'PARAMS': {
            # Net C
            'conv1': 'FlowNetCSS/FlowNetCS/FlowNetC/conv1',
            'conv2': 'FlowNetCSS/FlowNetCS/FlowNetC/conv2',
            'conv3': 'FlowNetCSS/FlowNetCS/FlowNetC/conv3',
            'conv_redir': 'FlowNetCSS/FlowNetCS/FlowNetC/conv_redir',
            'conv3_1': 'FlowNetCSS/FlowNetCS/FlowNetC/conv3_1',
            'conv4': 'FlowNetCSS/FlowNetCS/FlowNetC/conv4',
            'conv4_1': 'FlowNetCSS/FlowNetCS/FlowNetC/conv4_1',
            'conv5': 'FlowNetCSS/FlowNetCS/FlowNetC/conv5',
            'conv5_1': 'FlowNetCSS/FlowNetCS/FlowNetC/conv5_1',
            'conv6': 'FlowNetCSS/FlowNetCS/FlowNetC/conv6',
            'conv6_1': 'FlowNetCSS/FlowNetCS/FlowNetC/conv6_1',
            'Convolution1': 'FlowNetCSS/FlowNetCS/FlowNetC/predict_flow6',
            'deconv5': 'FlowNetCSS/FlowNetCS/FlowNetC/deconv5',
            'upsample_flow6to5': 'FlowNetCSS/FlowNetCS/FlowNetC/upsample_flow6to5',
            'Convolution2': 'FlowNetCSS/FlowNetCS/FlowNetC/predict_flow5',
            'deconv4': 'FlowNetCSS/FlowNetCS/FlowNetC/deconv4',
            'upsample_flow5to4': 'FlowNetCSS/FlowNetCS/FlowNetC/upsample_flow5to4',
            'Convolution3': 'FlowNetCSS/FlowNetCS/FlowNetC/predict_flow4',
            'deconv3': 'FlowNetCSS/FlowNetCS/FlowNetC/deconv3',
            'upsample_flow4to3': 'FlowNetCSS/FlowNetCS/FlowNetC/upsample_flow4to3',
            'Convolution4': 'FlowNetCSS/FlowNetCS/FlowNetC/predict_flow3',
            'deconv2': 'FlowNetCSS/FlowNetCS/FlowNetC/deconv2',
            'upsample_flow3to2': 'FlowNetCSS/FlowNetCS/FlowNetC/upsample_flow3to2',
            'Convolution5': 'FlowNetCSS/FlowNetCS/FlowNetC/predict_flow2',

            # Net S 1
            'net2_conv1': 'FlowNetCSS/FlowNetCS/FlowNetS/conv1',
            'net2_conv2': 'FlowNetCSS/FlowNetCS/FlowNetS/conv2',
            'net2_conv3': 'FlowNetCSS/FlowNetCS/FlowNetS/conv3',
            'net2_conv3_1': 'FlowNetCSS/FlowNetCS/FlowNetS/conv3_1',
            'net2_conv4': 'FlowNetCSS/FlowNetCS/FlowNetS/conv4',
            'net2_conv4_1': 'FlowNetCSS/FlowNetCS/FlowNetS/conv4_1',
            'net2_conv5': 'FlowNetCSS/FlowNetCS/FlowNetS/conv5',
            'net2_conv5_1': 'FlowNetCSS/FlowNetCS/FlowNetS/conv5_1',
            'net2_conv6': 'FlowNetCSS/FlowNetCS/FlowNetS/conv6',
            'net2_conv6_1': 'FlowNetCSS/FlowNetCS/FlowNetS/conv6_1',
            'net2_predict_conv6': 'FlowNetCSS/FlowNetCS/FlowNetS/predict_flow6',
            'net2_deconv5': 'FlowNetCSS/FlowNetCS/FlowNetS/deconv5',
            'net2_net2_upsample_flow6to5': 'FlowNetCSS/FlowNetCS/FlowNetS/upsample_flow6to5',
            'net2_predict_conv5': 'FlowNetCSS/FlowNetCS/FlowNetS/predict_flow5',
            'net2_deconv4': 'FlowNetCSS/FlowNetCS/FlowNetS/deconv4',
            'net2_net2_upsample_flow5to4': 'FlowNetCSS/FlowNetCS/FlowNetS/upsample_flow5to4',
            'net2_predict_conv4': 'FlowNetCSS/FlowNetCS/FlowNetS/predict_flow4',
            'net2_deconv3': 'FlowNetCSS/FlowNetCS/FlowNetS/deconv3',
            'net2_net2_upsample_flow4to3': 'FlowNetCSS/FlowNetCS/FlowNetS/upsample_flow4to3',
            'net2_predict_conv3': 'FlowNetCSS/FlowNetCS/FlowNetS/predict_flow3',
            'net2_deconv2': 'FlowNetCSS/FlowNetCS/FlowNetS/deconv2',
            'net2_net2_upsample_flow3to2': 'FlowNetCSS/FlowNetCS/FlowNetS/upsample_flow3to2',
            'net2_predict_conv2': 'FlowNetCSS/FlowNetCS/FlowNetS/predict_flow2',

            # Net S 2
            'net3_conv1': 'FlowNetCSS/FlowNetS/conv1',
            'net3_conv2': 'FlowNetCSS/FlowNetS/conv2',
            'net3_conv3': 'FlowNetCSS/FlowNetS/conv3',
            'net3_conv3_1': 'FlowNetCSS/FlowNetS/conv3_1',
            'net3_conv4': 'FlowNetCSS/FlowNetS/conv4',
            'net3_conv4_1': 'FlowNetCSS/FlowNetS/conv4_1',
            'net3_conv5': 'FlowNetCSS/FlowNetS/conv5',
            'net3_conv5_1': 'FlowNetCSS/FlowNetS/conv5_1',
            'net3_conv6': 'FlowNetCSS/FlowNetS/conv6',
            'net3_conv6_1': 'FlowNetCSS/FlowNetS/conv6_1',
            'net3_predict_conv6': 'FlowNetCSS/FlowNetS/predict_flow6',
            'net3_deconv5': 'FlowNetCSS/FlowNetS/deconv5',
            'net3_net3_upsample_flow6to5': 'FlowNetCSS/FlowNetS/upsample_flow6to5',
            'net3_predict_conv5': 'FlowNetCSS/FlowNetS/predict_flow5',
            'net3_deconv4': 'FlowNetCSS/FlowNetS/deconv4',
            'net3_net3_upsample_flow5to4': 'FlowNetCSS/FlowNetS/upsample_flow5to4',
            'net3_predict_conv4': 'FlowNetCSS/FlowNetS/predict_flow4',
            'net3_deconv3': 'FlowNetCSS/FlowNetS/deconv3',
            'net3_net3_upsample_flow4to3': 'FlowNetCSS/FlowNetS/upsample_flow4to3',
            'net3_predict_conv3': 'FlowNetCSS/FlowNetS/predict_flow3',
            'net3_deconv2': 'FlowNetCSS/FlowNetS/deconv2',
            'net3_net3_upsample_flow3to2': 'FlowNetCSS/FlowNetS/upsample_flow3to2',
            'net3_predict_conv2': 'FlowNetCSS/FlowNetS/predict_flow2',
        },
    },
    'CSS-ft-sd': {
        'CAFFEMODEL': '../models/FlowNet2-CSS-ft-sd/FlowNet2-CSS-ft-sd_weights.caffemodel.h5',
        'DEPLOY_PROTOTXT': '../models/FlowNet2-CSS-ft-sd/FlowNet2-CSS-ft-sd_deploy.prototxt.template',
        # Mappings between Caffe parameter names and TensorFlow variable names
        'PARAMS': {
            # Net C
            'conv1': 'FlowNetCSS/FlowNetCS/FlowNetC/conv1',
            'conv2': 'FlowNetCSS/FlowNetCS/FlowNetC/conv2',
            'conv3': 'FlowNetCSS/FlowNetCS/FlowNetC/conv3',
            'conv_redir': 'FlowNetCSS/FlowNetCS/FlowNetC/conv_redir',
            'conv3_1': 'FlowNetCSS/FlowNetCS/FlowNetC/conv3_1',
            'conv4': 'FlowNetCSS/FlowNetCS/FlowNetC/conv4',
            'conv4_1': 'FlowNetCSS/FlowNetCS/FlowNetC/conv4_1',
            'conv5': 'FlowNetCSS/FlowNetCS/FlowNetC/conv5',
            'conv5_1': 'FlowNetCSS/FlowNetCS/FlowNetC/conv5_1',
            'conv6': 'FlowNetCSS/FlowNetCS/FlowNetC/conv6',
            'conv6_1': 'FlowNetCSS/FlowNetCS/FlowNetC/conv6_1',
            'Convolution1': 'FlowNetCSS/FlowNetCS/FlowNetC/predict_flow6',
            'deconv5': 'FlowNetCSS/FlowNetCS/FlowNetC/deconv5',
            'upsample_flow6to5': 'FlowNetCSS/FlowNetCS/FlowNetC/upsample_flow6to5',
            'Convolution2': 'FlowNetCSS/FlowNetCS/FlowNetC/predict_flow5',
            'deconv4': 'FlowNetCSS/FlowNetCS/FlowNetC/deconv4',
            'upsample_flow5to4': 'FlowNetCSS/FlowNetCS/FlowNetC/upsample_flow5to4',
            'Convolution3': 'FlowNetCSS/FlowNetCS/FlowNetC/predict_flow4',
            'deconv3': 'FlowNetCSS/FlowNetCS/FlowNetC/deconv3',
            'upsample_flow4to3': 'FlowNetCSS/FlowNetCS/FlowNetC/upsample_flow4to3',
            'Convolution4': 'FlowNetCSS/FlowNetCS/FlowNetC/predict_flow3',
            'deconv2': 'FlowNetCSS/FlowNetCS/FlowNetC/deconv2',
            'upsample_flow3to2': 'FlowNetCSS/FlowNetCS/FlowNetC/upsample_flow3to2',
            'Convolution5': 'FlowNetCSS/FlowNetCS/FlowNetC/predict_flow2',

            # Net S 1
            'net2_conv1': 'FlowNetCSS/FlowNetCS/FlowNetS/conv1',
            'net2_conv2': 'FlowNetCSS/FlowNetCS/FlowNetS/conv2',
            'net2_conv3': 'FlowNetCSS/FlowNetCS/FlowNetS/conv3',
            'net2_conv3_1': 'FlowNetCSS/FlowNetCS/FlowNetS/conv3_1',
            'net2_conv4': 'FlowNetCSS/FlowNetCS/FlowNetS/conv4',
            'net2_conv4_1': 'FlowNetCSS/FlowNetCS/FlowNetS/conv4_1',
            'net2_conv5': 'FlowNetCSS/FlowNetCS/FlowNetS/conv5',
            'net2_conv5_1': 'FlowNetCSS/FlowNetCS/FlowNetS/conv5_1',
            'net2_conv6': 'FlowNetCSS/FlowNetCS/FlowNetS/conv6',
            'net2_conv6_1': 'FlowNetCSS/FlowNetCS/FlowNetS/conv6_1',
            'net2_predict_conv6': 'FlowNetCSS/FlowNetCS/FlowNetS/predict_flow6',
            'net2_deconv5': 'FlowNetCSS/FlowNetCS/FlowNetS/deconv5',
            'net2_net2_upsample_flow6to5': 'FlowNetCSS/FlowNetCS/FlowNetS/upsample_flow6to5',
            'net2_predict_conv5': 'FlowNetCSS/FlowNetCS/FlowNetS/predict_flow5',
            'net2_deconv4': 'FlowNetCSS/FlowNetCS/FlowNetS/deconv4',
            'net2_net2_upsample_flow5to4': 'FlowNetCSS/FlowNetCS/FlowNetS/upsample_flow5to4',
            'net2_predict_conv4': 'FlowNetCSS/FlowNetCS/FlowNetS/predict_flow4',
            'net2_deconv3': 'FlowNetCSS/FlowNetCS/FlowNetS/deconv3',
            'net2_net2_upsample_flow4to3': 'FlowNetCSS/FlowNetCS/FlowNetS/upsample_flow4to3',
            'net2_predict_conv3': 'FlowNetCSS/FlowNetCS/FlowNetS/predict_flow3',
            'net2_deconv2': 'FlowNetCSS/FlowNetCS/FlowNetS/deconv2',
            'net2_net2_upsample_flow3to2': 'FlowNetCSS/FlowNetCS/FlowNetS/upsample_flow3to2',
            'net2_predict_conv2': 'FlowNetCSS/FlowNetCS/FlowNetS/predict_flow2',

            # Net S 2
            'net3_conv1': 'FlowNetCSS/FlowNetS/conv1',
            'net3_conv2': 'FlowNetCSS/FlowNetS/conv2',
            'net3_conv3': 'FlowNetCSS/FlowNetS/conv3',
            'net3_conv3_1': 'FlowNetCSS/FlowNetS/conv3_1',
            'net3_conv4': 'FlowNetCSS/FlowNetS/conv4',
            'net3_conv4_1': 'FlowNetCSS/FlowNetS/conv4_1',
            'net3_conv5': 'FlowNetCSS/FlowNetS/conv5',
            'net3_conv5_1': 'FlowNetCSS/FlowNetS/conv5_1',
            'net3_conv6': 'FlowNetCSS/FlowNetS/conv6',
            'net3_conv6_1': 'FlowNetCSS/FlowNetS/conv6_1',
            'net3_predict_conv6': 'FlowNetCSS/FlowNetS/predict_flow6',
            'net3_deconv5': 'FlowNetCSS/FlowNetS/deconv5',
            'net3_net3_upsample_flow6to5': 'FlowNetCSS/FlowNetS/upsample_flow6to5',
            'net3_predict_conv5': 'FlowNetCSS/FlowNetS/predict_flow5',
            'net3_deconv4': 'FlowNetCSS/FlowNetS/deconv4',
            'net3_net3_upsample_flow5to4': 'FlowNetCSS/FlowNetS/upsample_flow5to4',
            'net3_predict_conv4': 'FlowNetCSS/FlowNetS/predict_flow4',
            'net3_deconv3': 'FlowNetCSS/FlowNetS/deconv3',
            'net3_net3_upsample_flow4to3': 'FlowNetCSS/FlowNetS/upsample_flow4to3',
            'net3_predict_conv3': 'FlowNetCSS/FlowNetS/predict_flow3',
            'net3_deconv2': 'FlowNetCSS/FlowNetS/deconv2',
            'net3_net3_upsample_flow3to2': 'FlowNetCSS/FlowNetS/upsample_flow3to2',
            'net3_predict_conv2': 'FlowNetCSS/FlowNetS/predict_flow2',
        },
    },
    'SD': {
        'CAFFEMODEL': '../models/FlowNet2-SD/FlowNet2-SD_weights.caffemodel.h5',
        'DEPLOY_PROTOTXT': '../models/FlowNet2-SD/FlowNet2-SD_deploy.prototxt.template',
        # Mappings between Caffe parameter names and TensorFlow variable names
        'PARAMS': {
            'conv0': 'FlowNetSD/conv0',
            'conv1': 'FlowNetSD/conv1',
            'conv1_1': 'FlowNetSD/conv1_1',
            'conv2': 'FlowNetSD/conv2',
            'conv2_1': 'FlowNetSD/conv2_1',
            'conv3': 'FlowNetSD/conv3',
            'conv3_1': 'FlowNetSD/conv3_1',
            'conv4': 'FlowNetSD/conv4',
            'conv4_1': 'FlowNetSD/conv4_1',
            'conv5': 'FlowNetSD/conv5',
            'conv5_1': 'FlowNetSD/conv5_1',
            'conv6': 'FlowNetSD/conv6',
            'conv6_1': 'FlowNetSD/conv6_1',
            'Convolution1': 'FlowNetSD/predict_flow6',
            'deconv5': 'FlowNetSD/deconv5',
            'upsample_flow6to5': 'FlowNetSD/upsample_flow6to5',
            'interconv5': 'FlowNetSD/interconv5',
            'Convolution2': 'FlowNetSD/predict_flow5',
            'deconv4': 'FlowNetSD/deconv4',
            'upsample_flow5to4': 'FlowNetSD/upsample_flow5to4',
            'interconv4': 'FlowNetSD/interconv4',
            'Convolution3': 'FlowNetSD/predict_flow4',
            'deconv3': 'FlowNetSD/deconv3',
            'upsample_flow4to3': 'FlowNetSD/upsample_flow4to3',
            'interconv3': 'FlowNetSD/interconv3',
            'Convolution4': 'FlowNetSD/predict_flow3',
            'deconv2': 'FlowNetSD/deconv2',
            'upsample_flow3to2': 'FlowNetSD/upsample_flow3to2',
            'interconv2': 'FlowNetSD/interconv2',
            'Convolution5': 'FlowNetSD/predict_flow2',
        },
    },
    '2': {
        'CAFFEMODEL': '../models/FlowNet2/FlowNet2_weights.caffemodel.h5',
        'DEPLOY_PROTOTXT': '../models/FlowNet2/FlowNet2_deploy.prototxt.template',
        # Mappings between Caffe parameter names and TensorFlow variable names
        'PARAMS': {
            # Net C
            'conv1': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetC/conv1',
            'conv2': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetC/conv2',
            'conv3': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetC/conv3',
            'conv_redir': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetC/conv_redir',
            'conv3_1': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetC/conv3_1',
            'conv4': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetC/conv4',
            'conv4_1': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetC/conv4_1',
            'conv5': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetC/conv5',
            'conv5_1': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetC/conv5_1',
            'conv6': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetC/conv6',
            'conv6_1': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetC/conv6_1',
            'Convolution1': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetC/predict_flow6',
            'deconv5': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetC/deconv5',
            'upsample_flow6to5': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetC/upsample_flow6to5',
            'Convolution2': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetC/predict_flow5',
            'deconv4': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetC/deconv4',
            'upsample_flow5to4': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetC/upsample_flow5to4',
            'Convolution3': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetC/predict_flow4',
            'deconv3': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetC/deconv3',
            'upsample_flow4to3': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetC/upsample_flow4to3',
            'Convolution4': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetC/predict_flow3',
            'deconv2': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetC/deconv2',
            'upsample_flow3to2': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetC/upsample_flow3to2',
            'Convolution5': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetC/predict_flow2',

            # Net S 1
            'net2_conv1': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetS/conv1',
            'net2_conv2': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetS/conv2',
            'net2_conv3': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetS/conv3',
            'net2_conv3_1': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetS/conv3_1',
            'net2_conv4': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetS/conv4',
            'net2_conv4_1': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetS/conv4_1',
            'net2_conv5': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetS/conv5',
            'net2_conv5_1': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetS/conv5_1',
            'net2_conv6': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetS/conv6',
            'net2_conv6_1': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetS/conv6_1',
            'net2_predict_conv6': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetS/predict_flow6',
            'net2_deconv5': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetS/deconv5',
            'net2_net2_upsample_flow6to5': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetS/upsample_flow6to5',
            'net2_predict_conv5': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetS/predict_flow5',
            'net2_deconv4': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetS/deconv4',
            'net2_net2_upsample_flow5to4': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetS/upsample_flow5to4',
            'net2_predict_conv4': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetS/predict_flow4',
            'net2_deconv3': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetS/deconv3',
            'net2_net2_upsample_flow4to3': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetS/upsample_flow4to3',
            'net2_predict_conv3': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetS/predict_flow3',
            'net2_deconv2': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetS/deconv2',
            'net2_net2_upsample_flow3to2': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetS/upsample_flow3to2',
            'net2_predict_conv2': 'FlowNet2/FlowNetCSS/FlowNetCS/FlowNetS/predict_flow2',

            # Net S 2
            'net3_conv1': 'FlowNet2/FlowNetCSS/FlowNetS/conv1',
            'net3_conv2': 'FlowNet2/FlowNetCSS/FlowNetS/conv2',
            'net3_conv3': 'FlowNet2/FlowNetCSS/FlowNetS/conv3',
            'net3_conv3_1': 'FlowNet2/FlowNetCSS/FlowNetS/conv3_1',
            'net3_conv4': 'FlowNet2/FlowNetCSS/FlowNetS/conv4',
            'net3_conv4_1': 'FlowNet2/FlowNetCSS/FlowNetS/conv4_1',
            'net3_conv5': 'FlowNet2/FlowNetCSS/FlowNetS/conv5',
            'net3_conv5_1': 'FlowNet2/FlowNetCSS/FlowNetS/conv5_1',
            'net3_conv6': 'FlowNet2/FlowNetCSS/FlowNetS/conv6',
            'net3_conv6_1': 'FlowNet2/FlowNetCSS/FlowNetS/conv6_1',
            'net3_predict_conv6': 'FlowNet2/FlowNetCSS/FlowNetS/predict_flow6',
            'net3_deconv5': 'FlowNet2/FlowNetCSS/FlowNetS/deconv5',
            'net3_net3_upsample_flow6to5': 'FlowNet2/FlowNetCSS/FlowNetS/upsample_flow6to5',
            'net3_predict_conv5': 'FlowNet2/FlowNetCSS/FlowNetS/predict_flow5',
            'net3_deconv4': 'FlowNet2/FlowNetCSS/FlowNetS/deconv4',
            'net3_net3_upsample_flow5to4': 'FlowNet2/FlowNetCSS/FlowNetS/upsample_flow5to4',
            'net3_predict_conv4': 'FlowNet2/FlowNetCSS/FlowNetS/predict_flow4',
            'net3_deconv3': 'FlowNet2/FlowNetCSS/FlowNetS/deconv3',
            'net3_net3_upsample_flow4to3': 'FlowNet2/FlowNetCSS/FlowNetS/upsample_flow4to3',
            'net3_predict_conv3': 'FlowNet2/FlowNetCSS/FlowNetS/predict_flow3',
            'net3_deconv2': 'FlowNet2/FlowNetCSS/FlowNetS/deconv2',
            'net3_net3_upsample_flow3to2': 'FlowNet2/FlowNetCSS/FlowNetS/upsample_flow3to2',
            'net3_predict_conv2': 'FlowNet2/FlowNetCSS/FlowNetS/predict_flow2',

            # Net SD
            'netsd_conv0': 'FlowNet2/FlowNetSD/conv0',
            'netsd_conv1': 'FlowNet2/FlowNetSD/conv1',
            'netsd_conv1_1': 'FlowNet2/FlowNetSD/conv1_1',
            'netsd_conv2': 'FlowNet2/FlowNetSD/conv2',
            'netsd_conv2_1': 'FlowNet2/FlowNetSD/conv2_1',
            'netsd_conv3': 'FlowNet2/FlowNetSD/conv3',
            'netsd_conv3_1': 'FlowNet2/FlowNetSD/conv3_1',
            'netsd_conv4': 'FlowNet2/FlowNetSD/conv4',
            'netsd_conv4_1': 'FlowNet2/FlowNetSD/conv4_1',
            'netsd_conv5': 'FlowNet2/FlowNetSD/conv5',
            'netsd_conv5_1': 'FlowNet2/FlowNetSD/conv5_1',
            'netsd_conv6': 'FlowNet2/FlowNetSD/conv6',
            'netsd_conv6_1': 'FlowNet2/FlowNetSD/conv6_1',
            'netsd_Convolution1': 'FlowNet2/FlowNetSD/predict_flow6',
            'netsd_deconv5': 'FlowNet2/FlowNetSD/deconv5',
            'netsd_upsample_flow6to5': 'FlowNet2/FlowNetSD/upsample_flow6to5',
            'netsd_interconv5': 'FlowNet2/FlowNetSD/interconv5',
            'netsd_Convolution2': 'FlowNet2/FlowNetSD/predict_flow5',
            'netsd_deconv4': 'FlowNet2/FlowNetSD/deconv4',
            'netsd_upsample_flow5to4': 'FlowNet2/FlowNetSD/upsample_flow5to4',
            'netsd_interconv4': 'FlowNet2/FlowNetSD/interconv4',
            'netsd_Convolution3': 'FlowNet2/FlowNetSD/predict_flow4',
            'netsd_deconv3': 'FlowNet2/FlowNetSD/deconv3',
            'netsd_upsample_flow4to3': 'FlowNet2/FlowNetSD/upsample_flow4to3',
            'netsd_interconv3': 'FlowNet2/FlowNetSD/interconv3',
            'netsd_Convolution4': 'FlowNet2/FlowNetSD/predict_flow3',
            'netsd_deconv2': 'FlowNet2/FlowNetSD/deconv2',
            'netsd_upsample_flow3to2': 'FlowNet2/FlowNetSD/upsample_flow3to2',
            'netsd_interconv2': 'FlowNet2/FlowNetSD/interconv2',
            'netsd_Convolution5': 'FlowNet2/FlowNetSD/predict_flow2',

            # Fusion Net
            'fuse_conv0': 'FlowNet2/fuse_conv0',
            'fuse_conv1': 'FlowNet2/fuse_conv1',
            'fuse_conv1_1': 'FlowNet2/fuse_conv1_1',
            'fuse_conv2': 'FlowNet2/fuse_conv2',
            'fuse_conv2_1': 'FlowNet2/fuse_conv2_1',
            'fuse__Convolution5': 'FlowNet2/predict_flow2',
            'fuse_deconv1': 'FlowNet2/fuse_deconv1',
            'fuse_upsample_flow2to1': 'FlowNet2/fuse_upsample_flow2to1',
            'fuse_interconv1': 'FlowNet2/fuse_interconv1',
            'fuse__Convolution6': 'FlowNet2/predict_flow1',
            'fuse_deconv0': 'FlowNet2/fuse_deconv0',
            'fuse_upsample_flow1to0': 'FlowNet2/fuse_upsample_flow1to0',
            'fuse_interconv0': 'FlowNet2/fuse_interconv0',
            'fuse__Convolution7': 'FlowNet2/predict_flow0',
        }
    },
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
        out[tf_param + '/weights'] = net.params[caffe_param][0].data.transpose((2, 3, 1, 0))
        out[tf_param + '/biases'] = net.params[caffe_param][1].data

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
        choices=['C', 'S', 'CS', 'CSS', 'CSS-ft-sd', 'SD', '2'],
        required=True,
        help='Name of the FlowNet arch: C, S, CS, CSS, CSS-ft-sd, SD or 2'
    )
    FLAGS = parser.parse_args()
    arch = ARCHS[FLAGS.arch]
    main()
