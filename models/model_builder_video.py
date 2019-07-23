import numpy as np
from caffe2.proto import caffe2_pb2
from caffe2.python import workspace, scope, core, cnn, data_parallel_model
from models import resnet_video_org

# All parameters from the config file have been substituted in
# This is a model builder for a hard-coded I3D-ResNet50
# All caffe2 .py files have been adapted from https://github.com/facebookresearch/video-nonlocal-net/

class ModelBuilder(cnn.CNNModelHelper):

    def __init__(self, **kwargs):
        kwargs['order'] = 'NCHW'
        self.train = kwargs.get('train', False)
        self.split = kwargs.get('split', 'train')
        self.use_mem_cache = kwargs.get('use_mem_cache', False)
        self.force_fw_only = kwargs.get('force_fw_only', False)

        if 'train' in kwargs:
            del kwargs['train']
        if 'split' in kwargs:
            del kwargs['split']
        if 'use_mem_cache' in kwargs:
            del kwargs['use_mem_cache']
        if 'force_fw_only' in kwargs:
            del kwargs['force_fw_only']

        super(ModelBuilder, self).__init__(**kwargs)

    # ----------------------------
    # customized layers
    # ----------------------------
    # relu with inplace option
    def Relu_(self, blob_in):
        blob_out = self.Relu(
            blob_in,
            blob_in)
        return blob_out

    def Conv3dBN(
        self, blob_in, prefix, dim_in, dim_out, kernels, strides, pads,
        group=1, bn_init=None,
        **kwargs
    ):
        conv_blob = self.ConvNd(
            blob_in, prefix, dim_in, dim_out, kernels, strides=strides,
            pads=pads, group=group,
            weight_init=("MSRAFill", {}),
            bias_init=('ConstantFill', {'value': 0.}), no_bias=1)
        blob_out = self.SpatialBN(
            conv_blob, prefix + "_bn", dim_out,
            epsilon=1.0000001e-05,
            momentum=0.9,
            is_test=self.split in ['test', 'val'])

        # set bn init if specified
        if bn_init is not None and bn_init != 1.0:  # numerical issue not matter
            self.param_init_net.ConstantFill(
                [prefix + "_bn_s"],
                prefix + "_bn_s", value=bn_init)

        return blob_out
