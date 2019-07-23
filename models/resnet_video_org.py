import models.resnet_helper as resnet_helper
import numpy as np


def create_model(model, data, labels, split, use_nl):
    group = 1
    width_per_group = 64
    batch_size = 8

    (n1, n2, n3, n4) = (3, 4, 6, 3)

    res_block = resnet_helper._generic_residual_block_3d
    dim_inner = group * width_per_group

    use_temp_convs_set = [[2], [1, 1, 1], [1, 0, 1, 0], [1, 0, 1, 0, 1, 0], [0, 1, 0]]
    temp_strides_set = [[2], [1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1]]
    pool_stride = 4

    conv_blob = model.ConvNd(
        data, 'conv1', 3, 64, [5, 7, 7], strides=[2, 2, 2],
        pads=[2, 3, 3] * 2,
        weight_init=('MSRAFill', {}),
        bias_init=('ConstantFill', {'value': 0.}), no_bias=1
    )

    test_mode = True
    bn_blob = model.SpatialBN(
        conv_blob, 'res_conv1_bn', 64, epsilon=1.0000001e-05,
        momentum=0.9, is_test=test_mode,
    )
    relu_blob = model.Relu(bn_blob, bn_blob)
    max_pool = model.MaxPool(relu_blob, 'pool1', kernels=[2, 3, 3], strides=[2, 2, 2], pads=[0, 0, 0] * 2)

    # ---------------- #

    nonlocal_mod = 2 if use_nl else 1000

    blob_in, dim_in = resnet_helper.res_stage_nonlocal(
        model, res_block, max_pool, 64, 256, stride=1, num_blocks=3,
        prefix='res2', dim_inner=64, group=1,
        use_temp_convs=[1, 1, 1], temp_strides=[1, 1, 1])

    blob_in = model.MaxPool(blob_in, 'pool2', kernels=[2, 1, 1], strides=[2, 1, 1], pads=[0, 0, 0] * 2)

    blob_in, dim_in = resnet_helper.res_stage_nonlocal(
        model, res_block, blob_in, dim_in, 512, stride=2, num_blocks=4,
        prefix='res3', dim_inner=64 * 2, group=1,
        use_temp_convs=[1, 0, 1, 0], temp_strides=[1, 1, 1, 1],
        batch_size=batch_size, nonlocal_name='nonlocal_conv3', nonlocal_mod=nonlocal_mod)

    blob_in, dim_in = resnet_helper.res_stage_nonlocal(
        model, res_block, blob_in, dim_in, 1024, stride=2, num_blocks=6,
        prefix='res4', dim_inner=64 * 4, group=1,
        use_temp_convs=[1, 0, 1, 0, 1, 0], temp_strides=[1, 1, 1, 1, 1, 1],
        batch_size=batch_size, nonlocal_name='nonlocal_conv4', nonlocal_mod=nonlocal_mod)

    blob_in, dim_in = resnet_helper.res_stage_nonlocal(
        model, res_block, blob_in, dim_in, 2048, stride=2, num_blocks=3,
        prefix='res5', dim_inner=dim_inner * 8, group=1,
        use_temp_convs=[0, 1, 0], temp_strides=[1, 1, 1])

    blob_out = model.AveragePool(blob_in, 'pool5', kernels=[4, 7, 7], strides=[1, 1, 1], pads=[0, 0, 0] * 2)

    # # NO DROPOUT LAYER IN TEST MODE?
    # if cfg.TRAIN.DROPOUT_RATE > 0 and test_mode is False:
    #     blob_out = model.Dropout(
    #         blob_out, blob_out + '_dropout', ratio=cfg.TRAIN.DROPOUT_RATE, is_test=False)

    if split in ['train', 'val']:
        blob_out = model.FC(
            blob_out, 'pred', dim_in, 400,
            weight_init=('GaussianFill', {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.})
        )
    elif split == 'test':
        blob_out = model.ConvNd(
            blob_out, 'pred', dim_in, 400,
            [1, 1, 1], strides=[1, 1, 1], pads=[0, 0, 0] * 2,
        )

    return model, blob_out
