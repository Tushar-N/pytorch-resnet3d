
# 3d spacetime nonlocal (v1: spatial downsample)
def spacetime_nonlocal(
        model, blob_in, dim_in, dim_out, batch_size, prefix, dim_inner,
        is_test, max_pool_stride=2):
    # ---------------------
    cur = blob_in
    # we do projection to convert each spacetime location to a feature
    # theta original size
    # e.g., (8, 1024, 4, 14, 14) => (8, 1024, 4, 14, 14)

    theta = model.ConvNd(
        cur, prefix + '_theta',
        dim_in,
        dim_inner,
        [1, 1, 1],
        strides=[1, 1, 1],
        pads=[0, 0, 0] * 2,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.}), no_bias=0)

    # phi and g: half spatial size
    # e.g., (8, 1024, 4, 14, 14) => (8, 1024, 4, 7, 7)
    max_pool = model.MaxPool(
        cur, prefix + '_pool',
        kernels=[1, max_pool_stride, max_pool_stride],
        strides=[1, max_pool_stride, max_pool_stride],
        pads=[0, 0, 0] * 2,
    )
   

    phi = model.ConvNd(
        max_pool, prefix + '_phi',
        dim_in,
        dim_inner,
        [1, 1, 1],
        strides=[1, 1, 1],
        pads=[0, 0, 0] * 2,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.}), no_bias=0)

    g = model.ConvNd(
        max_pool, prefix + '_g',
        dim_in,
        dim_inner,
        [1, 1, 1],
        strides=[1, 1, 1],
        pads=[0, 0, 0] * 2,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.}), no_bias=0)

    # we have to use explicit batch size (to support arbitrary spacetime size)
    # e.g., (8, 1024, 4, 14, 14) => (8, 1024, 784)
    theta, theta_shape_5d = model.Reshape(
        theta, [theta,
            theta + '_shape5d'],
        shape=(batch_size, dim_inner, -1))
    phi, phi_shape_5d = model.Reshape(
        phi, [phi,
            phi + '_shape5d'],
        shape=(batch_size, dim_inner, -1))
    g, g_shape_5d = model.Reshape(
        g, [g,
            g + '_shape5d'],
        shape=(batch_size, dim_inner, -1))

    # e.g., (8, 1024, 784) * (8, 1024, 784) => (8, 784, 784)
    theta_phi = model.net.BatchMatMul([theta, phi], prefix + '_affinity', trans_a=1)
    theta_phi_sc = model.Scale(theta_phi, theta_phi, scale=dim_inner**-.5)
    # softmax
    # sum(p[i, j, :]) == 1, for any i, j
    p = model.Softmax(theta_phi_sc, theta_phi + '_prob', engine='CUDNN', axis=2)


    # note: g's axis[2] corresponds to p's axis[2]
    # e.g., g(8, 1024, 784_2) * p(8, 784_1, 784_2) => (8, 1024, 784_1)
    t = model.net.BatchMatMul([g, p], prefix + '_y', trans_b=1)

    # reshape back:
    # e.g., (8, 1024, 784) => (8, 1024, 4, 14, 14)
    t_re, t_shape = model.Reshape(
        [t, theta_shape_5d],
        [t,
            t + '_shape3d'])
    blob_out = t_re

    blob_out = model.ConvNd(
        blob_out, prefix + '_out',
        dim_inner,
        dim_out,
        [1, 1, 1],
        strides=[1, 1, 1],
        pads=[0, 0, 0] * 2,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.}), no_bias=0)


    blob_out = model.SpatialBN(
        blob_out, prefix + "_bn", dim_out,
        epsilon=1.0000001e-05, momentum=0.9,
        is_test=is_test
    )
    model.param_init_net.ConstantFill(
        [prefix + "_bn_s"], prefix + "_bn_s", value=0.0)

    return blob_out


def add_nonlocal(model, blob_in, dim_in, dim_out, batch_size, prefix, dim_inner):
    is_test = model.split in ['test', 'val']
    blob_out = spacetime_nonlocal(
        model, blob_in, dim_in, dim_out, batch_size, prefix, dim_inner, is_test)
    blob_out = model.net.Sum([blob_in, blob_out], prefix + "_sum")
    return blob_out