def convolve(self, a, b, padding, back_conv = False):
    """Convolution between matrices `a` and `b`.

    Args:
        a: Stationary matrix.
        b: Moving matrix.
        padding: Two numbers representing padding.
        back_conv (bool): Whether the convolution is during a backward_pass or not. Defaults to False.

    Returns:
        array: Matrix containing convoluted output.
    """

    start_time = datetime.now()
    a_pad = np.zeros([a.shape[0], a.shape[1] + padding[0] * 2, a.shape[2] + padding[1] * 2])
    a_pad[:,padding[0]:padding[0] + a.shape[1], padding[1]:padding[1] + a.shape[2]] = a
    
    i0 = np.repeat(np.arange(b.shape[2]), b.shape[2])
    i1 = np.repeat(np.arange(a.shape[2]), a.shape[2])
    i = i0.reshape(-1,1) + i1.reshape(1, -1)

    j0 = np.tile(np.arange(b.shape[3]), b.shape[3])
    j1 = np.tile(np.arange(a.shape[2]), a.shape[2])
    j = j0.reshape(-1,1) + j1.reshape(1, -1)

    # k = np.repeat(np.arange(b.shape[0]), np.prod(b.shape[2:])).reshape(-1,1)

    select_img = a_pad[:, i, j]
    weights = b.reshape(b.shape[0], b.shape[1], b.shape[2] * b.shape[3])        
    dot_product = np.tensordot(weights[:], select_img).reshape(b.shape[0], 28, 28)        
    print(f"time {datetime.now() - start_time}")
    print(dot_product.shape)
    return dot_product