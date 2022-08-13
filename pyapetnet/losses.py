import tensorflow as tf
from tensorflow import keras


def tf_gauss_kernel_3d(sigma, size):
    """Generate 3D Gaussian kernel 
  
       Parameters
       ----------

       sigma : float
         width of the gaussian
       size : int 
         size of the gaussian (should be odd an approx 2*int(3.5*sigma + 0.5) + 1

       Returns
       -------
       tensorflow tensor with dimension [size,size,size,1,1] with tf.reduce_sum(k) = 1
    """
    size = tf.convert_to_tensor(size, tf.int32)
    sigma = tf.convert_to_tensor(sigma, tf.float32)

    coords = tf.cast(tf.range(size),
                     tf.float32) - tf.cast(size - 1, tf.float32) / 2.0

    g = -0.5 * tf.square(coords) / tf.square(sigma)
    g = tf.nn.softmax(g)

    g = tf.einsum('i,j,k->ijk', g, g, g)
    g = tf.expand_dims(tf.expand_dims(g, -1), -1)

    return g


def ssim_3d(x,
            y,
            sigma=1.5,
            size=11,
            L=4,
            K1=0.01,
            K2=0.03,
            return_image=False):
    """ Compute the structural similarity between two batches of 3D single channel images

        Parameters
        ----------

        x,y : tensorflow tensors with shape [batch_size,depth,height,width,1] 
          containing a batch of 3D images with 1 channel
        L : float
          dynamic range of the images. 
          default is 4 (assuming normalized inputs
          If None it is set to tf.reduce_max(x) - tf.reduce_min(x)
        K1, K2 : float
          small constants needed to avoid division by 0 see [1]. 
          Default 0.01, 0.03
        sigma : float 
          width of the gaussian filter in pixels
          Default 1.5
        size : int
          size of the gaussian kernel used to calculate local means and std.devs 
          Default 11

        Returns
        -------
        a 1D tensorflow tensor of length batch_size containing the SSIM for
        every image pair in the batch

        Note
        ----
        (1) This implementation is very close to [1] and 
            from skimage.metrics import structural_similarity
            structural_similarity(x, y, gaussian_weights = True, full = True, data_range = L)
        (2) The default way of how the dynamic range L is calculated (based on y)
            is different from [1] and structural_similarity()

        References
        ----------
        [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
        (2004). Image quality assessment: From error visibility to
        structural similarity. IEEE Transactions on Image Processing
    """

    # we have to test whether the last dimension is 1 or None
    # when compiling the model it can be None and we should not throw an error
    if ((x.shape[-1] is not None) and
        (x.shape[-1] != 1)) or ((y.shape[-1] is not None)
                                and y.shape[-1] != 1):
        raise ValueError('Last dimension of input x has to be 1')

    if L is None:
        L = tf.reduce_max(x) - tf.reduce_min(x)

    C1 = (K1 * L)**2
    C2 = (K2 * L)**2

    shape = x.shape
    kernel = tf_gauss_kernel_3d(sigma, size)

    mu_x = tf.nn.conv3d(x, kernel, strides=[1, 1, 1, 1, 1], padding='VALID')
    mu_y = tf.nn.conv3d(y, kernel, strides=[1, 1, 1, 1, 1], padding='VALID')

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_x_y = mu_x * mu_y

    sig_x_sq = tf.nn.conv3d(
        x * x, kernel, strides=[1, 1, 1, 1, 1], padding='VALID') - mu_x_sq
    sig_y_sq = tf.nn.conv3d(
        y * y, kernel, strides=[1, 1, 1, 1, 1], padding='VALID') - mu_y_sq
    sig_xy = tf.nn.conv3d(
        x * y, kernel, strides=[1, 1, 1, 1, 1], padding='VALID') - mu_x_y

    SSIM = (2 * mu_x_y + C1) * (2 * sig_xy + C2) / ((mu_x_sq + mu_y_sq + C1) *
                                                    (sig_x_sq + sig_y_sq + C2))

    if not return_image:
        SSIM = tf.reduce_mean(SSIM, [1, 2, 3, 4])

    return SSIM


def ssim_3d_loss(x, y, **kwargs):
    """ Compute the structural similarity loss between two batches of 3D single channel images

        Parameters
        ----------

        x,y : tensorflow tensors with shape [batch_size,depth,height,width,1] 
          containing a batch of 3D images with 1 channel
        **kwargs : dict
          passed to tf_ssim_3d

        Returns
        -------
        a 1D tensorflow tensor of length batch_size containing the 1 - SSIM for
        every image pair in the batch

        See also
        ----------
        tf_ssim_3d
    """
    return 1 - ssim_3d(x, y, **kwargs)


def mix_ssim_3d_mae_loss(x, y, alpha=0.5):
    return alpha * ssim_3d_loss(x, y) + (1 - alpha) * tf.reduce_mean(
        keras.losses.mean_absolute_error(x, y), axis=[1, 2, 3])
