from .AugmentationCoeff import (
    AugmentationCoeff, DEFAULT_CHROMATIC_TYPES,
    DEFAULT_CHROMATIC_EIGEN_TYPES, DEFAULT_EFFECT_TYPES
)
import tensorflow as tf
from tensorflow.contrib.image import angles_to_projective_transforms, compose_transforms


# TODO: def compute_chromatic_eigenspace(images, params):
#     chromatic_eigen_space = {
#         mean_eig: tf.Variable(np.zeros(3, 1)),
#         mean_rgb: tf.Variable(np.zeros(3, 1)),
#         max_abs_eig: tf.Variable(np.zeros(3, 1)),
#         max_rgb: tf.Variable(np.zeros(3, 1)),
#         min_rgb: tf.Variable(np.zeros(3, 1)),
#         max_l: tf.Variable(0.0),
#         eigvec: tf.Variable(np.array(params.chromatic_eigvec).reshape(3, 3)),
#     }
#
#     # Calculate mean/max/min per channel with dimensions (3, 1)
#     tf.assign(chromatic_eigen_space.mean_rgb,
#         tf.expand_dims(tf.reduce_mean(images, [0, 1, 2]), 1))
#     tf.assign(chromatic_eigen_space.max_rgb,
#         tf.expand_dims(tf.reduce_max(images, [0, 1, 2]), 1))
#     tf.assign(chromatic_eigen_space.min_rgb,
#         tf.expand_dims(tf.reduce_min(images, [0, 1, 2]), 1))
#
#     # Calculate maximum absolute eigen per channel with dimensions (1, 3)
#     # TODO: what is this doing exactly? is this computing eigenvalues?
#     # TODO: there's gotta be a way to speed this up...
#     eigvec0 = tf.constant(chromatic_eigen_space.eigvec[0, :])
#     eigvec1 = tf.constant(chromatic_eigen_space.eigvec[1, :])
#     eigvec2 = tf.constant(chromatic_eigen_space.eigvec[2, :])
#     max_abs_eig0 = tf.reduce_max(
#         tf.abs(tf.reduce_sum(tf.multiply(images, tf.reshape(eigvec0, [1, 1, 3])), 3)))
#     max_abs_eig1 = tf.reduce_max(
#         tf.abs(tf.reduce_sum(tf.multiply(images, tf.reshape(eigvec1, [1, 1, 3])), 3)))
#     max_abs_eig2 = tf.reduce_max(
#         tf.abs(tf.reduce_sum(tf.multiply(images, tf.reshape(eigvec2, [1, 1, 3])), 3)))
#     tf.assign(chromatic_eigen_space.max_abs_eig,
#         tf.constant([[max_abs_eig0, max_abs_eig1, max_abs_eig2]]))
#
#     # Calculate the mean eig
#     mean_eig = tf.matmul(chromatic_eigen_space.eigvec, chromatic_eigen_space.mean_rgb)
#     tf.assign(chromatic_eigen_space.mean_eig,
#         tf.where(chromatic_eigen_space.max_abs_eig > 0.01,
#             mean_eig / chromatic_eigen_space.max_abs_eig,
#             mean_eig
#     ))
#
#     # Assign max_l to be the magnitude of the max_abs_eig vector
#     tf.assign(chromatic_eigen_space.max_l,
#         tf.sqrt(tf.reduce_sum(tf.square(chromatic_eigen_space.max_abs_eig))))
#
#     return chromatic_eigen_space


def translations_to_projective_transforms(translations):
    """Returns projective transform(s) for the given translation(s).

    Args:
        translations: A 2-element list representing [dx, dy] or a vector of 2-element lists
            representing [dx, dy] to translate for each image (for a batch of images)

    Returns:
        A tensor of shape (num_images, 8) projective transforms which can be given
            to `tf.contrib.image.transform`.
    """
    translation_or_translations = tf.convert_to_tensor(
        translations, name="translations", dtype=tf.float32)
    if len(translation_or_translations.get_shape()) == 1:
        translations = translation_or_translations[None]
    elif len(translation_or_translations.get_shape()) == 2:
        translations = translation_or_translations
    else:
        raise TypeError("Translations should have rank 1 or 2.")
    num_translations = tf.shape(translations)[0]
    return tf.concat(
        values=[
            tf.ones((num_translations, 1), tf.float32),
            tf.zeros((num_translations, 1), tf.float32),
            -translations[:, 0, None],
            tf.zeros((num_translations, 1), tf.float32),
            tf.ones((num_translations, 1), tf.float32),
            -translations[:, 1, None],
            tf.zeros((num_translations, 2), tf.float32),
        ],
        axis=1)


def scales_to_projective_transforms(scales):
    """Returns projective transform(s) for the given scale(s).

    Args:
        scales: A 2-element list representing [dx, dy] or a vector of 2-element lists
            representing [dx, dy] to scale for each image (for a batch of images)

    Returns:
        A tensor of shape (num_images, 8) projective transforms which can be given
            to `tf.contrib.image.transform`.
    """
    scale_or_scales = tf.convert_to_tensor(
        scales, name="scales", dtype=tf.float32)
    if len(scale_or_scales.get_shape()) == 1:
        scales = scale_or_scales[None]
    elif len(scale_or_scales.get_shape()) == 2:
        scales = scale_or_scales
    else:
        raise TypeError("Scales should have rank 1 or 2.")
    num_scales = tf.shape(scales)[0]
    return tf.concat(
        values=[
            scales[:, 0, None],
            tf.zeros((num_scales, 3), tf.float32),
            scales[:, 1, None],
            tf.zeros((num_scales, 3), tf.float32),
        ],
        axis=1)


def compute_spatial_transformation_matrix(coeffs, samples):
    _, height, width, _ = samples.shape.as_list()
    transforms = []

    # Shift origin relative to crop
    origin_shifts = tf.convert_to_tensor(
        [[0.5 * coeff.crop_width, 0.5 * coeff.crop_height] for coeff in coeffs],
        dtype=tf.float32)
    transforms.append(translations_to_projective_transforms(origin_shifts))

    angles = tf.convert_to_tensor([coeff.rotate for coeff in coeffs], dtype=tf.float32)
    transforms.append(angles_to_projective_transforms(angles, height, width))

    translations = tf.convert_to_tensor(
        [[coeff.translate_x * coeff.crop_width, coeff.translate_y * coeff.crop_height]
            for coeff
            in coeffs],
        tf.float32)
    transforms.append(translations_to_projective_transforms(translations))

    scales = tf.convert_to_tensor(
        [[1.0 / coeff.zoom_x, 1.0 / coeff.zoom_y] for coeff in coeffs],
        tf.float32)
    transforms.append(scales_to_projective_transforms(scales))

    # Shift origin back relative to entire image
    origin_shifts_2 = tf.convert_to_tensor(
        [[-0.5 * width, -0.5 * height] for coeff in coeffs],
        dtype=tf.float32)
    transforms.append(translations_to_projective_transforms(origin_shifts_2))

    # Important!! We need to reverse these transforms so they are applied in order.
    # At this point transforms is [a, b, c, ..., z] which will do a * b * c * ... * z * coord_sys
    # This is wrong, we want to apply transformation 'a' first, not 'z'
    transforms.reverse()

    return compose_transforms(*transforms)


def color_contrast_augmentation(coeffs, samples):
    with tf.name_scope('color_augmentation'):
        mean_in = tf.reduce_sum(samples, 3, keep_dims=True)  # (B, H, W, 1)

        color_mult = [tf.stack([coeff.color1, coeff.color2, coeff.color3]) for coeff in coeffs]
        samples = samples * tf.reshape(color_mult, [len(coeffs), 1, 1, 3])

        mean_out = tf.reduce_sum(samples, 3, keep_dims=True)  # (B, H, W, 1)

        brightness_coeff = mean_in / (mean_out + 0.01)

        # Compensate brightness
        samples = tf.clip_by_value(samples * brightness_coeff, 0.0, 1.0)

        # Gamma
        samples = tf.pow(samples, [[[[coeff.gamma]]] for coeff in coeffs])

        # Brightness
        samples = samples + [[[[coeff.brightness]]] for coeff in coeffs]

        # Contrast
        contrast = [[[[coeff.contrast]]] for coeff in coeffs]
        samples = 0.5 + (samples - 0.5) * contrast

        return tf.clip_by_value(samples, 0.0, 1.0)


def effects_augmentation(coeffs, samples):
    with tf.name_scope('effects_augmentation'):
        num, height, width, channels = samples.shape.as_list()
        noises = [coeff.noise for coeff in coeffs]
        noises = [tf.random_normal([1, height, width, channels], 0.0, noise, dtype=tf.float32)
                  for noise in noises]
        noises = tf.concat(noises, 0)
        return tf.clip_by_value(samples + noises, 0.0, 1.0)


def inverse_transformation_matrix(matrix):
    """
    matrix is a (N, 8) matrix where N is number of samples to transform.

    Note this ignores matrix[:, 7] and matrix[:, 8].
    """
    with tf.name_scope('inverse_transformation_matrix'):
        N, _ = matrix.shape.as_list()
        a = tf.reshape(matrix[:, 0], [N, 1])
        b = tf.reshape(matrix[:, 3], [N, 1])
        c = tf.reshape(matrix[:, 1], [N, 1])
        d = tf.reshape(matrix[:, 4], [N, 1])
        e = tf.reshape(matrix[:, 2], [N, 1])
        f = tf.reshape(matrix[:, 5], [N, 1])
        g = tf.reshape(matrix[:, 6], [N, 1])
        h = tf.reshape(matrix[:, 7], [N, 1])
        denom = a*d - b*c

        return tf.concat([d / denom,
                          -c / denom,
                          (c*f-d*e) / denom,
                          -b / denom,
                          a / denom,
                          (b*e-a*f)/denom,
                          g,
                          h], 1)


def preprocess(images, params, global_step, batch_size, old_coeffs=None):
    with tf.name_scope('preprocess'):
        _, height, width, _ = images.shape.as_list()

        if ('crop_width' not in params or 'crop_height' not in params) and old_coeffs is None:
            raise ValueError('A crop_width and crop_height must be set in augmentation parameters.')

        # Scale from [0, 255] -> [0.0, 1.0]
        if 'scale' in params and params['scale']:
            images = images * 0.00392156862745

        # TODO: Maybe we should generate all coefficients at once...
        # ...in a matrix instead of individual values

        # Generate random coeffs, one per sample in batch
        if old_coeffs:
            # The newly generated coeffs will be combined with the old coeffs
            coeffs = [AugmentationCoeff(params, width, height, global_step, old_coeffs[i])
                      for i in range(batch_size)]
        else:
            coeffs = [AugmentationCoeff(params, width, height, global_step)
                      for i in range(batch_size)]

        # TODO: Setup for chromatic-eigen augmentation
        # has_chromatic_eigen_augmentation = not all([coeff.is_default_value('chromatic-eigen') for coeff in coeffs])
        # if has_chromatic_eigen_augmentation:
        #     chromatic_eigen_space = compute_chromatic_eigenspace(images, params)

        spatial_transformation_matrix = compute_spatial_transformation_matrix(coeffs, images)
        crop = [int(params['crop_height']), int(params['crop_width'])]
        # TODO: Apply transformation matrix!

        # TODO:
        # if has_chromatic_eigen_augmentation:
        #   ChromaticEigenAugmentation:
        #       image - eigen->mean_rgb{0,1,2} for each channel

        if len(set(DEFAULT_CHROMATIC_TYPES.keys()).intersection(params.keys())) != 0:
            images = color_contrast_augmentation(coeffs, images)

        if len(set(DEFAULT_EFFECT_TYPES.keys()).intersection(params.keys())) != 0:
            images = effects_augmentation(coeffs, images)

        return images, coeffs
