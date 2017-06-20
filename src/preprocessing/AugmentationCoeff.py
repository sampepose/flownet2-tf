import numpy as np
import tensorflow as tf

TF_NAN = tf.constant(float('nan'))

DEFAULT_SPATIAL_TYPES = {
    'translate_x': tf.constant(0, dtype=tf.float32),
    'translate_y': tf.constant(0, dtype=tf.float32),
    'rotate': tf.constant(0, dtype=tf.float32),
    'zoom_x': tf.constant(1, dtype=tf.float32),
    'zoom_y': tf.constant(1, dtype=tf.float32),
}
DEFAULT_CHROMATIC_TYPES = {
    'gamma': tf.constant(1, dtype=tf.float32),
    'brightness': tf.constant(0, dtype=tf.float32),
    'contrast': tf.constant(1, dtype=tf.float32),
    'color1': tf.constant(1, dtype=tf.float32),
    'color2': tf.constant(1, dtype=tf.float32),
    'color3': tf.constant(1, dtype=tf.float32),
}
DEFAULT_CHROMATIC_EIGEN_TYPES = {
    'pow_nomean0': 1,
    'pow_nomean1': 1,
    'pow_nomean2': 1,
    'add_nomean0': 0,
    'add_nomean1': 0,
    'add_nomean2': 0,
    'mult_nomean0': 1,
    'mult_nomean1': 1,
    'mult_nomean2': 1,
    'pow_withmean0': 1,
    'pow_withmean1': 1,
    'pow_withmean2': 1,
    'add_withmean0': 0,
    'add_withmean1': 0,
    'add_withmean2': 0,
    'mult_withmean0': 1,
    'mult_withmean1': 1,
    'mult_withmean2': 1,
    'lmult_pow': 1,
    'lmult_add': 0,
    'lmult_mult': 1,
    'col_angle': 0,
}
DEFAULT_EFFECT_TYPES = {
    'noise': tf.constant(0, dtype=tf.float32),
}
ALL_DEFAULT_TYPES = {}
ALL_DEFAULT_TYPES.update(DEFAULT_SPATIAL_TYPES)
ALL_DEFAULT_TYPES.update(DEFAULT_CHROMATIC_TYPES)
# ALL_DEFAULT_TYPES.update(DEFAULT_CHROMATIC_EIGEN_TYPES)
ALL_DEFAULT_TYPES.update(DEFAULT_EFFECT_TYPES)


# https://github.com/tgebru/transform/blob/master/src/caffe/layers/data_augmentation_layer.cpp#L34
def _generate_coeff(param, discount_coeff=tf.constant(1), default_value=None):
    if not all(name in param for name in ['rand_type', 'exp', 'mean', 'spread', 'prob']):
        raise RuntimeError('Expected rand_type, exp, mean, spread, prob in `param`')

    rand_type = param['rand_type']
    exp = float(param['exp'])
    mean = tf.convert_to_tensor(param['mean'], dtype=tf.float32)
    spread = float(param['spread'])  # AKA standard deviation
    prob = float(param['prob'])

    # Multiply spread by our discount_coeff so it changes over time
    spread = spread * discount_coeff

    if rand_type == 'uniform':
        value = tf.cond(spread > 0.0,
                        lambda: tf.random_uniform([], mean - spread, mean + spread),
                        lambda: mean)
        if exp:
            value = tf.exp(value)
    elif rand_type == 'gaussian':
        value = tf.cond(spread > 0.0,
                        lambda: tf.random_normal([], mean, spread),
                        lambda: mean)
        if exp:
            value = tf.exp(value)
    elif rand_type == 'bernoulli':
        if prob > 0.0:
            value = tf.contrib.distributions.Bernoulli(probs=prob).sample([])
        else:
            value = 0.0
    elif rand_type == 'uniform_bernoulli':
        tmp1 = 0.0
        tmp2 = 0
        if prob > 0.0:
            tmp2 = tf.contrib.distributions.Bernoulli(probs=prob).sample([])
        else:
            tmp2 = 0

        if tmp2 == 0:
            if default_value is not None:
                return default_value
        else:
            tmp1 = tf.cond(spread > 0.0,
                           lambda: tf.random_uniform([], mean - spread, mean + spread),
                           lambda: mean)
        if exp:
            tmp1 = tf.exp(tmp1)
        value = tmp1
    elif rand_type == 'gaussian_bernoulli':
        tmp1 = 0.0
        tmp2 = 0
        if prob > 0.0:
            tmp2 = tf.contrib.distributions.Bernoulli(probs=prob).sample([])
        else:
            tmp2 = 0

        if tmp2 == 0:
            if default_value is not None:
                return default_value
        else:
            tmp1 = tf.cond(spread > 0.0,
                           lambda: tf.random_normal([], mean, spread),
                           lambda: mean)
        if exp:
            tmp1 = tf.exp(tmp1)
        value = tmp1
    else:
        raise ValueError('Unknown distribution type %s.' % rand_type)
    return value


class AugmentationCoeff():
    def __init__(self, params, width, height, global_step, old_coeffs=None):
        self.crop_width = params['crop_width']
        self.crop_height = params['crop_height']

        # Setup exponential decay so preprocessing amount changes as training continues
        if 'coeff_schedule_param' in params:
            initial_coeff = params['coeff_schedule_param']['initial_coeff']
            final_coeff = params['coeff_schedule_param']['final_coeff']
            half_life = params['coeff_schedule_param']['half_life']
            multiplier = 2.0 / (1.0 + tf.exp(-1.0986 * (tf.to_float(global_step) / half_life))) - 1
            discount_coeff = initial_coeff + (final_coeff - initial_coeff) * multiplier
        else:
            discount_coeff = tf.constant(1.0)

        # Set initial coefficients to default values
        for (name, value) in ALL_DEFAULT_TYPES.iteritems():
            setattr(self, name, value)

        # Setup coefficients from parameters
        self.generate_valid_spatial_coeffs(params, discount_coeff, width, height, old_coeffs)
        self.generate_coeffs(params, DEFAULT_CHROMATIC_TYPES, discount_coeff, old_coeffs)
        # TODO: self.generate_chromatic_eigen_coeffs(params, discount_coeff)
        self.generate_coeffs(params, DEFAULT_EFFECT_TYPES, discount_coeff, old_coeffs)

    def is_set(self, name):
        return tf.not_equal(getattr(self, name), ALL_DEFAULT_TYPES[name])

    def generate_valid_spatial_coeffs(self, params, discount_coeff, width, height, old_coeffs=None):
        # Short circuit if we don't have any spatial transformations defined in the parameters
        if len(set(DEFAULT_SPATIAL_TYPES.keys()).intersection(params.keys())) == 0:
            return

        # We want all four corners of the transformed image to fit into the original image.
        # We will try 50 times before bailing.
        corners = tf.constant([[0., 0.],
                               [0., self.crop_height - 1.],
                               [self.crop_width - 1., 0.],
                               [self.crop_width - 1., self.crop_height - 1.]], dtype=tf.float32)
        half_crop_width = 0.5 * self.crop_width
        half_crop_height = 0.5 * self.crop_height
        half_width = 0.5 * width
        half_height = 0.5 * height

        # TODO: it may be faster to just do the operation 50 times and pick the first good one
        # than to do a while loop. Need to benchmark

        def condition(iter_count, good_corner_count, spatial_coeffs):
            return tf.logical_and(iter_count < 50, good_corner_count < 4)

        def body(iter_count, good_corner_count, spatial_coeffs):
            # Generate new random spatial coefficients
            spatial_coeffs = self.generate_spatial_coeffs(params, discount_coeff, old_coeffs)

            # Move the origin
            points = corners - [half_crop_width, half_crop_height]

            # Rotation
            sin_theta = tf.sin(self.rotate)
            cos_theta = tf.cos(self.rotate)
            rotation_mat = tf.stack([cos_theta, sin_theta, -sin_theta, cos_theta])
            rotation_mat = tf.reshape(rotation_mat, [2, 2])
            points = tf.matmul(points, rotation_mat)

            # Translation
            points = points + [self.translate_x * self.crop_width,
                               self.translate_y * self.crop_height]

            # Zoom
            points = points / [self.zoom_x, self.zoom_y]

            # Move the origin back
            points = points + [half_width, half_height]

            floor_points = tf.floor(points)

            condition = tf.reduce_all(tf.logical_and(floor_points >= 0,
                                                     floor_points <= [width - 2, height - 2]))
            new_corner_count = tf.cond(condition,
                                       lambda: good_corner_count + 1,
                                       lambda: good_corner_count)

            return (iter_count + 1, new_corner_count, spatial_coeffs)

        initial_spatial_coeffs = tf.zeros([len(DEFAULT_SPATIAL_TYPES.keys())])
        final_count, _, spatial_coeffs = tf.while_loop(condition, body, [0, 0, initial_spatial_coeffs])

        # Bind each coeff Tensor to object. If we could not find suitable spatial coeffs,
        # we apply the old transformation if it exists. If not, we use the default value.
        for (idx, name) in enumerate(DEFAULT_SPATIAL_TYPES.keys()):
            if old_coeffs:
                new_coeff = tf.cond(tf.equal(final_count, 50),
                                    lambda: getattr(old_coeffs, name),
                                    lambda: spatial_coeffs[idx])
            else:
                new_coeff = tf.cond(tf.equal(final_count, 50),
                                    lambda: DEFAULT_SPATIAL_TYPES[name],
                                    lambda: spatial_coeffs[idx])
            setattr(self, name, new_coeff)

    # https://github.com/lmb-freiburg/flownet2/blob/476d400834d96337165d8561a8ef469ecfea77cd/src/caffe/layers/augmentation_layer_base.cpp#L73
    def generate_spatial_coeffs(self, params, discount_coeff, old_coeffs=None):
        coeffs = []

        # Generate new coefficients
        for name in DEFAULT_SPATIAL_TYPES.keys():
            if name in params:
                coeffs.append(_generate_coeff(params[name],
                                               discount_coeff,
                                               DEFAULT_SPATIAL_TYPES[name]))
            else:
                coeffs.append(DEFAULT_SPATIAL_TYPES[name])

        # Handle the special 'squeeze' param. It doesn't store a value itself, just alters zoom.
        if 'squeeze' in params:
            coeff = _generate_coeff(params['squeeze'], discount_coeff, 1.0)
            zoom_x_idx = DEFAULT_SPATIAL_TYPES.keys().index('zoom_x')
            zoom_y_idx = DEFAULT_SPATIAL_TYPES.keys().index('zoom_y')
            coeffs[zoom_x_idx] = coeffs[zoom_x_idx] * coeff
            coeffs[zoom_y_idx] = coeffs[zoom_y_idx] / coeff

        # Combine with old coefficients
        if old_coeffs:
            for (idx, name) in enumerate(DEFAULT_SPATIAL_TYPES.keys()):
                our_coeff = coeffs[idx]
                their_coeff = getattr(old_coeffs, name)
                combined_coeff = tf.cond(old_coeffs.is_set(name),
                                         lambda: our_coeff * their_coeff,
                                         lambda: our_coeff)
                coeffs[idx] = combined_coeff

        return tf.stack(coeffs)

    def generate_coeffs(self, params, default_dict, discount_coeff, old_coeffs=None):
        for name in default_dict.keys():
            # Generate new coefficients if needed, otherwise use default value
            if name in params:
                new_coeff = _generate_coeff(params[name], discount_coeff, default_dict[name])
            else:
                new_coeff = default_dict[name]
            # Combine new coefficient with old coefficient if it exists
            if old_coeffs:
                new_coeff = tf.cond(old_coeffs.is_set(name),
                                    lambda: new_coeff * getattr(old_coeffs, name),
                                    lambda: new_coeff)
            setattr(self, name, new_coeff)

    # TODO: def generate_chromatic_eigen_coeffs(self, params, discount_coeff):
    #     if 'ladd_pow' in params:
    #         self.pow_nomean0 = _generate_coeff(params['ladd_pow'], discount_coeff)
    #     if 'col_pow' in params:
    #         self.pow_nomean1 = _generate_coeff(params['col_pow'], discount_coeff)
    #         self.pow_nomean2 = _generate_coeff(params['col_pow'], discount_coeff)
    #     if 'ladd_add' in params:
    #         self.add_nomean0 = _generate_coeff(params['ladd_add'], discount_coeff)
    #     if 'col_add' in params:
    #         self.add_nomean1 = _generate_coeff(params['col_add'], discount_coeff)
    #         self.add_nomean2 = _generate_coeff(params['col_add'], discount_coeff)
    #     if 'ladd_mult' in params:
    #         self.mult_nomean0 = _generate_coeff(params['ladd_mult'], discount_coeff)
    #     if 'col_mult' in params:
    #         self.mult_nomean1 = _generate_coeff(params['col_mult'], discount_coeff)
    #         self.mult_nomean2 = _generate_coeff(params['col_mult'], discount_coeff)
    #     if 'sat_pow' in params:
    #         self.pow_withmean1 = _generate_coeff(params['sat_pow'], discount_coeff)
    #         self.pow_withmean2 = self.pow_withmean1
    #     if 'sat_add' in params:
    #         self.add_withmean1 = _generate_coeff(params['sat_add'], discount_coeff)
    #         self.add_withmean2 = self.add_withmean1
    #     if 'sat_mult' in params:
    #         self.mult_withmean1 = _generate_coeff(params['sat_mult'], discount_coeff)
    #         self.mult_withmean2 = self.mult_withmean1
    #     if 'lmult_pow' in params:
    #         self.lmult_pow = _generate_coeff(params['lmult_pow'], discount_coeff)
    #     if 'lmult_mult' in params:
    #         self.lmult_mult = _generate_coeff(params['lmult_mult'], discount_coeff)
    #     if 'lmult_add' in params:
    #         self.lmult_add = _generate_coeff(params['lmult_add'], discount_coeff)
    #     if 'col_rotate' in params:
    #         self.col_angle = _generate_coeff(params['col_rotate'], discount_coeff)
