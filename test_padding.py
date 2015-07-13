# encoding: utf-8
"""Simple script to test the effects of padding on sequence labeling.

The toy problem this RNN is trying to solve is categorizing inputs as either
positive or negative:
input [-5] ==> output [1, 0]
input [5] ==> output [0, 1]
(Intentionally I use categories and not simple binary classification)

The layout is simple, and the network get one float each timestep, and
uses SimpleRNN to an output layer of size 2 (one for each category).
Timesteps doesn't really affect each other, but are here to test the padding.

"""

from __future__ import division, unicode_literals, print_function

import time
import random
import numpy as np
import theano
import theano.tensor as T
from keras.models import Sequential
from keras.layers.recurrent import SimpleRNN
from keras.layers.core import MaskedLayer

floatX = theano.config.floatX # pylint: disable=no-member,invalid-name
random.seed(1)
np.random.seed(1) # pylint: disable=no-member

class Masking(MaskedLayer):
    """Simple masking layer which gets a sentinal value.

    The output masks is 0 where the sentinal value is present, or 1 otherwise.

    """

    def __init__(self, mask_value=0, ndim=3, dtype=floatX):
        super(MaskedLayer, self).__init__()
        ttype = T.TensorType(dtype, [False]*ndim)
        self.input = ttype()
        self.mask_value = mask_value

    def get_output_mask(self, train=False):
        X = self.get_input(train)
        return T.any(T.ones_like(X) * (1 - T.eq(X, self.mask_value)), axis=-1)


def test_masking():
    """Test get_output_mask() really returns a correct mask"""
    layer = Masking(mask_value=100)
    f = theano.function([layer.input], layer.get_output_mask())
    # assert np.all(f(np.array([[[0, 1, 100, 1, 100]]])) == np.array([[[1, 1, 0, 1, 0]]]))
    print(f(np.array([[[0], [1], [100], [1], [100]]])))
    print("Test Passed!")


def main(samples, nb_epoch, use_layer, mask_zero, x_mask_value, y_mask_value, use_weights, padding):
    """Runs an experiment.

    Args:
        samples: list of 2-d numpy arrays,  not padded
        nb_epoch: number of epoch to run
        use_layer: use Masking layer before the SimpleRNN layer
        mask_zero: use the `mask_zero` parameter introduced in PR #361
        x_mask_value: what sentinal value to use (only with `use_layer`)
        y_mask_value: what is the padding value for the targets
        use_weights: should `samples_weight` be used to make the loss function ignore padding
        padding: 'pre' for pre-padding the data, 'post' for post-padding, None for no padding

    Returns:
        history object of fit()

    """
    # preparing the data
    maxlen = max(len(x) for x in samples)
    print("\n\nConf: nb_epoch={} use_layer={} mask_zero={} x_mask_value={} y_mask_value={}"
          " use_weights={} padding={}".format(
              nb_epoch, use_layer, mask_zero, x_mask_value, y_mask_value, use_weights, padding))
    if padding:
        X = np.zeros((len(samples), maxlen, 1), dtype=floatX) + x_mask_value
        for i, sample in enumerate(samples):
            if padding == 'pre':
                X[i, maxlen-len(sample):] = sample
            else:
                X[i, :len(sample)] = sample
        samples = X
    else:
        samples = np.asarray(samples)

    targets, weights = [], []
    for sample in samples:
        target, weight = [], []
        for timestep in sample:
            if timestep < 0:
                target.append(np.array([1, 0], dtype=np.int32))
                weight.append(np.array([1], dtype=np.int8))
            elif timestep != x_mask_value:
                target.append(np.array([0, 1], dtype=np.int32))
                weight.append(np.array([1], dtype=np.int8))
            else:
                target.append(np.array(y_mask_value, dtype=np.int32))
                weight.append(np.array([0], dtype=np.int8))
        targets.append(target)
        weights.append(weight)
    targets = np.asarray(targets)
    weights = np.asarray(weights)

    for name in ('samples', 'targets', 'weights'):
        value = locals()[name]
        print("{} shape: {}".format(name, value.shape))
        print(str(value[0]).replace('\n', ' '))

    # model
    layers = [Masking(x_mask_value)] if use_layer else []
    if mask_zero: # must merge PR #361 to work
        layers.append(SimpleRNN(1, 2, return_sequences=True, mask_zero=True))
    else:
        layers.append(SimpleRNN(1, 2, return_sequences=True))
    model = Sequential(layers)
    model.compile('sgd', 'categorical_crossentropy', theano_mode='FAST_RUN')
    tic = time.time()
    history = model.fit(samples, targets, nb_epoch=nb_epoch,
                        validation_split=0.2, show_accuracy=True, verbose=2,
                        sample_weight=weights if use_weights else None)
    print("Training time: {:.2f} seconds".format(time.time() - tic))
    return history


def generate_data(nsamples):
    """Generate `nsamples` random data in range [-10, 10]. All zeros are converted"""
    data = np.random.uniform(-10.0, 10.0, size=nsamples)
    for i, x in enumerate(data):
        if x == 0.0:
            if random.randint(0, 1):
                data[i] = 0.5
            else:
                data[i] = -0.5
    assert np.all(data != 0)
    return data


def split_data(data, minlen, maxlen):
    """Split one big `data` array to a list of small arrays with lengths between min and max"""
    out = []
    i = 0
    while i < len(data):
        size = min(random.randint(minlen, maxlen), len(data) - i)
        arr = data[i:i+size].reshape((size, 1))
        i += size
        out.append(arr)
    return out


if __name__ == '__main__':
    test_masking()
    data = generate_data(15*100000)
    same_length = split_data(data, 15, 15)
    var_length = split_data(data, 10, 20)
    main(same_length, 10, use_layer=False, mask_zero=False, x_mask_value=0, y_mask_value=[0, 0], use_weights=False, padding=False)
    main(var_length, 10, use_layer=False, mask_zero=False, x_mask_value=0, y_mask_value=[0, 0], use_weights=False, padding='pre')
    main(var_length, 10, use_layer=False, mask_zero=False, x_mask_value=0, y_mask_value=[0, 0], use_weights=False, padding='post')
    main(var_length, 10, use_layer=True, mask_zero=False, x_mask_value=0, y_mask_value=[0, 0], use_weights=False, padding='pre')
    main(var_length, 10, use_layer=True, mask_zero=False, x_mask_value=0, y_mask_value=[0, 0], use_weights=False, padding='post')
    main(var_length, 10, use_layer=True, mask_zero=False, x_mask_value=0, y_mask_value=[0, 0], use_weights=True, padding='pre')
    main(var_length, 10, use_layer=True, mask_zero=False, x_mask_value=0, y_mask_value=[1, 1], use_weights=True, padding='pre')
    main(var_length, 10, use_layer=True, mask_zero=False, x_mask_value=0, y_mask_value=[0, 0], use_weights=True, padding='post')
    main(var_length, 10, use_layer=True, mask_zero=False, x_mask_value=100, y_mask_value=[0, 0], use_weights=True, padding='post')
    main(var_length, 10, use_layer=False, mask_zero=True, x_mask_value=0, y_mask_value=[0, 0], use_weights=True, padding='pre')
    main(var_length, 10, use_layer=False, mask_zero=True, x_mask_value=0, y_mask_value=[1, 1], use_weights=True, padding='pre')
    main(var_length, 10, use_layer=False, mask_zero=True, x_mask_value=0, y_mask_value=[1, 1], use_weights=False, padding='pre')
    main(var_length, 10, use_layer=False, mask_zero=True, x_mask_value=0, y_mask_value=[0, 0], use_weights=True, padding='post')
    main(var_length, 10, use_layer=False, mask_zero=True, x_mask_value=0, y_mask_value=[1, 1], use_weights=True, padding='post')
