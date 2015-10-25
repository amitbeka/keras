# encoding: utf -8
"""Bidirectional RNN support from EderSantana/seya"""

import types
import theano.tensor as T

from keras.layers.recurrent import Recurrent

def _get_reversed_input(self, train=False):
    if hasattr(self, 'previous'):
        X = self.previous.get_output(train=train)
    else:
        X = self.input
    return X[::-1]


class Bidirectional(Recurrent):
    def __init__(self, forward, backward, return_sequences=False,
                 truncate_gradient=-1):
        super(Bidirectional, self).__init__()
        self.forward = forward
        self.backward = backward
        self.return_sequences = return_sequences
        self.truncate_gradient = truncate_gradient
        self.output_dim = self.forward.output_dim + self.backward.output_dim
        #if self.forward.output_dim != self.backward.output_dim:
        #    raise ValueError("Make sure `forward` and `backward` have " +
        #                     "the same `ouput_dim.`")

        rs = (self.return_sequences, forward.return_sequences,
              backward.return_sequences)
        if rs[1:] != rs[:-1]:
            raise ValueError("Make sure 'return_sequences' is equal for self," +
                             " forward and backward.")
        tg = (self.truncate_gradient, forward.truncate_gradient,
              backward.truncate_gradient)
        if tg[1:] != tg[:-1]:
            raise ValueError("Make sure 'truncate_gradient' is equal for self," +
                             " forward and backward.")

    def build(self):
        self.input = T.tensor3()
        self.forward.input = self.input
        self.backward.input = self.input
        self.forward.build()
        self.backward.build()
        self.params = self.forward.params + self.backward.params

    def set_previous(self, layer, connection_map={}):
        assert self.nb_input == layer.nb_output == 1, "Cannot connect layers: input count and output count should be 1."
        if hasattr(self, 'input_ndim'):
            assert self.input_ndim == len(layer.output_shape), "Incompatible shapes: layer expected input with ndim=" +\
                str(self.input_ndim) + " but previous layer has output_shape " + str(layer.output_shape)
        self.forward.set_previous(layer, connection_map)
        self.backward.set_previous(layer, connection_map)
        self.backward.get_input = types.MethodType(_get_reversed_input, self.backward)
        self.previous = layer
        self.build()

    @property
    def output_shape(self):
        input_shape = self.input_shape
        #f_out = self.forward.output_dim
        #b_out = self.backward.output_dim
        output_dim = self.output_dim
        if self.return_sequences:
            return (input_shape[0], input_shape[1], output_dim)
        else:
            return (input_shape[0], output_dim)

    def get_output(self, train=False):
        Xf = self.forward.get_output(train)
        Xb = self.backward.get_output(train)
        Xb = Xb[::-1]
        return T.concatenate([Xf, Xb], axis=-1)

    def get_config(self):
        new_dict = {}
        for k, v in self.forward.get_cofig.items():
            new_dict['forward_'+k] = v
        for k, v in self.backward.get_cofig.items():
            new_dict['backward_'+k] = v
        new_dict["name"] = self.__class__.__name__
        return new_dict
