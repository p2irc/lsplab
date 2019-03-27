from . import layers
import tensorflow as tf

class lstm(object):
    __model = None
    __output_layer = None

    def __init__(self, batch_size, num_units, graph):
        self.__model = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units=num_units)

        regularization_coefficient = 0.005

        layer = layers.fullyConnectedLayer('lstm_output',
                                           [batch_size, num_units],
                                           1,
                                           False,
                                           batch_size,
                                           'tanh',
                                           'xavier',
                                           regularization_coefficient)

        layer.add_to_graph(graph)

        self.__output_layer = layer

    def forward_pass(self, input):
        """Returns predicted_stress, predicted_treatment"""
        activations, _ = tf.contrib.rnn.static_rnn(self.__model, input, dtype='float32')
        step_last = activations[-1]

        predicted_treatment = tf.squeeze(self.__output_layer.forward_pass(step_last, deterministic=True))

        return predicted_treatment, activations

    def get_regularization_loss(self):
        return self.__output_layer.regularization_coefficient * tf.nn.l2_loss(self.__output_layer.weights)
