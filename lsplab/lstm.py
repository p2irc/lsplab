from . import layers
import tensorflow as tf

class lstm(object):
    __hidden_units = 16
    __model = None
    __output_layer1 = None
    __output_layer2 = None

    def __init__(self, batch_size, num_units, graph):
        self.__model = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units=num_units)

        regularization_coefficient = 0.005

        layer1 = layers.fullyConnectedLayer('lstm_output1',
                                           [batch_size, num_units],
                                           self.__hidden_units,
                                           False,
                                           batch_size,
                                           'tanh',
                                           'xavier',
                                           regularization_coefficient)

        layer2 = layers.fullyConnectedLayer('lstm_output2',
                                           [batch_size, self.__hidden_units],
                                           1,
                                           False,
                                           batch_size,
                                           None,
                                           'xavier',
                                           regularization_coefficient)

        layer1.add_to_graph(graph)
        layer2.add_to_graph(graph)

        self.__output_layer1 = layer1
        self.__output_layer2 = layer2

    def forward_pass(self, input):
        """Returns predicted_stress, predicted_treatment"""
        activations, _ = tf.contrib.rnn.static_rnn(self.__model, input, dtype='float32')
        step_last = activations[-1]

        predicted_treatment = tf.squeeze(self.__output_layer2.forward_pass(self.__output_layer1.forward_pass(step_last, deterministic=True), deterministic=True))

        return predicted_treatment, activations

    def get_regularization_loss(self):
        return self.__output_layer1.regularization_coefficient * tf.nn.l2_loss(self.__output_layer1.weights) + \
               self.__output_layer2.regularization_coefficient * tf.nn.l2_loss(self.__output_layer2.weights)
