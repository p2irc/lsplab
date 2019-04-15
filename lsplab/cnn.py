from . import layers

import tensorflow as tf
import datetime
import copy


class cnn(object):
    # Options
    __debug = False
    __weight_initializer = 'xavier'

    # Input formatting
    __image_width = None
    __image_height = None
    __image_depth = None

    # Network internal representation
    layers = []
    __global_epoch = 0
    __batch_size = None
    __output_size = None

    # Layer counts
    __num_layers_norm = 0
    __num_layers_conv = 0
    __num_layers_pool = 0
    __num_layers_fc = 0
    __num_layers_dropout = 0
    __num_layers_batchnorm = 0
    __num_layers_upsample = 0

    # Augmentation
    __augmentation_crop = False
    __crop_amount = 0.8

    __name_prefix = ""

    def __init__(self, debug, batch_size, name_prefix=""):
        self.__debug = debug
        self.__batch_size = batch_size
        self.__name_prefix = name_prefix

        self.layers = []

    def __log(self, message):
        if self.__debug:
            print('{0}: CNN - {1}'.format(datetime.datetime.now().strftime("%I:%M%p"), message))

    def __last_layer(self):
        return self.layers[-1]

    def __last_layer_outputs_volume(self):
        #return isinstance(self.__last_layer().output_size, (list,))
        return len(self.last_layer_output_size()) > 2

    def last_layer_output_size(self):
        return copy.deepcopy(self.__last_layer().output_size)

    def first_layer(self):
        return next(layer for layer in self.layers if
                    isinstance(layer, layers.convLayer) or isinstance(layer, layers.fullyConnectedLayer))

    def set_image_dimensions(self, image_height, image_width, image_depth):
        """Specify the image dimensions for images in the dataset (depth is the number of channels)"""
        self.__image_width = image_width
        self.__image_height = image_height
        self.__image_depth = image_depth

    def get_output_size(self):
        return self.__output_size

    def add_input_layer(self, reshape=False):
        """Add an input layer to the network"""
        self.__log('Adding the input layer...')

        size = [self.__batch_size, self.__image_height, self.__image_width, self.__image_depth]

        layer = layers.inputLayer(size, reshape)

        self.layers.append(layer)

    def add_convolutional_layer(self, filter_dimension, stride_length, activation_function):
        """
        Add a convolutional layer to the model.

        :param filter_dimension: array of dimensions in the format [x_size, y_size, depth, num_filters]
        :param stride_length: convolution stride length
        :param activation_function: the activation function to apply to the activation map
        """
        self.__num_layers_conv += 1
        layer_name = self.__name_prefix + 'conv%d' % self.__num_layers_conv
        self.__log('Adding convolutional layer %s...' % layer_name)

        reshape = not self.__last_layer_outputs_volume()

        layer = layers.convLayer(layer_name,
                                 self.last_layer_output_size(),
                                 filter_dimension,
                                 stride_length,
                                 activation_function,
                                 self.__weight_initializer,
                                 reshape)

        self.__log('Filter dimensions: {0} Outputs: {1}'.format(filter_dimension, layer.output_size))

        self.layers.append(layer)

    def add_pooling_layer(self, kernel_size, stride_length, pooling_type='max'):
        """
        Add a pooling layer to the model.

        :param kernel_size: an integer representing the width and height dimensions of the pooling operation
        :param stride_length: convolution stride length
        :param pooling_type: optional, the type of pooling operation
        """
        self.__num_layers_pool += 1
        layer_name = self.__name_prefix + 'pool%d' % self.__num_layers_pool
        self.__log('Adding pooling layer %s...' % layer_name)

        layer = layers.poolingLayer(self.last_layer_output_size(), kernel_size, stride_length, pooling_type)

        self.__log('Outputs: %s' % layer.output_size)

        self.layers.append(layer)

    def add_dropout_layer(self, p):
        """
        Add a DropOut layer to the model.

        :param p: the keep-probability parameter for the DropOut operation
        """
        self.__num_layers_dropout += 1
        layer_name = self.__name_prefix + 'drop%d' % self.__num_layers_dropout
        self.__log('Adding dropout layer %s...' % layer_name)

        layer = layers.dropoutLayer(self.last_layer_output_size(), p)

        self.layers.append(layer)

    def add_batchnorm_layer(self):
        """Add a batch normalization layer to the model."""
        self.__num_layers_batchnorm += 1
        layer_name = self.__name_prefix + 'bn%d' % self.__num_layers_batchnorm
        self.__log('Adding batch norm layer %s...' % layer_name)

        layer = layers.batchNormLayer(layer_name, self.last_layer_output_size())

        self.layers.append(layer)

    def add_fully_connected_layer(self, output_size, activation_function, regularization_coefficient=0.):
        """
        Add a fully connected layer to the model.

        :param output_size: the number of units in the layer
        :param activation_function: optionally, the activation function to use
        :param regularization_coefficient: optionally, an L2 decay coefficient for this layer (overrides the coefficient
         set by set_regularization_coefficient)
        """
        self.__num_layers_fc += 1
        layer_name = self.__name_prefix + 'fc%d' % self.__num_layers_fc
        self.__log('Adding fully connected layer %s...' % layer_name)

        reshape = self.__last_layer_outputs_volume()

        layer = layers.fullyConnectedLayer(layer_name,
                                           self.last_layer_output_size(),
                                           output_size,
                                           reshape,
                                           self.__batch_size,
                                           activation_function,
                                           self.__weight_initializer,
                                           regularization_coefficient)

        self.__log('Inputs: {0} Outputs: {1}'.format(layer.input_size, layer.output_size))

        self.layers.append(layer)

    def add_upsampling_layer(self, filter_size, num_filters, upscale_factor=2, activation_function=None):
        """
        Add a 2d upsampling layer to the model.
        :param filter_size: an int, representing the dimension of the square filter to be used
        :param num_filters: an int, representing the number of filters that will be outputted (the output tensor depth)
        :param upscale_factor: an int, or tuple of ints, representing the upsampling factor for rows and columns
        :param activation_function: the activation function to apply to the activation map
        :param regularization_coefficient: optionally, an L2 decay coefficient for this layer (overrides the coefficient
         set by set_regularization_coefficient)
        """
        self.__num_layers_upsample += 1
        layer_name = self.__name_prefix + 'upsample%d' % self.__num_layers_upsample
        self.__log('Adding upsampling layer %s...' % layer_name)

        layer = layers.upsampleLayer(layer_name,
                                     self.last_layer_output_size(),
                                     filter_size,
                                     num_filters,
                                     upscale_factor,
                                     activation_function,
                                     self.__weight_initializer)

        self.__log('Filter dimensions: {0} Outputs: {1}'.format(layer.weights_shape, layer.output_size))

        self.layers.append(layer)

    def add_output_layer(self, regularization_coefficient=None, output_size=8, activation_function='relu'):
        """
        Add an output layer to the network (affine layer where the number of units equals the number of network outputs)

        :param regularization_coefficient: optionally, an L2 decay coefficient for this layer (overrides the coefficient
         set by set_regularization_coefficient)
        :param output_size: optionally, override the output size of this layer. Typically not needed, but required for
        use cases such as creating the output layer before loading data.
        """
        self.__log('Adding output layer...')

        reshape = self.__last_layer_outputs_volume()
        num_out = output_size
        self.__output_size = output_size

        layer = layers.fullyConnectedLayer('output',
                                           self.last_layer_output_size(),
                                           num_out,
                                           reshape,
                                           self.__batch_size,
                                           activation_function,
                                           self.__weight_initializer,
                                           regularization_coefficient)

        self.__log('Inputs: {0} Outputs: {1}'.format(layer.input_size, layer.output_size))

        self.layers.append(layer)

    def send_ops_to_graph(self, graph):
        for layer in self.layers:
            if callable(getattr(layer, 'add_to_graph', None)):
                layer.add_to_graph(graph)

    def forward_pass(self, x, deterministic=False):
        """
        Perform a forward pass of the network with an input tensor.
        In general, this is only used when the model is integrated into a Tensorflow graph.
        See also forward_pass_with_file_inputs.

        :param x: input tensor where the first dimension is batch
        :param deterministic: if True, performs inference-time operations on stochastic layers e.g. DropOut layers
        :return: output tensor where the first dimension is batch
        """
        for layer in self.layers:
                x = layer.forward_pass(x, deterministic)

        return x

    def get_regularization_loss(self):
        l2_cost = tf.squeeze(tf.reduce_sum(
            [layer.regularization_coefficient * tf.nn.l2_loss(layer.weights) for layer in self.layers
             if isinstance(layer, layers.fullyConnectedLayer)]))

        return l2_cost

