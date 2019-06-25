from . import biotools
from . import cnn
from . import lstm
from . import plotter
from . import reporter
from . import timer
from . import layers

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from tqdm import trange
import emoji
import saliency
from PIL import Image
from joblib import load
import matplotlib.pyplot as plt

import datetime
import math
import os
import glob
import time
import shutil
import copy
import random


class lsp(object):
    # Options
    __debug = False
    __reporter = reporter.reporter()
    __batch_size = None
    __report_rate = None
    __loss_function = 'sce'
    __image_depth = 3
    __num_fold_restarts = 10
    __num_failed_attempts = 0
    __mode = 'longitudinal'
    __random_seed = None
    __use_batchnorm = False

    __current_fold = None
    __num_folds = None

    __pretraining_batches = None
    __training_batches = None
    __cache_files = []

    __major_queue_capacity = 64

    __main_lr = 0.001
    __global_weight_decay = 0.0001
    __global_reg = 0.0005
    __variance_constant = 0.2
    __lstm_units = 8

    __pretrain_convergence_thresh_upper = 0.5

    # Image options stuff
    __do_augmentation = False
    __do_crop = False
    __crop_amount = 0.75
    __standardize_images = True

    # Dataset info
    __num_records_test = None
    __num_timepoints = None
    __cache_filename = None
    __use_memory_cache = False

    # Graph machinery
    __session = None
    __graph = None
    __coord = None
    __threads = None
    __num_gpus = 1
    __num_threads = 4

    # Graph components
    __input_batch_train = None
    __input_batch_test = None
    __inorder_input_batch_test = None

    # Subgraph objects
    transformer_net = None
    feature_extractor = None
    lstm = None
    __decoder_net = None
    __n = 16
    __decoder_iterations = 30000
    __geodesic_path_iterations = 400
    __transformer_iterations = 25000
    __target_vertices = 30

    # Inter-space transformation stuff
    __transformation_method = 'Linear'
    __input_batch_canonical = None
    __trans_score_threshold = 0.5

    # Tensorboard stuff
    __pretraining_summaries = None
    __decoder_summaries = None
    __training_summaries = None
    __tb_writer = None
    __tb_file = None
    results_path = None

    # Results
    __all_projections = []
    __can_projections = []
    __num_can_points = None
    __total_treated_skipped = 0
    __features = None
    __p_values = None
    __key_path = None
    __global_timer = None

    def __init__(self, debug, batch_size=8):
        self.__debug = debug
        self.__batch_size = batch_size
        self.__global_timer = timer.timer()
        self.__total_treated_skipped = 0

    def __log(self, message):
        if self.__debug:
            if self.__current_fold is not None:
                print('{0}: (Fold {1}) {2}'.format(datetime.datetime.now().strftime("%I:%M%p"), self.__current_fold, message.encode('utf-8')))
            else:
                print('{0}: {1}'.format(datetime.datetime.now().strftime("%I:%M%p"), message.encode('utf-8')))

    def __initialize(self):
        self.__log('Initializing variables...')

        with self.__graph.as_default():
            if self.__random_seed is not None:
                tf.set_random_seed(self.__random_seed)

            self.__session.run(tf.global_variables_initializer())
            self.__session.run(tf.local_variables_initializer())
            self.__session.run(self.__queue_init_ops)

    def __shutdown(self):
        self.__log('Shutting down...')
        self.__session.close()

        self.__graph = None
        self.__session = None

        if self.__use_memory_cache is False:
            self.__log('Removing cache files...')

            for file in self.__cache_files:
                for filename in glob.glob('/tmp/{0}*'.format(file)):
                    os.remove(filename)

            self.__cache_files = []

    def __reset_graph(self):
        # Reset all graph elements
        self.__graph = tf.Graph()
        self.__session = tf.Session(graph=self.__graph)

    def set_random_seed(self, seed):
        self.__random_seed = seed

    def set_use_batchnorm(self, use_batchnorm):
        self.__use_batchnorm = use_batchnorm

    def save_state(self, directory=None):
        """Save all trainable variables as a checkpoint in the current working path"""
        self.__log('Saving parameters...')

        if directory is None:
            dir = './saved_state'
        else:
            dir = os.path.join(directory, 'saved_state')

        if not os.path.isdir(dir):
            os.mkdir(dir)

        with self.__graph.as_default():
            saver = tf.train.Saver(tf.trainable_variables())
            saver.save(self.__session, dir + '/tfhSaved')

    def __save_decoder(self):
        """Save all trainable variables as a checkpoint in the current working path"""
        self.__log('Saving decoder...')

        dir = os.path.join(self.results_path, 'decoder_vars', 'decoder_vars')

        with self.__graph.as_default():
            saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder'))
            saver.save(self.__session, dir)

    def __load_decoder(self):
        """Save all trainable variables as a checkpoint in the current working path"""
        self.__log('Loading decoder...')

        directory = os.path.join(self.results_path, 'decoder_vars')

        with self.__graph.as_default():
            saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder'))
            saver.restore(self.__session, tf.train.latest_checkpoint(directory))

    def load_state(self, directory='./saved_state'):
        """
        Load all trainable variables from a checkpoint file specified from the load_from_saved parameter in the
        class constructor.
        """
        self.__log('Loading from checkpoint file...')

        with self.__graph.as_default():
            saver = tf.train.Saver(tf.trainable_variables())
            saver.restore(self.__session, tf.train.latest_checkpoint(directory))

    def __save_as_image(self, mat, path):
        plt.clf()
        plt.imshow(mat, cmap='gray', vmin=0., vmax=1.)
        plt.savefig(path)

    def set_loss_function(self, lf):
        '''Set the loss function to use'''
        self.__loss_function = lf

    def set_transformation_method(self, m):
        self.__transformation_method = m

        if m == 'NeuralNet':
            self.__trans_score_threshold = 20.0

    def load_records(self, records_path, image_height, image_width, num_timepoints, image_depth=3):
        """Load records created from the dataset"""
        self.__record_files = [os.path.join(records_path, f) for f in os.listdir(records_path) if
                        os.path.isfile(os.path.join(records_path, f)) and not f.endswith('.csv')]

        self.__image_height = image_height
        self.__image_width = image_width
        self.__image_depth = image_depth
        self.__num_timepoints = num_timepoints

    def set_n(self, new_n):
        self.__n = new_n

    def set_augmentation(self, aug):
        self.__do_augmentation = aug

    def set_cropping_augmentation(self, do_crop, crop_amount=0.75):
        self.__do_crop = do_crop
        self.__crop_amount = crop_amount

    def set_standardize_images(self, standardize):
        self.__standardize_images = standardize

    def set_num_path_vertices(self, n):
        self.__target_vertices = n

    def set_num_decoder_iterations(self, n):
        self.__decoder_iterations = n

    def set_num_path_iterations(self, n):
        self.__geodesic_path_iterations = n

    def set_use_memory_cache(self, b):
        self.__use_memory_cache = b

    def set_variance_constant(self, c):
        self.__variance_constant = c

    def set_mode(self, mode):
        if mode == 'longitudinal' or mode == 'cross sectional':
            self.__mode = mode
        else:
            self.__log('Invalid mode set, must be longitudinal or cross sectional.')
            exit()

    def __initialize_data(self, can_fold_file):
        # Input pipelines for training
        self.__input_batch_train, init_op_1, cache_file_path = \
            biotools.get_sample_from_tfrecords_shuffled(self.__current_train_files,
                                                        self.__batch_size,
                                                        self.__image_height,
                                                        self.__image_width,
                                                        self.__image_depth,
                                                        self.__num_timepoints,
                                                        queue_capacity=self.__major_queue_capacity,
                                                        num_threads=self.__num_threads,
                                                        cached=True,
                                                        in_memory=self.__use_memory_cache)

        self.__cache_files.append(cache_file_path)

        # Input pipelines for testing

        self.__num_records_test = self.__get_num_records(self.__current_test_file)
        self.__log('Found {0} test records...'.format(self.__num_records_test))

        self.__input_batch_test, init_op_2, _ = \
            biotools.get_sample_from_tfrecords_shuffled(self.__current_test_file,
                                                        self.__batch_size,
                                                        self.__image_height,
                                                        self.__image_width,
                                                        self.__image_depth,
                                                        self.__num_timepoints,
                                                        queue_capacity=32,
                                                        num_threads=self.__num_threads,
                                                        cached=False,
                                                        in_memory=self.__use_memory_cache)

        self.__inorder_input_batch_test, init_op_3, _ = \
            biotools.get_sample_from_tfrecords_inorder(self.__current_test_file,
                                                       self.__batch_size,
                                                       self.__image_height,
                                                       self.__image_width,
                                                       self.__image_depth,
                                                       self.__num_timepoints,
                                                       queue_capacity=32,
                                                       num_threads=self.__num_threads,
                                                       cached=False,
                                                       in_memory=self.__use_memory_cache)

        self.__queue_init_ops = [init_op_1, init_op_2, init_op_3]

        # If this isn't the first fold, then also initialize a queue for the canonical points
        if self.__current_fold != 0:
            self.__input_batch_canonical, can_init_op, _ = \
                biotools.get_sample_from_tfrecords_inorder(can_fold_file,
                                                           self.__batch_size,
                                                           self.__image_height,
                                                           self.__image_width,
                                                           self.__image_depth,
                                                           self.__num_timepoints,
                                                           queue_capacity=32,
                                                           num_threads=self.__num_threads,
                                                           cached=False,
                                                           in_memory=self.__use_memory_cache)

            self.__queue_init_ops.append(can_init_op)


    def __get_num_records(self, records_path):
        self.__log('Counting records in {0}...'.format(records_path))
        return sum(1 for _ in tf.python_io.tf_record_iterator(records_path))

    def __resize_image(self, x):
        resized_height = int(self.__image_height * self.__crop_amount)
        resized_width = int(self.__image_width * self.__crop_amount)

        with self.__graph.device('/cpu:0'):
            image = tf.image.resize_image_with_crop_or_pad(x, resized_height, resized_width)

        return image

    def __apply_augmentations(self, image, resized_height, resized_width):
        with self.__graph.device('/cpu:0'):
            if self.__do_augmentation:
                image = tf.image.random_brightness(image, max_delta=0.5)
                image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
                image = tf.map_fn(lambda x: tf.image.random_flip_left_right(x), image)

            if self.__do_crop:
                image = tf.map_fn(lambda x: tf.random_crop(x, [resized_height, resized_width, self.__image_depth]), image)

        return image

    def __apply_image_standardization(self, image):
        with self.__graph.device('/cpu:0'):
            if self.__standardize_images:
                image = tf.map_fn(lambda x: tf.image.per_image_standardization(x), image)

        return image

    def __with_all_datapoints(self, op, datapoint_shape, num_records=None):
        """Get the results of running an op for all datapoints"""
        if num_records is None:
            num_records = self.__num_records_test

        total_batches = int(math.ceil(num_records / float(self.__batch_size)))
        remainder = (total_batches*self.__batch_size) - num_records
        outputs = np.empty(datapoint_shape)

        with self.__graph.as_default():
            for i in range(total_batches):
                batch_output = self.__session.run(op)
                batch_output = np.reshape(batch_output, [-1, datapoint_shape[1]])
                outputs = np.concatenate((outputs, batch_output), axis=0)

        outputs = np.delete(outputs, 0, axis=0)

        if remainder != 0:
            for i in range(remainder):
                outputs = np.delete(outputs, -1, axis=0)

        return outputs

    def __save_full_datapoints(self, id, treatment, processed_images, canon_matrix=None):
        """Gets the raw feature vector for all datapoints"""
        with self.__graph.as_default():
            # CNN embedding for first and last timepoints
            all_embeddings = tf.concat(processed_images, axis=1)

            # Append treatment and genotype data
            IID = tf.expand_dims(tf.cast(id, dtype=tf.float32), axis=-1)
            treatment = tf.expand_dims(tf.cast(treatment, dtype=tf.float32), axis=-1)
            all_features = tf.concat([IID, treatment, all_embeddings], axis=1)

            feature_length = (self.feature_extractor.get_output_size() * len(processed_images)) + 2

            all_outputs = self.__with_all_datapoints(all_features, [1, feature_length])

        temp = np.array(all_outputs)
        head = temp[:, :2]
        all_outputs_separated = [np.concatenate([head, temp[:, 2+(timestep*self.__n):2+((timestep+1)*self.__n)]], axis=1) for timestep in range(len(processed_images))]

        # If this is the canonical fold, save these as canonical points in canonical space
        if self.__current_fold == 0:
            self.__can_projections = copy.deepcopy(np.reshape(temp[:, 2:], [-1, self.__n]))
            self.__num_can_points = self.__num_records_test

        all_projections = []

        for i in range(self.__num_timepoints):
            all_projections.append([])

        for timestep in range(len(all_outputs_separated)):
            rows = all_outputs_separated[timestep]

            # Loop through all treated entries
            for row in rows:
                # Find corresponding non-treated entries
                IID = row[0]
                treatment = row[1]
                combined_features = row[2:]

                if canon_matrix is not None:
                    if self.__transformation_method == 'Linear':
                        org_shape = combined_features.shape
                        combined_features = canon_matrix.predict(combined_features.reshape((-1, self.feature_extractor.get_output_size())))
                        combined_features = combined_features.reshape(org_shape)
                    elif self.__transformation_method == 'NeuralNet':
                        combined_features = self.__predict_transformed_points(combined_features.reshape((-1, self.__n)))

                all_projections[timestep].append((int(IID), int(treatment), combined_features))

        # Also save the full data internally for later
        for timestep in range(len(all_outputs_separated)):
            self.__all_projections[timestep].extend(all_projections[timestep])

    def __log_loss(self, error):
        """Returns -log(1 - error)"""
        return -tf.log(tf.clip_by_value(1.0 - error, 1e-10, 1.0))

    def __linear_loss(self, vec1, vec2):
        return tf.abs(tf.subtract(vec1, vec2))

    def __sigmoid_cross_entropy_loss(self, treatment, logits):
        """Returns sigmoid cross entropy loss"""
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=treatment, logits=logits)

    def __get_treatment_loss(self, treatment, vec):
        treatment_float = tf.cast(treatment, dtype=tf.float32)

        if self.__loss_function == 'sce':
            losses = self.__sigmoid_cross_entropy_loss(treatment_float, vec)
        elif self.__loss_function == 'mse':
            self.__pretrain_convergence_thresh_upper = .2
            losses = tf.square(self.__linear_loss(treatment_float, vec))
        else:
            losses = self.__linear_loss(treatment_float, vec)

        return tf.reduce_mean(losses)

    def __get_clipped_gradients(self, loss, vars=None):
        optimizer = tf.train.AdamOptimizer(self.__main_lr)
        #optimizer = tf.contrib.opt.AdamWOptimizer(self.__global_weight_decay, learning_rate=self.__main_lr)
        gvs = optimizer.compute_gradients(loss, var_list=vars)
        capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]

        return capped_gvs, optimizer

    def __apply_gradients(self, gradients):
        optimizer = tf.train.AdamOptimizer(self.__main_lr)
        #optimizer = tf.contrib.opt.AdamWOptimizer(self.__global_weight_decay, learning_rate=self.__main_lr)
        objective = optimizer.apply_gradients(gradients)

        return objective

    def __minimize_with_clipped_gradients(self, loss, vars=None):
        capped_gvs, optimizer = self.__get_clipped_gradients(loss, vars=vars)
        objective = optimizer.apply_gradients(capped_gvs)

        return objective, capped_gvs

    def __average_gradients(self, tower_grads):
        average_grads = []

        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)

            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        return average_grads

    def __make_directory(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)

    def __parse_batch(self, batch_data):
        id = batch_data['id']
        treatment = batch_data['treatment']

        image_data = []

        for i in range(self.__num_timepoints):
            data = batch_data['image_data_{0}'.format(i)]
            image_data.append(data)

        return id, treatment, image_data

    def __learn_subspace_transform(self, targets, labels):
        # Train
        t = trange(self.__transformer_iterations)
        loss = None

        for i in t:
            indices = random.sample(range(len(targets)), self.__batch_size)
            c_batch = np.array([targets[k] for k in indices])
            c_labels = np.array([labels[k] for k in indices])

            fd = { self.__transformer_batch_pl: c_batch, self.__transformer_labels_pl: c_labels }
            _, loss = self.__session.run([self.__transformer_obj, self.__transformer_loss], feed_dict=fd)

            t.set_description('Loss: {0}'.format(loss))

        return loss

    def __predict_transformed_points(self, points):
        pred = self.__transformer_net.forward_pass(tf.cast(points, tf.float32))

        ret = self.__session.run(pred)

        return ret[0]

    def __build_transformer(self):
        # Define the structure of the network used for subspace transformation
        self.__transformer_net.add_input_layer()

        self.__transformer_net.add_fully_connected_layer(output_size=(self.__n * 2), activation_function='tanh', regularization_coefficient=self.__global_reg)

        self.__transformer_net.add_output_layer(output_size=self.__n, regularization_coefficient=self.__global_reg)

    def __build_convnet(self):
        # Define the structure of the CNN used for feature extraction
        self.feature_extractor.add_input_layer()

        self.feature_extractor.add_convolutional_layer(filter_dimension=[3, 3, self.__image_depth, 16], stride_length=1, activation_function='relu')
        self.feature_extractor.add_pooling_layer(kernel_size=3, stride_length=3)
        if self.__use_batchnorm:
            self.feature_extractor.add_batchnorm_layer()

        self.feature_extractor.add_convolutional_layer(filter_dimension=[3, 3, 16, 32], stride_length=1, activation_function='relu')
        self.feature_extractor.add_pooling_layer(kernel_size=3, stride_length=3)
        if self.__use_batchnorm:
            self.feature_extractor.add_batchnorm_layer()

        self.feature_extractor.add_convolutional_layer(filter_dimension=[3, 3, 32, 32], stride_length=1, activation_function='relu')
        self.feature_extractor.add_pooling_layer(kernel_size=3, stride_length=3)
        if self.__use_batchnorm:
            self.feature_extractor.add_batchnorm_layer()

        self.feature_extractor.add_convolutional_layer(filter_dimension=[3, 3, 32, 32], stride_length=1, activation_function='relu')
        self.feature_extractor.add_pooling_layer(kernel_size=3, stride_length=2)
        if self.__use_batchnorm:
            self.feature_extractor.add_batchnorm_layer()

        self.feature_extractor.add_fully_connected_layer(output_size=64, activation_function='relu', regularization_coefficient=self.__global_reg)

        self.feature_extractor.add_output_layer(output_size=self.__n, regularization_coefficient=self.__global_reg)

    def __build_decoder(self):
        self.__decoder_net.add_input_layer(reshape=True)

        self.__decoder_net.add_convolutional_layer(filter_dimension=[3, 3, self.__n, 16], stride_length=1, activation_function='relu')
        self.__decoder_net.add_upsampling_layer(filter_size=3, num_filters=16, upscale_factor=2, activation_function='relu')
        if self.__use_batchnorm:
            self.__decoder_net.add_batchnorm_layer()

        self.__decoder_net.add_convolutional_layer(filter_dimension=[3, 3, 16, 32], stride_length=1, activation_function='relu')
        self.__decoder_net.add_convolutional_layer(filter_dimension=[3, 3, 32, 32], stride_length=1, activation_function='relu')
        self.__decoder_net.add_convolutional_layer(filter_dimension=[3, 3, 32, 32], stride_length=1, activation_function='relu')
        self.__decoder_net.add_upsampling_layer(filter_size=3, num_filters=32, upscale_factor=2, activation_function='relu')
        if self.__use_batchnorm:
            self.__decoder_net.add_batchnorm_layer()

        self.__decoder_net.add_convolutional_layer(filter_dimension=[3, 3, 32, 32], stride_length=1, activation_function='relu')
        self.__decoder_net.add_convolutional_layer(filter_dimension=[3, 3, 32, 32], stride_length=1, activation_function='relu')
        self.__decoder_net.add_convolutional_layer(filter_dimension=[3, 3, 32, 32], stride_length=1, activation_function='relu')
        self.__decoder_net.add_upsampling_layer(filter_size=3, num_filters=32, upscale_factor=2, activation_function='relu')
        if self.__use_batchnorm:
            self.__decoder_net.add_batchnorm_layer()

        self.__decoder_net.add_convolutional_layer(filter_dimension=[3, 3, 32, 64], stride_length=1, activation_function='relu')
        self.__decoder_net.add_convolutional_layer(filter_dimension=[3, 3, 64, 64], stride_length=1,  activation_function='relu')
        self.__decoder_net.add_convolutional_layer(filter_dimension=[3, 3, 64, 64], stride_length=1, activation_function='relu')
        self.__decoder_net.add_upsampling_layer(filter_size=3, num_filters=64, upscale_factor=2, activation_function='relu')
        if self.__use_batchnorm:
            self.__decoder_net.add_batchnorm_layer()

        self.__decoder_net.add_convolutional_layer(filter_dimension=[3, 3, 64, 64], stride_length=1, activation_function='relu')
        self.__decoder_net.add_convolutional_layer(filter_dimension=[3, 3, 64, 64], stride_length=1, activation_function='relu')
        self.__decoder_net.add_convolutional_layer(filter_dimension=[3, 3, 64, 64], stride_length=1, activation_function='relu')
        self.__decoder_net.add_upsampling_layer(filter_size=3, num_filters=64, upscale_factor=2, activation_function='relu')
        if self.__use_batchnorm:
            self.__decoder_net.add_batchnorm_layer()

        self.__decoder_net.add_convolutional_layer(filter_dimension=[3, 3, 64, 32], stride_length=1, activation_function='relu')
        self.__decoder_net.add_convolutional_layer(filter_dimension=[3, 3, 32, 32], stride_length=1, activation_function='relu')
        self.__decoder_net.add_convolutional_layer(filter_dimension=[3, 3, 32, 16], stride_length=1, activation_function='relu')
        self.__decoder_net.add_upsampling_layer(filter_size=3, num_filters=16, upscale_factor=2, activation_function='relu')
        if self.__use_batchnorm:
            self.__decoder_net.add_batchnorm_layer()

        self.__decoder_net.add_convolutional_layer(filter_dimension=[3, 3, 16, 16], stride_length=1, activation_function='relu')
        self.__decoder_net.add_upsampling_layer(filter_size=3, num_filters=16, upscale_factor=2, activation_function='relu')
        if self.__use_batchnorm:
            self.__decoder_net.add_batchnorm_layer()

        self.__decoder_net.add_convolutional_layer(filter_dimension=[3, 3, 16, 16], stride_length=1, activation_function='relu')
        self.__decoder_net.add_upsampling_layer(filter_size=3, num_filters=16, upscale_factor=2, activation_function='relu')
        if self.__use_batchnorm:
            self.__decoder_net.add_batchnorm_layer()

        self.__decoder_net.add_convolutional_layer(filter_dimension=[1, 1, 16, self.__image_depth], stride_length=1, activation_function='tanh')

    def __find_canonical_transformation(self, processed_images):
        """Find a linear transformation between the test points projected in
        canonical space and the test points projected in the current projection space"""
        with self.__graph.as_default():
            can_points_can_space = self.__can_projections
            all_features = tf.concat(processed_images, axis=1)

            feature_length = (self.feature_extractor.get_output_size() * len(processed_images))

            can_points_current_space = self.__with_all_datapoints(all_features, [1, feature_length], num_records=self.__num_can_points)

            # Find a linear transformation between the can points in can space and can points in current space
            A = can_points_current_space.reshape((-1, self.__n))
            B = can_points_can_space.reshape((-1, self.__n))

            if self.__transformation_method == 'Linear':
                lin = LinearRegression()
                lin.fit(A, B)

                trans_score = lin.score(A, B)
                trans_succeeded = (trans_score > self.__trans_score_threshold)

                return trans_succeeded, trans_score, lin
            elif self.__transformation_method == 'NeuralNet':
                train_loss = self.__learn_subspace_transform(A, B)
                trans_succeeded = (train_loss < self.__trans_score_threshold)
                return trans_succeeded, train_loss

    def __build_geodesic_graph(self):
        self.__geodesic_path_lengths = []
        self.__geodesic_optimizers = []
        self.__geodesic_interpolated_points = []
        self.__geodesic_objectives = []
        self.__geodesic_distance_totals = []

        self.__geodesic_placeholder_A = []
        self.__geodesic_placeholder_B = []
        self.__geodesic_anchor_points = []

        # Calculate how many interpolated vertices we will need
        if self.__mode == 'longitudinal':
            self.__geodesic_num_interpolations = int((self.__target_vertices - self.__num_timepoints) / (self.__num_timepoints -1))
            total_vertices = self.__num_timepoints + (self.__geodesic_num_interpolations * (self.__num_timepoints - 1))
        elif self.__mode == 'cross sectional':
            self.__geodesic_num_interpolations = self.__target_vertices - 2
            total_vertices = self.__target_vertices

        # Assemble a graph
        for d in range(self.__num_gpus):
            with tf.device('/device:GPU:{0}'.format(d)):
                with tf.name_scope('gpu_%d_' % (d)) as scope:
                    def decoded_L2_distance(embedding_A, embedding_B):
                        return tf.norm(tf.subtract((self.__decoder_net.forward_pass(embedding_A) + 1.) / 2.,
                                                   (self.__decoder_net.forward_pass(embedding_B) + 1.) / 2.))

                    # Make static placeholders for the start, end, and all anchors in between
                    start_point = tf.placeholder(tf.float32, shape=(self.__n))
                    end_point = tf.placeholder(tf.float32, shape=(self.__n))

                    self.__geodesic_placeholder_A.append(start_point)
                    self.__geodesic_placeholder_B.append(end_point)

                    if self.__mode == 'longitudinal':
                        anchor_points = [tf.placeholder(tf.float32, shape=(self.__n)) for i in range(self.__num_timepoints - 2)]
                        self.__geodesic_anchor_points.append(anchor_points)

                        if self.__geodesic_num_interpolations > 0:
                            interpolated_points = [tf.Variable(tf.zeros([self.__n]), name='intermediate-%d' % x) for x in range(self.__geodesic_num_interpolations * (self.__num_timepoints - 1))]
                    elif self.__mode == 'cross sectional':
                        interpolated_points = [tf.Variable(tf.zeros([self.__n]), name='intermediate-%d' % x) for x in range(self.__geodesic_num_interpolations)]

                    if self.__geodesic_num_interpolations > 0:
                        self.__geodesic_interpolated_points.append(interpolated_points)

                    # Build the list of distances (losses) between points
                    previous_node = [start_point]
                    next_node = [None]
                    next_anchor = 0
                    next_interpolated = 0

                    intermediate_distances = []

                    for i in range(1, total_vertices):
                        if i == total_vertices - 1:
                            # Distance to end point
                            next_node[0] = end_point
                        elif i % (self.__geodesic_num_interpolations  + 1) == 0:
                            # Distance to an anchor point
                            next_node[0] = anchor_points[next_anchor]
                            next_anchor = next_anchor + 1
                        else:
                            # Distance to an interpolated point
                            next_node[0] = interpolated_points[next_interpolated]
                            next_interpolated = next_interpolated + 1

                        intermediate_distances.append(decoded_L2_distance(previous_node[0], next_node[0]))

                        previous_node[0] = next_node[0]

                    total_path_length = tf.reduce_sum(intermediate_distances)
                    self.__geodesic_path_lengths.append(total_path_length)

                    if self.__geodesic_num_interpolations > 0:
                        ms_dist = tf.reduce_mean(tf.square(intermediate_distances))

                        gradients, optimizer = self.__get_clipped_gradients(ms_dist, interpolated_points)
                        self.__geodesic_optimizers.append(optimizer)
                        self.__geodesic_objectives.append(self.__apply_gradients(gradients))

        if self.__geodesic_num_interpolations > 0:
            # Collect all of the optimizer variables we have to re-inititalize
            optimizer_vars = []
            intermediate_vars = []

            for x in self.__geodesic_optimizers:
                optimizer_vars.extend(x.variables())

            for x in self.__geodesic_interpolated_points:
                intermediate_vars.extend(x)

            self.__geodesic_init_ops = [tf.variables_initializer(var_list=intermediate_vars),
                                        tf.variables_initializer(var_list=optimizer_vars)]

            # Graph ops for generating the interpolation image sequence by decoding the intermediate points
            self.__geodesic_decoded_intermediate = [[(self.__decoder_net.forward_pass(x) + 1.) / 2. for x in d] for d in self.__geodesic_interpolated_points]


    def __geodesic_distance(self, series, t):
        '''Gets the geodesic distance for a series of points, can do n pairs in parallel where n is the number of gpus'''

        series = np.array(series)

        starts = series[:, 0, :]
        ends = series[:, -1, :]

        # Build the point placeholders for start and end points
        fd = {}

        if self.__mode == 'longitudinal':
            anchors = series[:, 1:-1, :]

            # Build the point placeholders for anchor points
            for (x, y) in zip(self.__geodesic_anchor_points, anchors):
                for a, b in zip(x, y):
                    fd[a] = b

        for (x, y) in zip(self.__geodesic_placeholder_A, starts):
            fd[x] = y

        for (x, y) in zip(self.__geodesic_placeholder_B, ends):
            fd[x] = y

        if self.__geodesic_num_interpolations > 0:
            self.__session.run(self.__geodesic_init_ops, feed_dict=fd)

            # Assign the interpolated points to a linear interpolation
            def get_midpoints(point_A, point_B):
                eps = (point_B - point_A) / self.__geodesic_num_interpolations
                return [np.add(point_A, eps * x) for x in range(1, self.__geodesic_num_interpolations + 1)]

            midpoints = []

            for d in range(self.__num_gpus):
                mps = []

                if self.__mode == 'longitudinal':
                    mps.extend(get_midpoints(starts[d], anchors[d][0]))

                    for i in range(len(anchors[d]) - 1):
                        mps.extend(get_midpoints(anchors[d][i], anchors[d][i + 1]))

                    mps.extend(get_midpoints(anchors[d][-1], ends[d]))

                    midpoints.append(mps)
                elif self.__mode == 'cross sectional':
                    mps.extend(get_midpoints(starts[d], ends[d]))

                    midpoints.append(mps)

            for d in range(0, self.__num_gpus):
                for j in range(0, len(midpoints[d])):
                       self.__geodesic_interpolated_points[d][j].load(midpoints[d][j], self.__session)

            # Train the parameters
            for k in range(self.__geodesic_path_iterations):
                _, current_distance = self.__session.run([self.__geodesic_objectives, self.__geodesic_path_lengths], feed_dict=fd)
                t.set_description('Path distance: {0}'.format(current_distance))
                t.refresh()

        # Get final distance
        # QQ
        dists = self.__session.run(self.__geodesic_path_lengths, feed_dict=fd)
        # dists, points = self.__session.run([self.__geodesic_path_lengths, self.__geodesic_anchor_points], feed_dict=fd)
        #
        # for a, b, c in zip(starts, anchors, ends):
        #    combined = np.vstack([a, b, c])
        #
        #    # Plot path plot
        #    rand_int = str(random.randint(1, 10000))
        #    plotter.plot_path(os.path.join(self.results_path, 'path_plots'), rand_int, combined)
        #
        #    # Generate image sequence
        #    plotter.make_directory(os.path.join(self.results_path, 'interpolations'))
        #
        #    decoder_output = self.__session.run(self.__geodesic_decoded_intermediate[0])
        #
        #    for i, generated in enumerate(decoder_output):
        #        self.__save_as_image(np.squeeze(generated), os.path.join(self.results_path, 'interpolations', '{0}-{1}.png'.format(rand_int, i)))

        return dists

    def __get_geodesics_for_all_projections(self):
        def get_sequence_at_index(idx, projections):
            return [p[idx][2] for p in projections]

        ret = []

        if self.__mode == 'longitudinal':
            num_rows = len(self.__all_projections[0])
            all_idxs = range(num_rows)

            self.__log('Calculating geodesic distances...')

            t = trange(0, num_rows, self.__num_gpus)

            for i in t:
                num_padding = 0

                if i + self.__num_gpus > num_rows:
                    idxs = all_idxs[i:]
                    num_padding = len(all_idxs) % self.__num_gpus
                    idxs.extend(([all_idxs[0]] * num_padding))
                else:
                    idxs = all_idxs[i:i+self.__num_gpus]

                if idxs is None:
                    break

                series = []
                meta = []

                for idx in idxs:
                    series.append(get_sequence_at_index(idx, self.__all_projections))
                    meta.append(self.__all_projections[0][idx][:2])

                # These are dummies, we won't use the results
                for j in range(num_padding):
                    series.append(get_sequence_at_index(0, self.__all_projections))

                # Do the evaluation
                dists = self.__geodesic_distance(series, t)

                r = [[metadata[0], metadata[1], dist] for metadata, dist in zip(meta, dists)]

                ret.extend(r)
        elif self.__mode == 'cross sectional':
            def get_projections_with_treatment(tr):
                return [filter(lambda x: x[1] == tr, p) for p in self.__all_projections]

            def get_idx_for_accid(accid, projections):
                for p in range(len(projections[0])):
                    if projections[0][p][0] == accid:
                        return p

                return None

            treated_projections = get_projections_with_treatment(1)
            untreated_projections = get_projections_with_treatment(0)

            # Remove all treated with no corresponding untreated
            treated_projections = [filter(lambda x: get_idx_for_accid(x[0], untreated_projections) is not None, p) for p in treated_projections]

            num_rows = len(treated_projections[0])
            all_idxs = range(num_rows)

            self.__log('Calculating geodesic distances...')

            t = trange(0, num_rows, self.__num_gpus)

            for i in t:
                num_padding = 0

                if i + self.__num_gpus > num_rows:
                    idxs = all_idxs[i:]
                    num_padding = len(all_idxs) % self.__num_gpus
                    idxs.extend(([all_idxs[0]] * num_padding))
                else:
                    idxs = all_idxs[i:i + self.__num_gpus]

                if idxs is None:
                    break

                series = []
                meta = []

                for idx in idxs:
                    accid = treated_projections[0][idx][0]
                    treated_point = treated_projections[-1][idx][2]
                    untreated_idx = get_idx_for_accid(accid, untreated_projections)

                    untreated_point = untreated_projections[-1][untreated_idx][2]

                    series.append([treated_point, untreated_point])
                    meta.append(treated_projections[0][idx][:2])

                # These are dummies, we won't use the results
                for j in range(num_padding):
                    series.append(series[-1])

                # Do the evaluation
                dists = self.__geodesic_distance(series, t)

                r = [[metadata[0], metadata[1], dist] for metadata, dist in zip(meta, dists)]

                ret.extend(r)

        return ret

    def start(self, pretraining_batches=100, report_rate=80, name='results', tensorboard=None, ordination_vis=False, num_gpus=1, num_threads=1, saliency_target=None, decoder_vis=False):
        """Begins training"""

        self.results_path = './' + os.path.basename(name) + '-results'
        self.__num_folds = len(self.__record_files)

        self.__pretraining_batches = pretraining_batches
        self.__report_rate = report_rate
        self.__num_gpus = num_gpus
        self.__num_threads = num_threads

        self.__make_directory(self.results_path)

        for i in range(self.__num_timepoints):
            self.__all_projections.append([])

        self.__log('Using {0} GPUs and {1} CPU threads'.format(self.__num_gpus, self.__num_threads))
        self.__log('Results will be saved into {0}'.format(self.results_path))

        # Folds control
        for current_fold in range(self.__num_folds):
            self.__current_fold = current_fold
            self.__log('Doing fold {0}'.format(current_fold))

            can_fold_file = self.__record_files[0]
            self.__current_test_file = self.__record_files[current_fold]
            self.__current_train_files = [f for f in self.__record_files if f != self.__current_test_file]

            if tensorboard is not None:
                self.__tb_file = os.path.join(tensorboard, '{0}_fold{1}'.format(name, self.__current_fold))
            else:
                self.__tb_file = None

            # Failure loop
            for current_attempt in range(self.__num_fold_restarts):
                self.__log('This is attempt {0}'.format(current_attempt))

                self.__reset_graph()

                # If this isn't the first try, delete the old tensorflow accumulator
                if tensorboard is not None and current_attempt > 0:
                    shutil.rmtree(self.__tb_file)

                # Gotta add all the shit to the graph again
                with self.__graph.as_default():
                    self.__initialize_data(can_fold_file)

                    with tf.variable_scope('pretraining'):
                        # Build the CNN for feature extraction
                        self.feature_extractor = cnn.cnn(debug=self.__debug, batch_size=self.__batch_size)

                        if self.__do_crop:
                            self.feature_extractor.set_image_dimensions(int(self.__image_height * self.__crop_amount), int(self.__image_width  * self.__crop_amount), self.__image_depth)
                        else:
                            self.feature_extractor.set_image_dimensions(self.__image_height, self.__image_width, self.__image_depth)

                        self.__build_convnet()
                        self.feature_extractor.send_ops_to_graph(self.__graph)

                        # Build the LSTM
                        self.lstm = lstm.lstm(self.__batch_size, self.__lstm_units, self.__graph)

                    with tf.variable_scope('transformer'):
                        # Build the subspace transformer network
                        self.__transformer_net = cnn.cnn(debug=self.__debug, batch_size=self.__batch_size)
                        self.__transformer_net.set_image_dimensions(1, 1, self.__n)

                        self.__build_transformer()
                        self.__transformer_net.send_ops_to_graph(self.__graph)

                    with tf.variable_scope('decoder'):
                        self.__decoder_net = cnn.cnn(debug=True, batch_size=self.__batch_size, name_prefix="decoder-")
                        self.__decoder_net.set_image_dimensions(1, 1, self.__n)

                        self.__build_decoder()

                        self.__decoder_net.send_ops_to_graph(self.__graph)

                    all_pretrain_gradients = []
                    all_reconstruction_gradients = []

                    for d in range(num_gpus):
                        with tf.device('/device:GPU:{0}'.format(d)):
                            with tf.name_scope('gpu_%d_' % (d)) as scope:
                                # --- Components for training ---

                                # Graph inputs
                                batch_data = self.__input_batch_train

                                id, treatment, image_data = self.__parse_batch(batch_data)

                                # Graph components for main objective
                                embeddings = []

                                resized_height = int(self.__image_height * self.__crop_amount)
                                resized_width = int(self.__image_width * self.__crop_amount)

                                # Embed the images
                                for image in image_data:
                                    image = self.__apply_image_standardization(image)
                                    image = self.__apply_augmentations(image, resized_height, resized_width)

                                    emb = self.feature_extractor.forward_pass(image)
                                    embeddings.append(emb)

                                unaugmented_embeddings = []

                                for image in image_data:
                                    if self.__do_crop:
                                        image = self.__resize_image(image)

                                    image = self.__apply_image_standardization(image)

                                    unaugmented_embeddings.append(self.feature_extractor.forward_pass(image))

                                all_emb = tf.concat(embeddings, 0)
                                avg = tf.reduce_mean(all_emb, axis=0)
                                emb_centered = all_emb - avg
                                cov = tf.matmul(tf.transpose(emb_centered), emb_centered) / (self.__batch_size * self.__num_timepoints)

                                # Add a small epsilon to the diagonal to make sure it's invertible
                                cov = tf.linalg.set_diag(cov, (tf.linalg.diag_part(cov) + self.__variance_constant))

                                # Determinant of the covariance matrix
                                emb_cost = tf.linalg.det(cov)

                                predicted_treatment, _ = self.lstm.forward_pass(embeddings)

                                treatment_loss = self.__get_treatment_loss(treatment, predicted_treatment)

                                cnn_reg_loss = self.feature_extractor.get_regularization_loss()
                                lstm_reg_loss = self.lstm.get_regularization_loss()

                                # Decoder takes the output from the latent space encoder and tries to reconstruct the input
                                reconstructions = tf.concat([(self.__decoder_net.forward_pass(emb) + 1.) / 2. for emb in unaugmented_embeddings], axis=0)

                                decoder_out = self.__decoder_net.layers[-1].output_size

                                original_images = tf.image.resize_images(tf.concat(image_data, axis=0), [decoder_out[1], decoder_out[2]])

                                # A measure of how diverse the reconstructions are
                                _, rec_var = tf.nn.moments(reconstructions, axes=[0])
                                reconstruction_diversity = tf.reduce_mean(rec_var)

                                reconstruction_losses = tf.reduce_mean(tf.square(tf.subtract(original_images, reconstructions)), axis=[1, 2, 3])
                                reconstruction_loss, reconstruction_var = tf.nn.moments(reconstruction_losses, axes=[0])

                                reconstruction_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'decoder')

                                reconstruction_gradients, _ = self.__get_clipped_gradients(reconstruction_loss, reconstruction_vars)

                                all_reconstruction_gradients.append(reconstruction_gradients)

                                pretrain_total_loss = tf.reduce_sum([treatment_loss, cnn_reg_loss, emb_cost])

                                pt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'pretraining')

                                pretrain_gradients, _ = self.__get_clipped_gradients(pretrain_total_loss, pt_vars)
                                all_pretrain_gradients.append(pretrain_gradients)

                    # Average gradients and apply
                    if num_gpus == 1:
                        average_pretrain_gradients = all_pretrain_gradients[0]
                        average_reconstruction_gradients = all_reconstruction_gradients[0]
                    else:
                        average_pretrain_gradients = self.__average_gradients(all_pretrain_gradients)
                        average_reconstruction_gradients = self.__average_gradients(all_reconstruction_gradients)

                    pretrain_objective = self.__apply_gradients(average_pretrain_gradients)

                    reconstruction_objective = self.__apply_gradients(average_reconstruction_gradients)

                    # Ops for subspace transformation
                    self.__transformer_batch_pl = tf.placeholder(tf.float32, shape=(self.__batch_size, self.__n))
                    self.__transformer_labels_pl = tf.placeholder(tf.float32, shape=(self.__batch_size, self.__n))

                    pred = self.__transformer_net.forward_pass(self.__transformer_batch_pl)
                    self.__transformer_loss = tf.reduce_mean(tf.square(pred - self.__transformer_labels_pl))
                    self.__transformer_obj, _ = self.__minimize_with_clipped_gradients(self.__transformer_loss, vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='transformer'))

                    # Add ops for geodesic calculations
                    self.__build_geodesic_graph()

                    # --- Components for testing ---

                    # Test inorder
                    batch_data_ti = self.__inorder_input_batch_test
                    id_ti, treatment_ti, image_data_ti = self.__parse_batch(batch_data_ti)

                    processed_images_ti = []

                    for image in image_data_ti:
                        if self.__do_crop:
                            image = self.__resize_image(image)

                        image = self.__apply_image_standardization(image)

                        processed_images_ti.append(self.feature_extractor.forward_pass(image, deterministic=True))

                    # Canonical fold
                    if self.__input_batch_canonical is not None:
                        batch_data_c = self.__input_batch_canonical
                        _, _, can_point_image_data = self.__parse_batch(batch_data_c)

                        processed_images_c = []

                        for image in can_point_image_data:
                            if self.__do_crop:
                                image = self.__resize_image(image)

                            image = self.__apply_image_standardization(image)

                            processed_images_c.append(self.feature_extractor.forward_pass(image))

                    # Test (embedding) set
                    batch_data_test = self.__input_batch_test

                    id_test, treatment_test, image_data_test = self.__parse_batch(batch_data_test)

                    processed_images_test = []

                    for image in image_data_test:
                        if self.__do_crop:
                            image = self.__resize_image(image)

                        image = self.__apply_image_standardization(image)

                        processed_images_test.append(self.feature_extractor.forward_pass(image, deterministic=True))

                    predicted_treatment_test, _ = self.lstm.forward_pass(processed_images_test)

                    treatment_loss_test = self.__get_treatment_loss(treatment_test, predicted_treatment_test)

                    # For saliency visualization
                    if saliency_target is not None:
                        saliency_image = tf.placeholder(tf.float32, shape=(None, self.__image_height, self.__image_width, self.__image_depth))

                        if self.__do_crop:
                            saliency_image_resized = self.__resize_image(saliency_image)
                        else:
                            saliency_image_resized = saliency_image

                        saliency_result = self.feature_extractor.forward_pass(saliency_image_resized)

                    decoder_test_vec = [(self.__decoder_net.forward_pass(p) + 1.) / 2. for p in processed_images_test]

                    # Aggregate tensorboard summaries
                    if tensorboard is not None:
                        self.__log('Creating Tensorboard summaries...')

                        tf.summary.scalar('pretrain/treatment_loss', treatment_loss, collections=['pretrain_summaries'])
                        tf.summary.scalar('pretrain/cnn_reg_loss', cnn_reg_loss, collections=['pretrain_summaries'])
                        tf.summary.scalar('pretrain/lstm_reg_loss', lstm_reg_loss, collections=['pretrain_summaries'])
                        tf.summary.scalar('pretrain/emb_cost', emb_cost, collections=['pretrain_summaries'])
                        tf.summary.histogram('pretrain/predicted_treatment', predicted_treatment, collections=['pretrain_summaries'])
                        [tf.summary.histogram('gradients/%s-gradient' % g[1].name, g[0], collections=['pretrain_summaries']) for g in average_pretrain_gradients]

                        tf.summary.scalar('test/treatment_loss', treatment_loss_test, collections=['pretrain_summaries'])

                        tf.summary.scalar('decoder/reconstruction_loss_batch_mean', reconstruction_loss, collections=['decoder_summaries'])
                        tf.summary.scalar('decoder/reconstruction_loss_batch_var', reconstruction_var, collections=['decoder_summaries'])
                        tf.summary.scalar('decoder/reconstruction_diversity', reconstruction_diversity, collections=['decoder_summaries'])
                        tf.summary.image('decoder/reconstructions', reconstructions, collections=['decoder_summaries'])

                        # Filter visualizations
                        filter_summary = self.__get_weights_as_image(self.feature_extractor.first_layer().weights)
                        tf.summary.image('filters/first', filter_summary, collections=['pretrain_summaries'])

                        # Summaries for each layer
                        for layer in self.feature_extractor.layers:
                            if isinstance(layer, layers.fullyConnectedLayer) or isinstance(layer, layers.convLayer):
                                tf.summary.histogram('weights/' + layer.name, layer.weights, collections=['pretrain_summaries'])
                                tf.summary.histogram('biases/' + layer.name, layer.biases, collections=['pretrain_summaries'])
                                tf.summary.histogram('activations/' + layer.name, layer.activations, collections=['pretrain_summaries'])

                        self.__pretraining_summaries = tf.summary.merge_all(key='pretrain_summaries')
                        self.__decoder_summaries = tf.summary.merge_all(key='decoder_summaries')
                        self.__tb_writer = tf.summary.FileWriter(self.__tb_file)

                    # Initialize network and threads
                    self.__initialize()

                    shortcut = False

                    if shortcut:
                        self.__log('DEBUGGING: Taking post-training shortcut...')

                        pretrain_succeeded = True
                        self.__current_fold = self.__num_folds - 1

                        self.__all_projections = load('all_projections.pkl')

                        self.load_state()
                    else:
                        pretrain_succeeded = self.__pretrain(pretrain_objective, treatment_loss, treatment_loss_test)

                        self.__log('Pretraining finished.')

                        if pretrain_succeeded:
                            # Save all the embeddings from the test set, using the canonical transform if this is not the first fold
                            if self.__current_fold == 0:
                                self.__log('Saving projections for test set...')
                                self.__save_full_datapoints(id_ti, treatment_ti, processed_images_ti)
                            else:
                                self.__log('Learning a transformation from this learned space to the canonical space...')

                                if self.__transformation_method == 'Linear':
                                    trans_succeeded, trans_score, canon_model = self.__find_canonical_transformation(processed_images_c)
                                    self.__log('Transformation R^2 score: {0}'.format(trans_score))
                                elif self.__transformation_method == 'NeuralNet':
                                    trans_succeeded, trans_score = self.__find_canonical_transformation(processed_images_c)
                                    self.__log('Transformation loss: {0}'.format(trans_score))

                                if trans_succeeded:
                                    if self.__transformation_method == 'Linear':
                                        self.__reporter.add('Fold {0} transformation R^2: {1}'.format(self.__current_fold, trans_score), trans_succeeded)
                                        self.__log('Saving projections for test set...')
                                        self.__save_full_datapoints(id_ti, treatment_ti, processed_images_ti, canon_matrix=canon_model)
                                    elif self.__transformation_method == 'NeuralNet':
                                        self.__reporter.add('Fold {0} transformation loss: {1}'.format(self.__current_fold, trans_score), trans_succeeded)
                                        self.__log('Saving projections for test set...')
                                        self.__save_full_datapoints(id_ti, treatment_ti, processed_images_ti, canon_matrix=True)
                                else:
                                    # Even though the pretrain worked, the transform did not so retry.
                                    self.__log('Pretrain succeeded but transform did not.')
                                    pretrain_succeeded = False

                    if pretrain_succeeded:
                        if not shortcut:
                            self.__reporter.add('Pretraining fold %d converged' % self.__current_fold, True)

                            if self.__current_fold == 0:
                                # Train and test the decoder
                                self.__log('Training decoder...')
                                self.__train_decoder(reconstruction_objective, reconstruction_loss)

                                if decoder_vis:
                                    self.__log('Testing decoder...')
                                    self.__test_decoder(decoder_test_vec, image_data_test)

                                # Save the parameters of the decoder so we can load it later
                                self.__save_decoder()

                            # If we are done all the folds now
                            if (self.__current_fold + 1) == self.__num_folds:
                                # Ordination plots
                                if ordination_vis:
                                    self.__log('Saving ordination plots...')

                                    plotter.plot_general_ordination_plot(self.__all_projections,
                                                                         self.results_path + '/ordination-plots',
                                                                         self.__n)

                                # Saliency visualization
                                if saliency_target is not None:
                                    self.__log('Outputting saliency figure...')

                                    plotter.make_directory(os.path.join(self.results_path, 'saliency'))

                                    def LoadImage(file_path):
                                        im = Image.open(file_path)
                                        im = np.asarray(im)
                                        return im / 127.5 - 1.0

                                    activations = saliency_result
                                    y = tf.norm(activations)

                                    saliency_test_image = LoadImage(saliency_target)

                                    gbp = saliency.GuidedBackprop(self.__graph, self.__session, y, saliency_image)

                                    gbp_mask = gbp.GetSmoothedMask(saliency_test_image)

                                    smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(gbp_mask)

                                    self.__save_as_image(smoothgrad_mask_grayscale, os.path.join(self.results_path, 'saliency', 'saliency-fold{0}.png'.format(current_fold)))

                        if (self.__current_fold + 1) == self.__num_folds:
                            # Load the decoder back up
                            self.__load_decoder()

                            # Compute all geodesics
                            geo_pheno = self.__get_geodesics_for_all_projections()

                            # Write to disk in .pheno format
                            df = pd.DataFrame(geo_pheno)
                            df.columns = ['genotype', 'treatment', 'geodesic']
                            df.to_csv(os.path.join(self.results_path, name + '-geo.csv'), sep=' ', index=False)

                            self.__log('.pheno file saved.')

                            # Write a plot of output values
                            self.__log('Writing trait value plot...')

                            bins = np.linspace(np.amin(df['geodesic'].tolist()), np.amax(df['geodesic'].tolist()), 100)
                            treated = df.loc[df['treatment'] == 1, 'geodesic'].tolist()
                            untreated = df.loc[df['treatment'] == 0, 'geodesic'].tolist()

                            plt.clf()
                            plt.hist(treated, bins, alpha=0.5, label='treated')
                            plt.hist(untreated, bins, alpha=0.5, label='control')
                            plt.legend(loc='upper right')
                            plt.savefig(os.path.join(self.results_path, 'trait-histogram.png'))

                        self.__shutdown()
                        break
                    else:
                        self.__shutdown()
                        self.__reset_graph()

                        if current_attempt == self.__num_fold_restarts - 1:
                            self.__reporter.add('Pretraining fold %d did NOT converge' % self.__current_fold, False)
                            self.__log('Pretraining failed the maximum number of times.')

                            if self.__current_fold == 0:
                                self.__log('Could not embed the first fold, will terminate here.')
                                self.__shutdown()
                                exit()

                            break
                        else:
                            self.__log('Pretraining attempt did not succeed, will try again.')
                            self.__num_failed_attempts += 1
                            continue

        self.__log('Sanity checks:')
        self.__reporter.print_all()

        if self.__reporter.all_succeeded():
            self.__log(emoji.emojize('Everything looks like it succeeded! Check the output folder for results. :beer_mug:'))
        else:
            self.__log('One or more folds failed. The output was not written.')

        self.__log('Run statistics:')
        self.__log('Total time elapsed: {0}'.format(self.__global_timer.elapsed()))
        self.__log('Total failed pretrain attempts: {0}'.format(self.__num_failed_attempts))
        self.__log('Total population size ultimately used: {0}'.format(len(self.__all_projections[0])))

    def __pretrain(self, pretrain_op, loss_op, test_loss_op):
        self.__log('Starting embedding learning...')

        batch_loss = None
        samples_per_sec = 0.

        # Needed for batch norm
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        pretrain_op = tf.group([pretrain_op, update_ops])

        t = trange(self.__pretraining_batches)

        for i in t:
            if i % self.__report_rate == 0 and i > 0:
                if self.__tb_file is not None:
                    _, batch_loss, summary = self.__session.run([pretrain_op, loss_op, self.__pretraining_summaries])
                    self.__tb_writer.add_summary(summary, i)
                    self.__tb_writer.flush()
                else:
                    _, batch_loss = self.__session.run([pretrain_op, loss_op])

                t.set_description('EMBEDDING - Batch {}: Loss {:.3f}, samples/sec: {:.2f}'.format(i, batch_loss, samples_per_sec))
                t.refresh()
            else:
                start_time = time.time()
                self.__session.run([pretrain_op])
                elapsed = time.time() - start_time
                samples_per_sec = (self.__batch_size / elapsed) * self.__num_gpus

        # Use the mean loss from 4 test batches to determine training success
        test_loss = np.mean([self.__session.run(test_loss_op) for x in range(4)])

        self.__log('Test loss: {0}'.format(test_loss))

        return test_loss <= self.__pretrain_convergence_thresh_upper

    def __train_decoder(self, train_op, loss_op):
        samples_per_sec = 0.

        t = trange(self.__decoder_iterations)

        for i in t:
            if i % self.__report_rate == 0 and i > 0:
                if self.__tb_file is not None:
                    _, batch_loss, summary = self.__session.run([train_op, loss_op, self.__decoder_summaries])
                    self.__tb_writer.add_summary(summary, i)
                    self.__tb_writer.flush()
                else:
                    _, batch_loss = self.__session.run([train_op, loss_op])

                t.set_description(
                    'DECODER - Batch {}: Loss {:.3f}, samples/sec: {:.2f}'.format(i, batch_loss, samples_per_sec))
                t.refresh()
            else:
                start_time = time.time()
                self.__session.run([train_op])
                elapsed = time.time() - start_time
                samples_per_sec = (self.__batch_size / elapsed) * self.__num_gpus

    def __test_decoder(self, decoder_ops, test_image_ops):
        plotter.make_directory(os.path.join(self.results_path, 'decoder'))
        test_results = self.__session.run(test_image_ops + decoder_ops)
        test_images = test_results[:self.__num_timepoints]
        decoder_images = test_results[self.__num_timepoints:]

        for i, (test_image, decoder_output) in enumerate(list(zip(test_images, decoder_images))):
            for j in range(self.__batch_size):
                real = np.squeeze(test_image[j, :, :, :])
                self.__save_as_image(real, os.path.join(self.results_path, 'decoder', 'decoder-real-sample{0}-timestep{1}.png'.format(j, i)))

                generated = np.squeeze(decoder_output[j, :, :, :])
                self.__save_as_image(generated, os.path.join(self.results_path, 'decoder', 'decoder-generated-sample{0}-timestep{1}.png'.format(j, i)))

    def __get_weights_as_image(self, kernel, normalize=True):
        """Filter visualization, adapted with permission from https://gist.github.com/kukuruza/03731dc494603ceab0c5"""
        with self.__graph.as_default():
            pad = 1
            grid_X = 4
            grid_Y = (kernel.get_shape().as_list()[-1] / 4)
            num_channels = kernel.get_shape().as_list()[2]

            # pad X and Y
            x1 = tf.pad(kernel, tf.constant([[pad, 0], [pad, 0], [0, 0], [0, 0]]))

            # X and Y dimensions, w.r.t. padding
            Y = kernel.get_shape()[0] + pad
            X = kernel.get_shape()[1] + pad

            # pack into image with proper dimensions for tf.image_summary
            x2 = tf.transpose(x1, (3, 0, 1, 2))
            x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, num_channels]))
            x4 = tf.transpose(x3, (0, 2, 1, 3))
            x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, num_channels]))
            x6 = tf.transpose(x5, (2, 1, 3, 0))
            x7 = tf.transpose(x6, (3, 0, 1, 2))

            if normalize:
                # scale to [0, 1]
                x_min = tf.reduce_min(x7)
                x_max = tf.reduce_max(x7)
                x8 = (x7 - x_min) / (x_max - x_min)

        return x8
