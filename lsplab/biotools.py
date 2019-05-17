import pandas as pd
import numpy as np
import tensorflow as tf
import datetime
from PIL import Image
from tqdm import tqdm

import os
import re
import gc
import random


num_encodings = 3
queue_capacity = 100
min_queue_size = 16
angle_regex = '^VIS_SV_(\d+)_'
# Assuming AA = unstressed, AB = stressed
treatments = {'AA': '0', 'AB': '1'}


def __index_to_label__(index, gen_trans, chr_pos):
    chromosome = int(gen_trans.loci[index].name[chr_pos])
    bp_pos = gen_trans.loci[index].bp_position
    label = 'Chr{0}_{1}'.format(chromosome, bp_pos)

    return label


def bgwas2tfrecords(index_file, images_directory, output_dir, tfrecord_filename, multi_angle=False, num_folds=5):
    """Convert a .bgwas index file and its accompanying image directory into .tfrecords (split into five folds)"""
    output_path = os.path.join(output_dir, tfrecord_filename)
    key_output_path = os.path.join(output_dir, 'key.csv')

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    print "Reading .bgwas index file..."

    # Read metadata about samples from .bgwas file
    labels = pd.read_csv(index_file, header=None).values

    print "Assembling data..."

    # Create a list of all images in the images directory
    all_image_filenames = []

    for root, dirnames, filenames in os.walk(images_directory):
        for filename in [filename for filename in filenames if filename.endswith('.png')]:
            all_image_filenames.append(os.path.join(root, filename))

    # Create sequences of images to write as records
    print "Finding image sequences..."

    # First sort labels into a dictionary of days present in the dataset
    days_dict = {}

    for label in labels:
        #day = datetime.datetime.strptime(label[2], timestamp_format).date()
        day = label[2]

        if day in days_dict:
            days_dict[day].append(label)
        else:
            days_dict[day] = [label]

    print "Number of timepoints: {0}...".format(len(days_dict))
    print "Sorting images into sequences..."

    # Okay now step through the items in the first day
    all_records = []
    first_day = min(days_dict.keys())
    primaries = days_dict[first_day]

    def search_for_filename(search_filename):
        for img_path in all_image_filenames:
            if os.path.basename(img_path) == search_filename:
                return img_path

        return None

    for label in tqdm(primaries):
        IID = label[1]
        treatment = label[3]
        image_filename = label[4]

        if multi_angle:
            image_angle = re.findall(angle_regex, image_filename)

            if not image_angle:
                # This could be a top-view image, or can't find the angle from the filename.
                continue

            if isinstance(image_angle, list):
                image_angle = image_angle[0]

        all_images = []

        # Step through the days, looking for corresponding images through to the end date.
        for key in sorted(days_dict.keys()):
            records = days_dict[key]
            found = False

            for record in records:
                r_IID = record[1]
                r_treatment = record[3]
                r_fn = record[4]

                if multi_angle:
                    if ('_'+image_angle+'_' in r_fn) and (r_IID == IID) and (r_treatment == treatment):
                        # Got a hit, add it to the list
                        found = True
                        r_file_path = search_for_filename(r_fn)
                        all_images.append(r_file_path)
                        break
                else:
                    if (r_IID == IID) and (r_treatment == treatment):
                        # Got a hit, add it to the list
                        found = True
                        r_file_path = search_for_filename(r_fn)
                        all_images.append(r_file_path)
                        break

            # By default, just add a None value which we will detect later.
            if not found:
                all_images.append(None)

        if not isinstance(label, list):
            label = label.tolist()

        label.append(all_images)
        all_records.append(label)

    # Check for errors in all records:
    clean_records = []

    for record in all_records:
        ok = True

        for i, image_path in enumerate(record[5]):
            if image_path is None:
                print "No image at a timepoint {0} for record (ID {1})".format(i, record[1])
                ok = False
                break

        if ok:
            clean_records.append(record)

    num_samples = len(clean_records)

    all_IIDs = list(set([label[1] for label in clean_records]))

    # Shuffle IIDs to get a random assignment to folds
    np.random.shuffle(all_IIDs)

    fold_IIDs = {i:[] for i in range(num_folds)}

    for i, current_IID in enumerate(all_IIDs):
        fold_IIDs[i % num_folds].append(current_IID)

    fold_writers = [tf.python_io.TFRecordWriter('{0}_{1}'.format(output_path, i)) for i in range(num_folds)]

    print "Number of records: {0}".format(num_samples)
    print "Total removed due to missing images/genotypes: {0}".format(len(all_records)-num_samples)

    # Write tfrecords
    print "Writing tfrecord files in {0}...".format(output_path)
    print "Please be patient, this could take a while..."

    _int_feature = lambda v: tf.train.Int64List(value=v)
    _byte_feature = lambda v: tf.train.BytesList(value=v)

    for i, label in tqdm(enumerate(clean_records)):
        IID = int(label[1])
        treatment = int(label[3])
        all_record_images = label[5]

        feature_dict = {
            'id': tf.train.Feature(int64_list=_int_feature([IID])),
            'treatment': tf.train.Feature(int64_list=_int_feature([treatment]))
        }

        # add all images to the feature list
        for j, image_path in enumerate(all_record_images):
            image_data = np.array(Image.open(image_path), dtype=np.uint8)
            image_raw = image_data.tostring()
            feature_dict['image_data_{0}'.format(j)] = tf.train.Feature(bytes_list=_byte_feature(image_raw))

        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

        serialized_example = example.SerializeToString()

        # Figure out which fold file this RIL goes in
        for key, value in fold_IIDs.iteritems():
            if IID in value:
                writer = fold_writers[key]
                writer.write(serialized_example)
                writer.flush()
                break

        # Try to manually free memory to fix memory leak?
        image_data = None
        image_raw = None
        feature_dict = None
        example = None
        serialized_example = None
        gc.collect()

    for writer in fold_writers:
        writer.close()

    print "Done"


def read_tfrecords_dataset(filename, image_height, image_width, image_depth, num_timepoints, num_threads, cached=True, in_memory=False):
    def parse_fn(example):
        features_dict = {
            'id': tf.FixedLenFeature((), tf.int64),
            'genotype': tf.VarLenFeature(tf.int64),
            'treatment': tf.FixedLenFeature((), tf.int64)
        }

        for i in range(num_timepoints):
            features_dict['image_data_{0}'.format(i)] = tf.VarLenFeature(tf.string)

        outputs = tf.parse_single_example(example, features=features_dict)

        genotype = tf.cast(tf.sparse_tensor_to_dense(outputs['genotype']), tf.float32)
        id = tf.cast(outputs['id'], tf.int32)
        treatment = tf.cast(outputs['treatment'], tf.int32)

        ret = {'id': id, 'treatment': treatment, 'genotype': genotype}

        for i in range(num_timepoints):
            image_name = 'image_data_{0}'.format(i)
            image = tf.decode_raw(tf.sparse_tensor_to_dense(outputs[image_name], default_value=''), tf.uint8)
            image = tf.reshape(image, [image_height, image_width, image_depth])
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)

            ret[image_name] = image

        return ret

    dataset = tf.data.TFRecordDataset([filename])
    dataset = dataset.map(map_func=parse_fn, num_parallel_calls=num_threads)

    if cached:
        if in_memory:
            dataset = dataset.cache()
            return dataset, None
        else:
            cache_file = 'lscache-' + str(random.randint(1,10000))
            dataset = dataset.cache(filename='/tmp/{0}'.format(cache_file))
            return dataset, cache_file

    return dataset, None


def get_sample_from_tfrecords_shuffled(filename, batch_size, image_height, image_width, image_depth, num_timepoints, queue_capacity, num_threads, cached=True, in_memory=False):
    """Returns a batch from the specified .tfrecords file"""

    dataset, cache_file_path = read_tfrecords_dataset(filename, image_height, image_width, image_depth, num_timepoints, num_threads, cached, in_memory)

    dataset_shuf = dataset.apply(tf.contrib.data.shuffle_and_repeat(queue_capacity)).batch(batch_size=batch_size).prefetch(buffer_size=batch_size)

    iterator_shuf = dataset_shuf.make_initializable_iterator()
    init_op_shuf = iterator_shuf.make_initializer(dataset_shuf)

    next_element_shuf = iterator_shuf.get_next()

    return next_element_shuf, init_op_shuf, cache_file_path


def get_sample_from_tfrecords_inorder(filename, batch_size, image_height, image_width, image_depth, num_timepoints, queue_capacity, num_threads, cached=True, in_memory=False):
    """Returns a batch from the specified .tfrecords file"""

    dataset, cache_file_path = read_tfrecords_dataset(filename, image_height, image_width, image_depth, num_timepoints, num_threads, cached, in_memory)

    dataset_inorder = dataset.repeat().batch(batch_size=batch_size).prefetch(buffer_size=batch_size)

    iterator_inorder = dataset_inorder.make_initializable_iterator()
    init_op_inorder = iterator_inorder.make_initializer(dataset_inorder)

    next_element_inord = iterator_inorder.get_next()

    return next_element_inord, init_op_inorder, cache_file_path


def write_qassoc_file(key_file, beta_values, se_values, t_values, p_values, num_samples, output_path):
    """Take a series of P values and a SNP key file and write a .qassoc file"""
    kf = pd.DataFrame.from_csv(key_file)
    keys = kf.as_matrix()

    header = ['CHR', 'SNP', 'BP', 'NMISS', 'BETA', 'SE', 'R2', 'T', 'P']
    output = []

    for k, beta, se, t, p in zip(keys, beta_values, se_values, t_values, p_values):
        k = k[0]
        chr = int(k[3])
        bp = int(k[5:])
        output.append([chr, k, bp, num_samples, beta, se, 0.0, t, p])

    of = pd.DataFrame(output)
    of.to_csv(output_path, sep=' ', header=header, index=False)


def csv2ped(input_filename, output_file):
    """Converts a csv file for the R/qtl package into plink text format (.ped and .map)"""
    variant_key = {'AA': 'T', 'BB': 'C', '-': '0'}
    output_delimiter = '\t'
    cM_ratio=1000000

    df_out = pd.DataFrame()
    df_map_out = pd.DataFrame()

    # Read input file
    df = pd.read_csv(input_filename, sep=',')

    chromes = df.iloc[[0]]
    chromes = chromes.drop(['id'], axis=1).T.as_matrix()

    distances = df.iloc[[1]]
    distances = distances.drop(['id'], axis=1).T.as_matrix()

    df = df[pd.notnull(df['id'])]

    # Loop through RILs
    for i, row in df.iterrows():
        # extract RIL name
        RIL_id = row.id[-3:]
        row.pop('id')
        print('Parsing individual %s...' % RIL_id)

        # Convert variants from CSV into SNPs
        SNPs = row.as_matrix()

        for key, value in variant_key.iteritems():
            SNPs = [entry.replace(key, value) for entry in SNPs]

        # Create a copy with two copies of each SNP
        duplicated_SNPs = []

        for SNP in SNPs:
            duplicated_SNPs.append(SNP)
            duplicated_SNPs.append(SNP)

        # Construct a PED row from what we have
        ped_preamble = ['0', RIL_id, '0', '0', '0', '0']
        ped_row = ped_preamble + duplicated_SNPs

        # Append row to output
        df_out = df_out.append(pd.DataFrame(ped_row).T)

    print('Writing ped file...')

    # Write output PED file
    df_out.to_csv(output_file+'.ped', sep=output_delimiter, header=None, index=False)

    print('Writing map file...')

    # Write output MAP file
    markers = df.keys().values[1:]

    for marker, chromosome, distance in zip(markers, chromes, distances):
        row = pd.DataFrame([chromosome[0], marker, 0, float(distance[0])*cM_ratio])
        df_map_out = df_map_out.append(row.T)

    df_map_out[[3]] = df_map_out[[3]].apply(pd.to_numeric)
    df_map_out.to_csv(output_file+'.map', sep=output_delimiter, header=None, index=False, float_format='%.0f')


def snapshot2bgwas(input_filename, output_filename, barcode_regex='^([A-Za-z]+)+(\d+)(AA|AB)\d+$', timestamp_format='%Y-%m-%d %H:%M:%S.%f', prefix='VIS', not_before=None, only_last=False):
    """Converts a Lemnatec SnapshotInfo.csv file into a .bgwas file."""
    df_out = pd.DataFrame()
    uid = 0

    df = pd.read_csv(input_filename, sep=',', delim_whitespace=False)

    bc_dict = {}

    for i, row in df.iterrows():
        if row['tiles'] is np.nan:
            print('No images for this row, continuing...')
            continue

        barcode = row['plant barcode']
        timestamp = datetime.datetime.strptime(row['timestamp'], timestamp_format)
        image_filenames = row['tiles']

        # Skip this entry if it is before the date range
        if not_before is not None and timestamp < not_before:
            print('Entry is before not_before cutoff, continuing...')
            continue

        if only_last:
            all_images = [image_filenames.split(';')[-1]]

            if all_images[0] == '':
                all_images = [image_filenames.split(';')[-2]]
        else:
            all_images = image_filenames.split(';')

        for image in all_images:
            if image and image.startswith(prefix):
                image = image + '.png'
                print('Processing entry for image %s...' % image)

                if barcode in bc_dict.keys():
                    bc_dict[barcode].append(image)
                else:
                    bc_dict[barcode] = [image]

    min_image_count = min([len(images) for (_, images) in list(bc_dict.items())])

    print('Truncating all image sequences to {0} timepoints.'.format(min_image_count))

    for (barcode, images) in bc_dict.iteritems():
        # Decode barcode
        matches = re.findall(barcode_regex, barcode)

        if not matches:
            print('Failed to parse barcode for this row, continuing...')
            continue

        if isinstance(matches, list):
            matches = matches[0]

        RIL = matches[1]
        treatment = treatments[matches[2]]

        images = images[:min_image_count]

        # Write a new row for each image
        for j, image in enumerate(images):
            #row = [uid, RIL, timestamp.strftime(timestamp_format), treatment, image]
            row = [uid, RIL, j, treatment, image]
            df_out = df_out.append(pd.DataFrame(row).T)
            uid += 1

    df_out.to_csv(output_filename, sep=',', header=None, index=False)

    return min_image_count


def directory2bgwas(input_filename, output_filename, num_timepoints, timestamp_format='%Y-%m-%d %H:%M:%S.%f'):
    """Converts a directory of images with a csv file of labels into a .bgwas file."""
    df_out = pd.DataFrame()
    uid = 0

    df = pd.read_csv(input_filename, sep=' ', delim_whitespace=False)

    # build a phony list of timestamps
    timestamps = []

    for i in range(num_timepoints):
        t = datetime.datetime.now() + datetime.timedelta(days=i)
        timestamps.append(t)

    for i, row in df.iterrows():
        IID = int(row['genotype'])
        timestamp = timestamps[int(row['timestep'])]
        treatment = int(row['treatment'])
        image = "{0}_{1}_{2}.png".format(IID, int(row['timestep']), treatment)

        row = [uid, IID, timestamp.strftime(timestamp_format), treatment, image]
        df_out = df_out.append(pd.DataFrame(row).T)
        uid += 1

    df_out.to_csv(output_filename, sep=',', header=None, index=False)


def filenames2bgwas(dir_path, output_filename, num_timepoints, timestamp_format='%Y-%m-%d %H:%M:%S.%f'):
    """Converts a directory of images with a csv file of labels into a .bgwas file."""
    df_out = pd.DataFrame()
    uid = 0

    all_image_filenames = []

    for root, dirnames, filenames in os.walk(dir_path):
        for filename in [filename for filename in filenames if filename.endswith('.png')]:
            all_image_filenames.append(filename)

    # build a phony list of timestamps
    timestamps = []

    for i in range(num_timepoints):
        t = datetime.datetime.now() + datetime.timedelta(days=i)
        timestamps.append(t)

    for filename in all_image_filenames:
        (IID, timestep, treatment) = filename.split('.')[0].split('_')

        IID = int(IID)
        timestamp = timestamps[int(timestep)]
        treatment = int(treatment)
        image = filename

        row = [uid, IID, timestamp.strftime(timestamp_format), treatment, image]
        df_out = df_out.append(pd.DataFrame(row).T)
        uid += 1

    df_out.to_csv(output_filename, sep=',', header=None, index=False)