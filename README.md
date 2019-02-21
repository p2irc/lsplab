# LSP-Lab - Latent Space Phenotyping

This repo provides a complete implementation of Latent Space Phenotyping in Python.

## Requirements and Installation

LSP-Lab requires at least one CUDA-compatible GPU. At least 6GB of VRAM and 16GB of system RAM are recommended.

Tensorflow 1.0 or higher is required. In order to install all the other required packages, run `pip install -r requirements.txt`.

## Example Usage

```
import lsplab as lab
import os
import datetime

# Paths and other info to change for your experiment.

# The location of the images
image_directory = './data/setaria/images'
# We're loading a dataset from a Lemnatec system so let's point to the metadata file.
snapshot_info_path = './data/setaria/images/SnapshotInfo.csv'

# These are where to save intermediate files we are going to generate.
index_output_path = './data/setaria/setaria.bgwas'
records_output_path = './data/setaria/records'
records_filename = 'Setaria-RIL'
records_full_path = os.path.join(records_output_path, records_filename)

# Height and width of images in the dataset. If they're huge, resize them first. Between 256x256 and 512x512 is good.
image_height = 411
image_width = 490

# Number of cpu threads to use.
num_threads = 20

# How many gpus to use for training.
num_gpus = 1

# This is usually a good batch size. Larger batch sizes may cause you to run out of GPU memory and cause a crash.
# Make sure you divide the batch size by the number of GPUs, as one batch will be run per GPU.
batch_size = 16

# What you want to name this run. The results will be saved in a folder called Setaria-results.
name = 'Setaria'

# Create the .bgwas index file from the snapshot file (if necessary). See below for options.
num_timepoints = lab.biotools.snapshot2bgwas(snapshot_info_path, index_output_path, not_before=datetime.datetime.strptime('2014-02-01', '%Y-%m-%d'), prefix='VIS_SV_90')

# Create the tfrecords from the dataset using the .bgwas index, and images folder (if necessary)
lab.biotools.bgwas2tfrecords(index_output_path, image_directory, records_output_path, records_filename)

# Make a new experiment. debug=True means that we will print out info to the console.
model = lab.lsp(debug=True, batch_size=batch_size)

# Load the records we just built.
model.load_records(records_output_path, image_height, image_width, num_timepoints)

# Start the model. See below for options.
model.start(pretraining_batches=1000, report_rate=100, name=name, decoder_vis=True, num_gpus=num_gpus, num_threads=num_threads)
```

## Loading Data

There are currently three methods available for loading your data. These will create the index file (called a `.bgwas` file) necessary to compile the data into a usable format (`.tfrecords`).

### From A CSV File

`biotools.directory2bgwas` will read the a `.csv` file of sample labels, assuming that the file has the columns `genotype`, `treatment`, and `timestep` which are all integers.

#### Parameters

`timestamp_format` - (string, optional) specifies in datetime format how to save the timestamps.

### From File Names

`biotools.filenames2bgwas` will read the .png files from a directory, assuming that the sample information is coded in the filename in the format `[genotype]_[treatment]_[timestep].png` where genotype, treatment, and timestep are all integers.

#### Parameters

`timestamp_format` - (string, optional) specifies in datetime format how to save the timestamps.

### From Lemnatec Scanalyzer

`biotools.snpshot2bgwas` will read the relevant metadata from a Lemnatec Scanalyzer `SnapshotInfo.csv` file.

#### Parameters

`barcode_format` - (string, optional) A regex string representing how genotype and treatment can be extracted from the barcode. Default is `'^([A-Za-z]+)+(\d+)(AA|AB)\d+$'`, meaning that genotype comes first and treatment is coded either `AA` or `AB`.

`timestamp_format` - (string, optional) specifies in datetime format how to save the timestamps.

`not_before` - (datetime, optional) Don't include any samples with a timestamp before this point.

`prefix` - (string, optional) Only include samples with filenames beginning with this string.

## Options

### Available parameters for `model.start`:

`pretraining_batches` - (int) The number of batches for doing embedding learning in each fold.

`report_rate` - (int) The number of batches to do before reporting performance in the terminal or in tensorboard.

`name` - (string) What to call the run.

`tensorboard` - (string) The path to a tensorboard log directory, if you want to plot tensorboard summaries.

`ordination_vis` - (bool) Set to `True` if you want to output an ordination plot.

`num_gpus` - (int) Number of GPUs to use. Note that Tensorflow will always allocate all memory on all GPUs unless you use the environment variable `CUDA_VISIBLE_DEVICES`.

`num_threads` - (int) Number of CPU threads to use. These are just for feeding the GPUs.

`saliency_target` - (string) The path to an image you want to plot a saliency visualization of.

`decoder_vis` - (bool) Set to `True` if you want to visualize a test batch from the decoder. This is handy for sanity checking.


### Advanced Options

These are advanced settings you can change before calling `model.start` if you want. However, the defaults are usually okay for most datasets.

`model.set_loss_function(['sce', 'mse'])` - sets the loss function used for embedding learning to either sigmoid cross-entropy (`sce`, default) or mean squared error (`mse`).

`model.set_n(8)` - sets the dimensionality of the latent space (default is 16). Higher values are more likely to converge and decode properly, lower values are less likely to overfit.

`model.set_transformation_method(['Linear', 'NeuralNet'])` - use either a linear regression method (`Linear`, default) or a 2-layer neural network (`NeuralNet`) for subspace transformations. Linear methods are less likely to overfit but may lose imprtant non-linearity in the data manifold.

`model.set_augmentation(True)` - Use data augmentation for embedding learning and decoder training (default is `False`). Performs horizontal flipping and brightness/contrast skews.

`model.set_cropping_augmentation(True)` - Use random cropping augmentation when augmentation is turned on (default is `False`).

`model.set_image_standardization(False)` - Don't standardize images during embedding learning (default is `True`).



