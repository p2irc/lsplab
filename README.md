# LSP-Lab - Latent Space Phenotyping

This repo provides a complete multi-gpu implementation of Latent Space Phenotyping in Python.

See the pre-print on [biorxiv](https://www.biorxiv.org/content/10.1101/557678v1?rss=1) for details.

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

## How to Run and Interpret an Experiment

Running an LSP analysis requires a sequence of images taken at multiple timepoints for each individual in the population. See the next subsection on how to load data. There are two important things to keep in mind during data collection:

- There cannot be any incidental differences between the treatment and control imaging - for example, if the treated samples were taken with a different background or a different camera configuration.

- Ideally, the first timepoint should be at the onset of the treatment, after visual differences between accessions have already manifested. However, it is possible to perform the analysis from the start of the trial.

The image dataset is automatically split up into five evenly sized chunks, called *folds*. When you run it, this first thing you will see is it going through each of the five folds in sequence. For each one, it will learn an embedding of the images not in that fold. The first fold will look something like this.

```
02:25PM: (Fold 0) Starting embedding learning...
EMBEDDING - Batch 400: Loss 0.351, samples/sec: 45.39: 100%|███████████████████████| 1000/1000 [06:04<00:00,  1.37it/s]
02:31PM: (Fold 0) Test loss: 0.459445685148
```

Here we're trying to figure out what the effect of the treatment looks like in the images. The loss function is sigmoid cross-entropy, so if it can't figure out which samples are treated better than randomly guessing then you will see the loss settle at about 0.69 and the fold will fail. That's okay! It will repeat each fold up to ten times before giving up on that fold and excluding the individuals in it from the experiment. If it succeeds, it will check if what it learned on the out-of-fold images generalizes to the images in the fold. If it fails this generalization check, again, the fold will fail and re-start.

After embedding learning happens for the first fold (called fold 0), it trains a decoder as well. Unlike for the embedding learning phase, there is no way to sanity check this process automatically. For this reason we recommend the use of the `decoder_vis=True` parameter as above. When the decoder training is finished, it will spit out decoded pictures from the in-fold samples. These generated images may look strange but should generally look different from one another - for example, they shouldn't all be completely black images. This edge case may arise in datasets with solid color backgrounds if the decoder cannot understand the mapping from the latent space to the image space. In this case, abort the run and try it again.

After embedding learning for the second, third, fourth, and last folds, it will also try to learn a mapping from the new learned subspace into the subspace from the first fold using a linear model. As a sanity check, it will report R<sup>2</sup> for the points it fit to learn the regression. This is sanity check is *not* an indication of how effective the transformation is, only an indication of how well the non-linear transformation can be approximated with a linear transformation.

Following the final fold, if there are fewer than the selected number of timepoints in the dataset, then the path optimization step will occur. The path for one individual will be computed on each GPU. Following this step, a summary screen is displayed showing statistics for the run.

## Loading Data

There are currently three methods available for loading your data. These will create the index file (called a `.bgwas` file) necessary to compile the data into a usable format (`.tfrecords`).

### From A CSV File

`biotools.directory2bgwas` will read the a `.csv` file of sample labels, assuming that the file has the columns `genotype`, `treatment`, and `timestep` which are all integers.

#### Parameters

`timestamp_format` - (`string`, optional) specifies in datetime format how to save the timestamps.

### Returns

Nothing.

### From File Names

`biotools.filenames2bgwas` will read the .png files from a directory, assuming that the sample information is coded in the filename in the format `[genotype]_[treatment]_[timestep].png` where genotype, treatment, and timestep are all integers.

#### Parameters

`timestamp_format` - (`string`, optional) specifies in datetime format how to save the timestamps.

#### Returns

Nothing.

### From Lemnatec Scanalyzer

`biotools.snpshot2bgwas` will read the relevant metadata from a Lemnatec Scanalyzer `SnapshotInfo.csv` file.

#### Parameters

`barcode_format` - (`string`, optional) A regex string representing how genotype and treatment can be extracted from the barcode. Default is `'^([A-Za-z]+)+(\d+)(AA|AB)\d+$'`, meaning that genotype comes first and treatment is coded either `AA` or `AB`.

`timestamp_format` - (`string`, optional) specifies in datetime format how to save the timestamps.

`not_before` - (`datetime`, optional) Don't include any samples with a timestamp before this point.

`prefix` - (`string`, optional) Only include samples with filenames beginning with this string.

#### Returns

Number of timepoints inferred from metadata (`int`).

## Options

### Available parameters for `model.start`:

`pretraining_batches` - (`int`) The number of batches for doing embedding learning in each fold.

`report_rate` - (`int`) The number of batches to do before reporting performance in the terminal or in tensorboard.

`name` - (`string`) What to call the run.

`tensorboard` - (`string`) The path to a tensorboard log directory, if you want to plot tensorboard summaries.

`ordination_vis` - (`bool`) Set to `True` if you want to output an ordination plot.

`num_gpus` - (`int`) Number of GPUs to use. Note that Tensorflow will always allocate all memory on all GPUs unless you use the environment variable `CUDA_VISIBLE_DEVICES`.

`num_threads` - (`int`) Number of CPU threads to use. These are just for feeding the GPUs.

`saliency_target` - (`string`) The path to an image you want to plot a saliency visualization of. For visualization purposes only.

`decoder_vis` - (`bool`) Set to `True` if you want to visualize a test batch from the decoder. This is handy for sanity checking.


### Advanced Options

These are advanced settings you can change before calling `model.start` if you want. However, the defaults are usually okay for most datasets.

`model.set_loss_function(['sce', 'mse'])` - sets the loss function used for embedding learning to either sigmoid cross-entropy (`sce`, default) or mean squared error (`mse`).

`model.set_n(int)` - sets the dimensionality of the latent space (default is 16). Higher values are more likely to converge and decode properly, lower values are less likely to overfit.

`model.set_transformation_method(['Linear', 'NeuralNet'])` - use either a linear regression method (`Linear`, default) or a 2-layer neural network (`NeuralNet`) for subspace transformations. Linear methods are less likely to overfit but may lose imprtant non-linearity in the data manifold.

`model.set_augmentation(bool)` - Use data augmentation for embedding learning and decoder training (default is `False`). Performs horizontal flipping and brightness/contrast skews.

`model.set_cropping_augmentation(bool)` - Use random cropping augmentation when augmentation is turned on (default is `False`).

`model.set_image_standardization(bool)` - Standardize images during embedding learning (default is `True`).

`model.set_num_path_vertices(int)` - Set the number of path vertices for interpolation (default is `30`).

`model.set_num_decoder_iterations(int)` - Set the number of batches to run for training the decoder (default is `12000`).

`model.set_num_path_iterations(int)` - Set the number of batches to run for training the decoder (default is `400`).

`model.set_use_memory_cache(bool)` - Set whether the data is cached on disk or in RAM (default is `False`). Using `True` is faster, but may fill system memory and cause a crash if the dataset is too large.
