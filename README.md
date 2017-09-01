# Handwritten Korean Character Recognition with TensorFlow and Android

Hangul, the Korean alphabet, has 19 consonant and 21 vowel letters.
Combinations of these letters give a total of 11,172 possible Hangul
syllables/characters. However, only a small subset of these are typically used.

This journey will cover the creation process of an Android application that
will utilize a TensorFlow model trained to recognize Korean syllables.
In this application, users will be able to draw a Korean syllable on their
phone, and the application will attempt to infer what the character is by using
the trained model.

The following steps will be covered:
1. Generating image data using free Hangul-supported fonts found online and
   elastic distortion.
2. Converting images to TFRecords format to be used for input and training of
   the model.
3. Training and saving the model.
4. Using the saved model in a simple Android application.


## Prerequisites

Make sure you have the python requirements for this journey installed on you
system. From the root of the repository, run:

```
pip install -r requirements.txt
```


## Generating Image Data

In order to train a decent model, having copious amounts of data is necessary.
However, getting a large enough dataset of actual handwritten Korean characters
is challenging to find and cumbersome to create.

One way to deal with this data issue is to programmatically generate the data
yourself, taking advantage of the abundance of Korean font files found online.
So, that is exactly what we will be doing.

Provided in the tools directory of this repo is
[hangul-image-generator.py](./tools/hangul-image-generator.py).
This script will use fonts found in the fonts directory to create several images
for each character provided
in the given labels file. The default labels file is
[2350-common-hangul.txt](./labels/2350-common-hangul.txt)
which contains 2350 frequent characters derived from the
[KS X 1001 encoding](https://en.wikipedia.org/wiki/KS_X_1001).

The [fonts](./fonts) folder is currently empty, so before you can generate the
Hangul dataset, you must first download
several font files as described in the fonts directory [README](./fonts/README.md).
For my dataset, I used around 22 different font files, but more can always be
used to improve your dataset. Once your fonts directory is populated,
then you can proceed with the actual image generation:

```
./tools/hangul-image-generator.py
```

Optional flags for this are:

* `--label-file` for specifying a different label file (perhaps with less characters).
  Default is ./labels/2350-common-hangul.txt.
* `--font-dir` for specifying a different fonts directory. Default is _./fonts_.
* `--output-dir` for specifying the output directory to store generated images.
  Default is _./image-data_.

Depending on how many labels and fonts there are, this script may take a while
to complete. In order to bolster the dataset, random elastic distortions are also
performed on each generated character image. An example is shown below, with the
original character displayed first, followed by three elastic distortions.

![Normal Image](doc/source/images/hangul_normal.jpeg "Normal font character image")
![Distorted Image 1](doc/source/images/hangul_distorted1.jpeg "Distorted font character image")
![Distorted Image 2](doc/source/images/hangul_distorted2.jpeg "Distorted font character image")
![Distorted Image 3](doc/source/images/hangul_distorted3.jpeg "Distorted font character image")

Once the script is done, the output directory will contain a _test-images_ folder
and a _train-images_ folder. This is just a partition to separate training data
from testing data for when we train and test our model. Both of these image folders
will contain several directories corresponding to the labels, and each of these
label directories will contain several 64x64 JPEG images of the parent label.


## Converting Images to TFRecords

The TensorFlow standard input format is TFRecords, so in order to better feed in
data to a TensorFlow Model, let's first create several TFRecords files from our
images. Fortunately, there exists a
[script](https://github.com/tensorflow/models/blob/master/inception/inception/data/build_image_data.py)
that will do this for us in the tensorflow/models repository.

Download that script into the root of the project:

```
curl -O https://raw.githubusercontent.com/tensorflow/models/master/inception/inception/data/build_image_data.py
```

Create an output directory to store the TFRecords files:

```
mkdir ./tfrecords-output
```

Then run the script while specifying the _test-images_ and _train-images_
directories created earlier.

```
python build_image_data.py --train_directory=./image-data/train-images \
    --validation_directory=./image-data/test-images \
    --output_directory=./tfrecords-output \
    --labels_file=./labels/2350-common-hangul.txt --train_shards=6
```

Note: The value for `--train-shards` is the number of files to partition the training
data into. This can be increased or decreased depending on how much data you have.
It is used for splitting up the data so you don't just have one big file.

Once this script has completed, you should have sharded TFRecords files in the
output directory _./tfrecords-output_.

```
$ ls ./tfrecords-output
train-00000-of-00006    train-00003-of-00006    validation-00000-of-00002
train-00001-of-00006    train-00004-of-00006    validation-00001-of-00002
train-00002-of-00006    train-00005-of-00006
```

## Training the Model

Now that we have a lot of data, it is time to actually use it. In the root of
the project is [hangul-model.py](./hangul-model.py). This script will handle
creating an input pipeline for reading in TFRecords files and producing random
batches of images and labels. Next, a convolutional neural network (CNN) is
defined, and training is performed. After training, the model is exported
so that it can be used in our Android application.

The model here is similar to one described on the TensorFlow
[website](https://www.tensorflow.org/get_started/mnist/pros). A third
convolutional layer is added to extract more features to help classify for the
much greater number of classes.

To run the script, simply do the following from the root of the project:

```
python ./hangul_model.py
```

Optional flags for this script are:

* `--label-file` for specifying the labels that correspond to your training set.
  This is used by the script to determine the number of classes to classify for.
  Default is ./labels/2350-common-hangul.txt.
* `--tfrecords-dir` for specifying the directory containing the TFRecords shards.
  Default is _./tfrecords-output.
* `--output-dir` for specifying the output directory to store model checkpoints,
   graphs, and Protocol Buffer files. Default is _./saved-model_.

Note: In this script there is a NUM_TRAIN_STEPS variable defined at the top.
This should be increased with more data (or vice versa). The number of steps
should cover several iterations over all of the training data (epochs).
For example, if I had 200,000 images in my training set, one epoch would be
_200000/100 = 2000_ steps where _100_ is the batch size. So, if I wanted to
train for 30 epochs, I would simply do _2000*30 = 60000_ training steps.

Depending on how many images you have, this will likely take a long time to
train (several hours to maybe even a day), especially if only training on a laptop.
If you have access to GPUs, these will definitely help speed things up, and you
should certainly install the TensorFlow version with GPU support (supported on
[Ubuntu](https://www.tensorflow.org/install/install_linux) and
[Windows](https://www.tensorflow.org/install/install_windows) only).

On my Windows desktop computer with an Nvidia GTX 1080 graphics card, training
about 200,000 images with the script defaults took about an hour and a half.

Another alternative is to use a reduced label set (i.e. 256 vs 2350 Hangul
characters) which can reduce the computational complexity quite a bit.

As the script runs, you should hopefully see the printed training accuracies
grow towards 1.0, and you should also see a respectable testing accuracy after
the training. When the script completes, the exported model we should use will
be saved, by default, as `./saved-model/optimized_hangul_tensorflow.pb`. This is
a [Protocol Buffer](https://en.wikipedia.org/wiki/Protocol_Buffers) file
which represents a serialized version of our model with all the learned weights
and biases. This specific one is optimized for inference-only usage.


## Creating the Android Application

With the saved model, a simple Android application can be created that will be
able to classify handwritten Hangul that a user has drawn.
