#!/usr/bin/env python

import argparse
import io
import os

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Default paths.
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH,
                                  './labels/2350-common-hangul.txt')
DEFAULT_TFRECORDS_DIR = os.path.join(SCRIPT_PATH, 'tfrecords-output')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, 'saved-model')

MODEL_NAME = 'hangul_tensorflow'
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

DEFAULT_NUM_TRAIN_STEPS = 30000
BATCH_SIZE = 100


def get_image(files, num_classes):
    """This method defines the retrieval image examples from TFRecords files.

    Here we will define how the images will be represented (grayscale,
    flattened, floating point arrays) and how labels will be represented
    (one-hot vectors).
    """

    # Convert filenames to a queue for an input pipeline.
    file_queue = tf.train.string_input_producer(files)

    # Create object to read TFRecords.
    reader = tf.TFRecordReader()

    # Read the full set of features for a single example.
    key, example = reader.read(file_queue)

    # Parse the example to get a dict mapping feature keys to tensors.
    # image/class/label: integer denoting the index in a classification layer.
    # image/encoded: string containing JPEG encoded image
    features = tf.parse_single_example(
        example,
        features={
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                                default_value='')
        })

    label = features['image/class/label']
    image_encoded = features['image/encoded']

    # Decode the JPEG.
    image = tf.image.decode_jpeg(image_encoded, channels=1)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.reshape(image, [IMAGE_WIDTH*IMAGE_HEIGHT])

    # Represent the label as a one hot vector.
    label = tf.stack(tf.one_hot(label, num_classes))
    return label, image


def export_model(model_output_dir, input_node_names, output_node_name):
    """Export the model so we can use it later.

    This will create two Protocol Buffer files in the model output directory.
    These files represent a serialized version of our model with all the
    learned weights and biases. One of the ProtoBuf files is a version
    optimized for inference-only usage.
    """

    name_base = os.path.join(model_output_dir, MODEL_NAME)
    frozen_graph_file = os.path.join(model_output_dir,
                                     'frozen_' + MODEL_NAME + '.pb')
    freeze_graph.freeze_graph(
        name_base + '.pbtxt', None, False, name_base + '.chkp',
        output_node_name, "save/restore_all", "save/Const:0",
        frozen_graph_file, True, ""
    )

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(frozen_graph_file, "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    optimized_graph_file = os.path.join(model_output_dir,
                                        'optimized_' + MODEL_NAME + '.pb')
    with tf.gfile.FastGFile(optimized_graph_file, "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("Inference optimized graph saved at: " + optimized_graph_file)


def weight_variable(shape):
    """Generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='weight')


def bias_variable(shape):
    """Generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='bias')


def main(label_file, tfrecords_dir, model_output_dir, num_train_steps):
    """Perform graph definition and model training.

    Here we will first create our input pipeline for reading in TFRecords
    files and producing random batches of images and labels.
    Next, a convolutional neural network is defined, and training is performed.
    After training, the model is exported to be used in applications.
    """
    labels = io.open(label_file, 'r', encoding='utf-8').read().splitlines()
    num_classes = len(labels)

    # Define names so we can later reference specific nodes for when we use
    # the model for inference later.
    input_node_name = 'input'
    keep_prob_node_name = 'keep_prob'
    output_node_name = 'output'

    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)

    print('Processing data...')

    tf_record_pattern = os.path.join(tfrecords_dir, '%s-*' % 'train')
    train_data_files = tf.gfile.Glob(tf_record_pattern)
    label, image = get_image(train_data_files, num_classes)

    tf_record_pattern = os.path.join(tfrecords_dir, '%s-*' % 'test')
    test_data_files = tf.gfile.Glob(tf_record_pattern)
    tlabel, timage = get_image(test_data_files, num_classes)

    # Associate objects with a randomly selected batch of labels and images.
    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label], batch_size=BATCH_SIZE,
        capacity=2000,
        min_after_dequeue=1000)

    # Do the same for the testing data.
    timage_batch, tlabel_batch = tf.train.batch(
        [timage, tlabel], batch_size=BATCH_SIZE,
        capacity=2000)

    # Create the model!

    # Placeholder to feed in image data.
    x = tf.placeholder(tf.float32, [None, IMAGE_WIDTH*IMAGE_HEIGHT],
                       name=input_node_name)
    # Placeholder to feed in label data. Labels are represented as one_hot
    # vectors.
    y_ = tf.placeholder(tf.float32, [None, num_classes])

    # Reshape the image back into two dimensions so we can perform convolution.
    x_image = tf.reshape(x, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1])

    # First convolutional layer. 32 feature maps.
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1],
                           padding='SAME')
    h_conv1 = tf.nn.relu(x_conv1 + b_conv1)

    # Max-pooling.
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding='SAME')

    # Second convolutional layer. 64 feature maps.
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    x_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1],
                           padding='SAME')
    h_conv2 = tf.nn.relu(x_conv2 + b_conv2)

    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding='SAME')

    # Third convolutional layer. 128 feature maps.
    W_conv3 = weight_variable([3, 3, 64, 128])
    b_conv3 = bias_variable([128])
    x_conv3 = tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1],
                           padding='SAME')
    h_conv3 = tf.nn.relu(x_conv3 + b_conv3)

    h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding='SAME')

    # Fully connected layer. Here we choose to have 1024 neurons in this layer.
    h_pool_flat = tf.reshape(h_pool3, [-1, 8*8*128])
    W_fc1 = weight_variable([8*8*128, 1024])
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

    # Dropout layer. This helps fight overfitting.
    keep_prob = tf.placeholder(tf.float32, name=keep_prob_node_name)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Classification layer.
    W_fc2 = weight_variable([1024, num_classes])
    b_fc2 = bias_variable([num_classes])
    y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # This isn't used for training, but for when using the saved model.
    tf.nn.softmax(y, name=output_node_name)

    # Define our loss.
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.stop_gradient(y_),
            logits=y
        )
    )

    # Define our optimizer for minimizing our loss. Here we choose a learning
    # rate of 0.0001 with AdamOptimizer. This utilizes someting
    # called the Adam algorithm, and utilizes adaptive learning rates and
    # momentum to get past saddle points.
    train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

    # Define accuracy.
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Initialize the variables.
        sess.run(tf.global_variables_initializer())

        # Initialize the queue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        checkpoint_file = os.path.join(model_output_dir, MODEL_NAME + '.chkp')

        # Save the graph definition to a file.
        tf.train.write_graph(sess.graph_def, model_output_dir,
                             MODEL_NAME + '.pbtxt', True)

        for step in range(num_train_steps):
            # Get a random batch of images and labels.
            train_images, train_labels = sess.run([image_batch, label_batch])

            # Perform the training step, feeding in the batches.
            sess.run(train_step, feed_dict={x: train_images, y_: train_labels,
                                            keep_prob: 0.5})

            # Print the training accuracy every 100 iterations.
            if step % 100 == 0:
                train_accuracy = sess.run(
                    accuracy,
                    feed_dict={x: train_images, y_: train_labels,
                               keep_prob: 1.0}
                )
                print("Step %d, Training Accuracy %g" %
                      (step, float(train_accuracy)))

            # Every 10,000 iterations, we save a checkpoint of the model.
            if step % 10000 == 0:
                saver.save(sess, checkpoint_file, global_step=step)

        # Save a checkpoint after training has completed.
        saver.save(sess, checkpoint_file)

        # Get number of samples in test set.
        sample_count = 0
        for f in test_data_files:
            sample_count += sum(1 for _ in tf.python_io.tf_record_iterator(f))

        # See how model did by running the testing set through the model.
        print('Testing model...')

        # We will run the test set through batches and sum the total number
        # of correct predictions.
        num_batches = int(sample_count/BATCH_SIZE) or 1
        total_correct_preds = 0

        # Define a different tensor operation for summing the correct
        # predictions.
        accuracy2 = tf.reduce_sum(correct_prediction)
        for step in range(num_batches):
            image_batch2, label_batch2 = sess.run([timage_batch, tlabel_batch])
            acc = sess.run(accuracy2, feed_dict={x: image_batch2,
                                                 y_: label_batch2,
                                                 keep_prob: 1.0})
            total_correct_preds += acc

        accuracy_percent = total_correct_preds/(num_batches*BATCH_SIZE)
        print("Testing Accuracy {}".format(accuracy_percent))

        export_model(model_output_dir, [input_node_name, keep_prob_node_name],
                     output_node_name)

        # Stop queue threads and close session.
        coord.request_stop()
        coord.join(threads)
        sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-file', type=str, dest='label_file',
                        default=DEFAULT_LABEL_FILE,
                        help='File containing newline delimited labels.')
    parser.add_argument('--tfrecords-dir', type=str, dest='tfrecords_dir',
                        default=DEFAULT_TFRECORDS_DIR,
                        help='Directory of TFRecords files.')
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR,
                        help='Output directory to store saved model files.')
    parser.add_argument('--num-train-steps', type=int, dest='num_train_steps',
                        default=DEFAULT_NUM_TRAIN_STEPS,
                        help='Number of training steps to perform. This value '
                             'should be increased with more data. The number '
                             'of steps should cover several iterations over '
                             'all of the training data (epochs). Example: If '
                             'you have 15000 images in the training set, one '
                             'epoch would be 15000/100 = 150 steps where 100 '
                             'is the batch size. So, for 10 epochs, you would '
                             'put 150*10 = 1500 steps.')
    args = parser.parse_args()
    main(args.label_file, args.tfrecords_dir,
         args.output_dir, args.num_train_steps)
