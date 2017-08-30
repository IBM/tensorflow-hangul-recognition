#!/usr/bin/env python

import argparse
import glob
import io
import os
import random

import numpy
from PIL import Image, ImageFont, ImageDraw
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Default data paths.
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH,
                                  '../labels/2350-common-hangul.txt')
DEFAULT_FONTS_DIR = os.path.join(SCRIPT_PATH, '../fonts')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, '../image-data')

# Subdirectory names that will be created in the output directory for
# partitioning the training set and testing set.
TRAIN_IMAGES_DIR = 'train-images'
TEST_IMAGES_DIR = 'test-images'

# Number of random distortion images to generate per font and character.
DISTORTION_COUNT = 3

# Width and height of the resulting image.
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64


def generate_hangul_images(label_file, fonts_dir, output_dir):
    """Generate Hangul image files.

    This will take in the passed in labels file and will generate several
    images using the font files provided in the font directory. The font
    directory is expected to be populated with *.ttf (True Type Font) files.
    The generated images will be stored in the given output directory.
    """
    labels = io.open(label_file, 'r', encoding='utf-8').read().splitlines()

    train_dir = os.path.join(output_dir, TRAIN_IMAGES_DIR)
    if not os.path.exists(train_dir):
        os.makedirs(os.path.join(train_dir))

    test_dir = os.path.join(output_dir, TEST_IMAGES_DIR)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # Get a list of the fonts.
    fonts = glob.glob(os.path.join(fonts_dir, '*.ttf'))

    count = 0
    for character in labels:
        count += 1
        subcount = 0

        # Create directory for current character.
        train_char_dir = os.path.join(train_dir, character)
        test_char_dir = os.path.join(test_dir, character)
        if not os.path.exists(train_char_dir):
            os.makedirs(train_char_dir)
        if not os.path.exists(test_char_dir):
            os.makedirs(test_char_dir)

        # Loop through each font, creating several different images of each
        # character.
        for font in fonts:
            subcount += 1
            image = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), color=0)
            font = ImageFont.truetype(font, 48)
            drawing = ImageDraw.Draw(image)
            w, h = drawing.textsize(character, font=font)
            drawing.text(
                ((IMAGE_WIDTH-w)/2, (IMAGE_HEIGHT-h)/2),
                character,
                fill=(255),
                font=font
            )
            file_string = 'hangul_{}_{}.jpeg'.format(count, subcount)

            # Randomly assign roughly 10% of the images to the testing set.
            if random.randint(0, 9) < 1:
                file_path = os.path.join(test_char_dir, file_string)
            else:
                file_path = os.path.join(train_char_dir, file_string)

            image.save(file_path, 'JPEG')

            for i in range(DISTORTION_COUNT):
                file_string = 'hangul_{}_{}_{}.jpeg'.format(count, subcount, i)
                if random.randint(0, 9) < 1:
                    file_path = os.path.join(test_char_dir, file_string)
                else:
                    file_path = os.path.join(train_char_dir, file_string)
                arr = numpy.array(image)

                distorted_array = elastic_distort(
                    arr, alpha=random.randint(30, 36),
                    sigma=random.randint(5, 6)
                )
                distorted_image = Image.fromarray(distorted_array)
                distorted_image.save(file_path, 'JPEG')


def elastic_distort(image, alpha, sigma):
    """Perform elastic distortion on an image.

    Here, alpha refers to the scaling factor that controls the intensity of the
    deformation. The sigma variable refers to the Gaussian filter standard
    deviation.
    """
    random_state = numpy.random.RandomState(None)
    shape = image.shape

    dx = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1),
        sigma, mode="constant"
    ) * alpha
    dy = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1),
        sigma, mode="constant"
    ) * alpha

    x, y = numpy.meshgrid(numpy.arange(shape[0]), numpy.arange(shape[1]))
    indices = numpy.reshape(y+dy, (-1, 1)), numpy.reshape(x+dx, (-1, 1))
    return map_coordinates(image, indices, order=1).reshape(shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-file', type=str, dest='label_dir',
                        default=DEFAULT_LABEL_FILE,
                        help='File containing newline delimited labels.')
    parser.add_argument('--font-dir', type=str, dest='fonts_dir',
                        default=DEFAULT_FONTS_DIR,
                        help='Directory of ttf fonts to use.')
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR,
                        help='Output directory to store generated images.')
    args = parser.parse_args()
    generate_hangul_images(args.label_dir, args.fonts_dir, args.output_dir)
