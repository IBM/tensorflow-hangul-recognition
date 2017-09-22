import imp
import glob
import os
import shutil
import tempfile
import unittest


TEST_PATH = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.join(TEST_PATH, '../tools/hangul-image-generator.py')
generator = imp.load_source('hangul-image-generator', SCRIPT_PATH)


class ImageGeneratorTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.expected_labels = ['a', 'b', 'c']

        # Create a label file and temporary output directory.
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.label_file = temp_file.name
        labels = open(self.label_file, 'w')
        for label in self.expected_labels:
            labels.write('%s\n' % label)

        self.output_dir = tempfile.mkdtemp()
        self.fonts_dir = os.path.join(TEST_PATH, 'test-fonts')

    @classmethod
    def tearDownClass(self):
        # Remove the created label file and output directory.
        os.remove(self.label_file)
        shutil.rmtree(self.output_dir)

    def test_file_generation(self):
        """Test the image generation function."""
        generator.generate_hangul_images(self.label_file,
                                         self.fonts_dir,
                                         self.output_dir)

        # Check that number of generated images is correct.
        expected_num_images = \
            len(self.expected_labels) * (1 + generator.DISTORTION_COUNT)
        num_images = len(glob.glob(os.path.join(self.output_dir,
                                                'hangul-images/*.jpeg')))
        self.assertEqual(expected_num_images, num_images)

        # Check that the CSV file was properly generated.
        with open(os.path.join(self.output_dir, 'labels-map.csv'), 'r') as f:
            csv_lines = f.read().splitlines()
        for line in csv_lines:
            file_path, label = line.strip().split(',')
            self.assertIn('.jpeg', file_path)
            self.assertIn(label, self.expected_labels)
