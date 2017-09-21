package ibm.tf.hangul;

import android.content.res.AssetManager;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.TreeMap;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;


public class HangulClassifier {

    private TensorFlowInferenceInterface tfInterface;

    private String inputName;
    private String keepProbName;
    private String outputName;
    private int imageDimension;

    private List<String> labels;
    private float[] output;
    private String[] outputNames;

    /**
     * This function will create the TensorFlow inference interface and will populate all
     * the necessary data needed for inference.
     */
    public static HangulClassifier create(AssetManager assetManager,
                                          String modelPath, String labelFile, int inputDimension,
                                          String inputName, String keepProbName,
                                          String outputName) throws IOException {

        HangulClassifier classifier = new HangulClassifier();

        // These refer to the names of the nodes we care about in the model graph.
        classifier.inputName = inputName;
        classifier.keepProbName = keepProbName;
        classifier.outputName = outputName;

        // Read the labels from the given label file.
        classifier.labels = readLabels(assetManager, labelFile);

        // Create the TensorFlow interface using the specified model.
        classifier.tfInterface = new TensorFlowInferenceInterface(assetManager, modelPath);
        int numClasses = classifier.labels.size();

        // The size (in pixels) of each dimension of the image. Each dimension should be the same
        // since this is a square image.
        classifier.imageDimension = inputDimension;

        // This is a list of output nodes which should be filled by the inference pass.
        classifier.outputNames = new String[] { outputName };

        // This is the output node we care about.
        classifier.outputName = outputName;

        // The float buffer where the output of the softmax/output node will be stored.
        classifier.output = new float[numClasses];

        return classifier;
    }

    /**
     * Use the TensorFlow model and the given pixel data to produce possible classifications.
     */
    public String[] classify(final float[] pixels) {

        // Feed the image data and the keep probability (for the dropout placeholder) to the model.
        tfInterface.feed(inputName, pixels, 1, imageDimension, imageDimension, 1);
        tfInterface.feed(keepProbName, new float[] { 1 });

        // Run inference between the input and output.
        tfInterface.run(outputNames);

        // Fetch the contents of the Tensor denoted by 'outputName', copying the contents
        // into 'output'.
        tfInterface.fetch(outputName, output);

        // Map each Float to it's index. The higher values are the only ones we care about
        // and these are unlikely to have duplicates.
        TreeMap<Float,Integer> map = new TreeMap<>();
        for (int i = 0; i < output.length; i++) {
            map.put( output[i], i );
        }

        // Let's only return the top five labels in order of confidence.
        Arrays.sort(output);
        String[] topLabels = new String[5];
        for (int i = output.length; i > output.length-5; i--) {
            topLabels[output.length - i] = labels.get(map.get(output[i-1]));
        }
        return topLabels;
    }

    /**
     * Read in all our labels into memory so that we can map Korean characters to confidences
     * listed in the output vectors.
     */
    private static List<String> readLabels(AssetManager am, String fileName) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(am.open(fileName)));

        String line;
        List<String> labels = new ArrayList<>();
        while ((line = reader.readLine()) != null) {
            labels.add(line);
        }
        reader.close();
        return labels;
    }
}
