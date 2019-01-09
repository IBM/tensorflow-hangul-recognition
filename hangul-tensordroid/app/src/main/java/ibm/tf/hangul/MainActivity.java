package ibm.tf.hangul;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;

import java.util.HashMap;


public class MainActivity extends AppCompatActivity implements View.OnClickListener {

    private static final String LABEL_FILE = "2350-common-hangul.txt";
    private static final String MODEL_FILE = "optimized_hangul_tensorflow.pb";

    private HangulClassifier classifier;
    private PaintView paintView;
    private Button alt1, alt2, alt3, alt4;
    private LinearLayout altLayout;
    private EditText resultText;
    private TextView translationText;
    private String[] currentTopLabels;

    /**
     * This is called when the application is first initialized/started. Basic setup logic is
     * performed here.
     * @param savedInstanceState Bundle
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        paintView = (PaintView) findViewById(R.id.paintView);

        TextView drawHereText = (TextView) findViewById(R.id.drawHere);
        paintView.setDrawText(drawHereText);

        Button clearButton = (Button) findViewById(R.id.buttonClear);
        clearButton.setOnClickListener(this);

        Button classifyButton = (Button) findViewById(R.id.buttonClassify);
        classifyButton.setOnClickListener(this);

        Button backspaceButton = (Button) findViewById(R.id.buttonBackspace);
        backspaceButton.setOnClickListener(this);

        Button spaceButton = (Button) findViewById(R.id.buttonSpace);
        spaceButton.setOnClickListener(this);

        Button submitButton = (Button) findViewById(R.id.buttonSubmit);
        submitButton.setOnClickListener(this);

        altLayout = (LinearLayout) findViewById(R.id.altLayout);
        altLayout.setVisibility(View.INVISIBLE);

        alt1 = (Button) findViewById(R.id.alt1);
        alt1.setOnClickListener(this);
        alt2 = (Button) findViewById(R.id.alt2);
        alt2.setOnClickListener(this);
        alt3 = (Button) findViewById(R.id.alt3);
        alt3.setOnClickListener(this);
        alt4 = (Button) findViewById(R.id.alt4);
        alt4.setOnClickListener(this);

        translationText = (TextView) findViewById(R.id.translationText);
        resultText = (EditText) findViewById(R.id.editText);

        loadModel();
    }

    /**
     * This method is called when the user clicks a button in the view.
     * @param view
     */
    @Override
    public void onClick(View view) {
        switch (view.getId()) {
            case R.id.buttonClear:
                clear();
                break;
            case R.id.buttonClassify:
                classify();
                paintView.reset();
                paintView.invalidate();
                break;
            case R.id.buttonBackspace:
                backspace();
                altLayout.setVisibility(View.INVISIBLE);
                paintView.reset();
                paintView.invalidate();
                break;
            case R.id.buttonSpace:
                space();
                break;
            case R.id.buttonSubmit:
                altLayout.setVisibility(View.INVISIBLE);
                translate();
                break;
            case R.id.alt1:
            case R.id.alt2:
            case R.id.alt3:
            case R.id.alt4:
                useAltLabel(Integer.parseInt(view.getTag().toString()));
                break;
        }
    }

    /**
     * Delete the last character in the text input field.
     */
    private void backspace() {
        int len = resultText.getText().length();
        if (len > 0) {
            resultText.getText().delete(len - 1, len);
        }
    }

    /**
     * Add a space to the text input.
     */
    private void space() {
        resultText.append(" ");
    }

    /**
     * Clear the text and drawing to return to the beginning state.
     */
    private void clear() {
        paintView.reset();
        paintView.invalidate();
        resultText.setText("");
        translationText.setText("");
        altLayout.setVisibility(View.INVISIBLE);
    }

    /**
     * Perform the classification, updating UI elements with the results.
     */
    private void classify() {
        float pixels[] = paintView.getPixelData();
        currentTopLabels = classifier.classify(pixels);
        resultText.append(currentTopLabels[0]);
        altLayout.setVisibility(View.VISIBLE);
        alt1.setText(currentTopLabels[1]);
        alt2.setText(currentTopLabels[2]);
        alt3.setText(currentTopLabels[3]);
        alt4.setText(currentTopLabels[4]);
    }

    /**
     * Perform the translation using the current Korean text in the text input field.
     */
    private void translate() {
        String text = resultText.getText().toString();
        if (text.isEmpty()) {
            return;
        }

        HashMap<String, String> postData = new HashMap<>();
        postData.put("text", text);
        postData.put("source", "ko");
        postData.put("target", "en");
        String apikey = getResources().getString(R.string.apikey);
        String url = getResources().getString(R.string.url);
        HangulTranslator translator = new HangulTranslator(postData, translationText, apikey, url);
        translator.execute();
    }

    /**
     * This function will switch out the last classified character with the alternative given the
     * index in the top labels array.
     */
    private void useAltLabel(int index) {
        backspace();
        resultText.append(currentTopLabels[index]);
    }

    @Override
    protected void onResume() {
        paintView.onResume();
        super.onResume();
    }

    @Override
    protected void onPause() {
        paintView.onPause();
        super.onPause();
    }

    /**
     * Load pre-trained model in memory.
     */
    private void loadModel() {
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    classifier = HangulClassifier.create(getAssets(),
                            MODEL_FILE, LABEL_FILE, PaintView.FEED_DIMENSION,
                            "input", "keep_prob", "output");
                } catch (final Exception e) {
                    throw new RuntimeException("Error loading pre-trained model.", e);
                }
            }
        }).start();
    }
}
