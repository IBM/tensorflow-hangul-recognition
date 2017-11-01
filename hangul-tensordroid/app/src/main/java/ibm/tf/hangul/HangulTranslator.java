package ibm.tf.hangul;


import android.os.AsyncTask;
import android.util.Base64;
import android.widget.TextView;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.URL;
import java.util.Map;

import javax.net.ssl.HttpsURLConnection;


/**
 * This class is used for asynchronously sending Korean text to the Watson Language Translator
 * service for translation.
 */
public class HangulTranslator extends AsyncTask<String, Void, String> {

    private static final String TRANSLATE_API_ENDPOINT =
            "https://gateway.watsonplatform.net/language-translator/api/v2/translate";

    private JSONObject postData;
    private TextView view;

    // These are the service provided username and password used in authenticating with the
    // translate API service.
    private String username;
    private String password;

    public HangulTranslator(Map<String, String> postData, TextView view, String username,
                            String password) {
        if (postData != null) {
            this.postData = new JSONObject(postData);
        }
        this.view = view;
        this.username = username;
        this.password = password;
    }

    @Override
    protected void onPreExecute() {
        super.onPreExecute();
    }

    /**
     * This is an asynchronously called function, that will send an HTTP POST request to the
     * translator endpoint with the Korean text in the request.
     * @return String response from the translator service.
     */
    @Override
    protected String doInBackground(String... params) {

        String result = "";

        try {
            URL url = new URL(TRANSLATE_API_ENDPOINT);
            HttpsURLConnection urlConnection = (HttpsURLConnection) url.openConnection();
            urlConnection.setDoInput(true);
            urlConnection.setDoOutput(true);

            urlConnection.setRequestProperty("Content-Type", "application/json");
            urlConnection.setRequestProperty("Accept", "application/json");
            urlConnection.setRequestMethod("POST");

            // Set authorization header.
            String authString = this.username + ":" + this.password;
            byte[] base64Bytes = Base64.encode(authString.getBytes(), Base64.DEFAULT);
            String base64String = new String(base64Bytes);
            urlConnection.setRequestProperty("Authorization", "Basic " + base64String);

            if (this.postData != null) {
                OutputStreamWriter writer = new OutputStreamWriter(urlConnection.getOutputStream());
                writer.write(postData.toString());
                writer.flush();
                writer.close();
            }

            int statusCode = urlConnection.getResponseCode();
            if (statusCode ==  200) {
                InputStreamReader streamReader = new InputStreamReader(
                        urlConnection.getInputStream());
                BufferedReader bufferedReader = new BufferedReader(streamReader);

                String inputLine;
                StringBuilder response = new StringBuilder();

                while ((inputLine = bufferedReader.readLine()) != null) {
                    response.append(inputLine);
                }
                streamReader.close();
                bufferedReader.close();
                result = response.toString();
            }
            else {
                System.out.println("Error translating. Response Code: " + statusCode);
            }
        }
        catch (Exception e) {
            System.out.println(e.getMessage());

        }
        return result;
    }

    /**
     * This is called after the response string is returned. This parses out the English
     * translation and sets the necessary TextView for displaying the translation.
     * @param result String
     */
    @Override
    protected void onPostExecute(String result) {
        if (result.isEmpty()) {
            return;
        }
        try {
            JSONObject json = new JSONObject(result);
            JSONArray array = json.optJSONArray("translations");
            Object translation = array.optJSONObject(0).get("translation");
            view.setText(translation.toString());
        }
        catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }
}
