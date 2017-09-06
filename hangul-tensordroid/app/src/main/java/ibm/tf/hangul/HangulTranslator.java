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


public class HangulTranslator extends AsyncTask<String, Void, String> {

    private static final String TRANSLATE_API_ENDPOINT =
            "https://gateway.watsonplatform.net/language-translator/api/v2/translate";

    private JSONObject postData;
    private TextView view;

    public HangulTranslator(Map<String, String> postData, TextView view) {
        if (postData != null) {
            this.postData = new JSONObject(postData);
        }
        this.view = view;
    }

    @Override
    protected void onPreExecute() {
        super.onPreExecute();
    }

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

            // TODO: Extract to config file/store.
            String name = "CHANGE ME";
            String password = "CHANGE ME";

            // Set authorization header.
            String authString = name + ":" + password;
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

    @Override
    protected void onPostExecute(String result) {
        if (result.isEmpty()) {
            return;
        }
        try {
            JSONObject json = new JSONObject(result);
            JSONArray array = (JSONArray) json.get("translations");
            Object translation = ((JSONObject) array.get(0)).get("translation");
            view.setText(translation.toString());
        }
        catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }
}
