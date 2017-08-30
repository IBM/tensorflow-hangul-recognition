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
