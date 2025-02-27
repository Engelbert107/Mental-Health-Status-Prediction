import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore



def predict_sentiment_base(model, vectorizer, text, class_labels):
    """
    Predict the sentiment of a given text using a trained model and vectorizer (TF-IDF vectorizer).

    Parameters:
        model: Trained classifier (e.g., LinearSVC)
        vectorizer: Trained TF-IDF vectorizer
        text: A string or list of strings containing the input text
        class_labels: A list mapping encoded labels to sentiment categories

    Returns:
        Predicted sentiment class
    """
    # Ensure input is a list
    if isinstance(text, str):
        text = [text]
    
    # Transform text using the loaded vectorizer
    text_tfidf = vectorizer.transform(text)
    
    # Predict using the trained model
    predicted_label = model.predict(text_tfidf)
    
    # Map the predicted label to the class name
    return class_labels[predicted_label[0]]


def predict_sentiment(model, tokenizer, text, max_length, class_labels):
    """
    Predict the sentiment of a given text using the trained BiLSTM model.
    
    Args:
        text (str): Input text to classify.
        model (tf.keras.Model): Loaded BiLSTM model.
        tokenizer (Tokenizer): Loaded tokenizer.
        max_length (int): Maximum sequence length for padding.
        class_labels (list): List of sentiment labels.

    Returns:
        str: Predicted sentiment label.
    """
    # Tokenize and pad the input text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')

    # Get model prediction
    prediction = model.predict(padded_sequence)

    # Convert prediction to label
    predicted_label = np.argmax(prediction, axis=1)[0] # Get the index of the highest probability

    return class_labels[predicted_label]
