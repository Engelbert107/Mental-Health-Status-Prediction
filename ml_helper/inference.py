import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore


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
