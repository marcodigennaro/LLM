from typing import Tuple, Dict, List
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding


def encode_text(text: str) -> Tuple[List[int], Dict[str, int]]:
    """
    Encodes a given text string into a sequence of integers where each integer represents a unique word in the text,
    based on the tokenizer's vocabulary. Additionally, it returns the mapping of words to their respective indices
    in the tokenizer's internal dictionary.

    Args:
        text (str): The text string to be encoded.

    Returns:
        Tuple[List[int], Dict[str, int]]:
            - A list of integers representing the encoded text.
            - A dictionary where keys are words and values are the corresponding unique indices used for encoding.

    Example:
        >>> encode_text("hello world hello")
        ([1, 2, 1], {'hello': 1, 'world': 2})
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    encoded = tokenizer.texts_to_sequences([text])[0]
    return encoded, tokenizer


def create_sequences(encoded: List[int]) -> List[List[int]]:
    sequences = []
    for i in range(1, len(encoded)):
        sequence = encoded[:i + 1]
        sequences.append(sequence)
    return sequences


def pad_sequences_to_same_length(sequences: List[List[int]], padding: str = 'pre') -> np.ndarray:
    max_length = max(len(seq) for seq in sequences)
    return pad_sequences(sequences, maxlen=max_length, padding=padding)


def split_sequences(sequences: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X, y = sequences[:, :-1], sequences[:, -1]
    return X, y


def one_hot_encode_labels(labels: np.ndarray, num_classes: int) -> np.ndarray:
    return to_categorical(labels, num_classes=num_classes)


def prepare_data(text: str) -> Tuple[np.ndarray, np.ndarray, int, int]:
    encoded, tokenizer = encode_text(text)
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1  # Including zero index
    sequences = create_sequences(encoded)
    padded_sequences = pad_sequences_to_same_length(sequences)
    X, y = split_sequences(padded_sequences)
    y = one_hot_encode_labels(y, vocab_size)
    max_length = padded_sequences.shape[1]  # Obtain the length of the padded sequences
    return X, y, vocab_size, max_length


def create_rnn_model(vocab_size: int, max_length: int) -> Sequential:
    model = Sequential()
    model.add(
        Embedding(input_dim=vocab_size, output_dim=50, input_length=max_length - 1))  # input_length is max_length - 1
    model.add(SimpleRNN(100))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def generate_text(model, tokenizer, seed_text, num_words, max_length):
    """
    Generate text using a trained RNN model based on a seed text.

    Args:
        model (Sequential): The trained Keras model for text generation.
        tokenizer (Tokenizer): The Keras Tokenizer object used for text encoding.
        seed_text (str): Initial text to start the text generation process.
        num_words (int): Number of words to generate after the seed text.
        max_length (int): The maximum length of the sequence that the model was trained on.

    Returns:
        str: The generated text.
    """
    result = seed_text
    for _ in range(num_words):
        # Encode the current set of words
        encoded = tokenizer.texts_to_sequences([seed_text])[0]
        encoded = pad_sequences([encoded], maxlen=max_length-1, truncating='pre')

        # Predict the probability of the next word
        preds = model.predict(encoded, verbose=0).flatten()
        next_word_index = np.argmax(preds)
        next_word = tokenizer.index_word.get(next_word_index, '')

        # Append to the result and update seed text
        result += ' ' + next_word
        seed_text += ' ' + next_word  # Update seed text to include the new word

    return result
