import re
import emoji
import string
import pickle
import gensim
import inflect
import numpy as np
import unicodedata
# import contractions
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import TruncatedSVD
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore


import nltk
nltk.download('punkt')
nltk.download('stopwords')


# Precompile regex patterns for efficiency
URL_PATTERN = re.compile(r'http\S+|www\S+|https\S+')
EXTRA_SPACES_PATTERN = re.compile(r'\s+')
CHAT_WORDS_PATTERN = re.compile(r'\b\w+\b')

# Initialize objects once
INFLECT_ENGINE = inflect.engine()
STEMMER = PorterStemmer()
STOP_WORDS = frozenset(stopwords.words('english'))  # Faster lookup

def denoise_text(text):
    """Remove HTML tags and fix contractions."""
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    return text  # Removed contractions for speed

def remove_non_ascii(words):
    """Remove non-ASCII characters using list comprehension."""
    return [unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore') for word in words]

def to_lowercase(words):
    """Convert words to lowercase using list comprehension."""
    return [word.lower() for word in words]

def remove_punctuation(words):
    """Remove punctuation using `str.translate()` (Faster than list comprehensions)."""
    translator = str.maketrans('', '', string.punctuation)
    return [word.translate(translator) for word in words]

def remove_extra_spaces(text):
    """Remove extra whitespaces using precompiled regex."""
    return EXTRA_SPACES_PATTERN.sub(' ', text).strip()

def replace_numbers(words):
    """Replace digits with words using Inflect (Vectorized Processing)."""
    return [INFLECT_ENGINE.number_to_words(word) if word.isdigit() else word for word in words]

def replace_emojis(text):
    """Convert emojis to text efficiently."""
    return emoji.demojize(text, delimiters=(" ", " ")).replace("_", " ")

def find_and_replace_chat_words(text, chat_words):
    """Replace chat words using regex and dictionary lookup."""
    def replace_match(match):
        word = match.group(0).lower()
        return chat_words.get(word, match.group(0))  # Preserve original if not found
    return CHAT_WORDS_PATTERN.sub(replace_match, text)

def remove_stopwords(words):
    """Remove stopwords using frozenset for fast lookup."""
    return [word for word in words if word not in STOP_WORDS]

def stem_words(words):
    """Apply stemming using list comprehension."""
    return [STEMMER.stem(word) for word in words]

def preprocess_text(text, chat_words={}):
    """Preprocess text with optimized pipeline."""
    if not isinstance(text, str):
        return ""

    # 1. Denoising
    text = denoise_text(text)
    text = replace_emojis(text)
    text = remove_extra_spaces(text)
    text = find_and_replace_chat_words(text, chat_words)

    # 2. Tokenization
    words = word_tokenize(text)

    # 3. Processing pipeline
    words = to_lowercase(words)
    words = remove_non_ascii(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    words = stem_words(words)

    return ' '.join(words)



def stratified_train_val_test_split(df, text_column, label_column, 
                                    val_size=0.1, test_size=0.1, random_state=6):
    """
    Splits a DataFrame into stratified train, validation, and test sets.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - text_column (str): Column name containing text data.
    - label_column (str): Column name containing categorical labels.
    - test_size (float): Proportion of the dataset to reserve for testing (default: 0.1).
    - val_size (float): Proportion of the remaining 20% to reserve for validation (default: 0.1).
    - random_state (int): Random seed for reproducibility (default: 42).

    Returns:
    - X_train, X_val, X_test: Numpy arrays of text data.
    - y_train, y_val, y_test: Numpy arrays of labels.
    """
    # Convert data to NumPy arrays
    X = df[text_column].values
    y = df[label_column].values

    # First, split into train + temp (80% train, 20% temp)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(test_size + val_size), random_state=random_state)
    for train_idx, temp_idx in sss1.split(X, y):
        X_train, X_temp = X[train_idx], X[temp_idx]
        y_train, y_temp = y[train_idx], y[temp_idx]

    # Now split the temp set (20% of the data) into validation and test (50% each of temp set, i.e., 10% each of the total dataset)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state)
    for val_idx, test_idx in sss2.split(X_temp, y_temp):
        X_val, X_test = X_temp[val_idx], X_temp[test_idx]
        y_val, y_test = y_temp[val_idx], y_temp[test_idx]

    return X_train, X_val, X_test, y_train, y_val, y_test



def save_pickle(obj, path):
    """Save an object using pickle."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    """Load an object using pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)

def save_numpy_array(arr, path):
    """Save a NumPy array."""
    np.save(path, arr)

def load_numpy_array(path):
    """Load a NumPy array."""
    return np.load(path)

def save_padded_sequences(train, val, test, path):
    """Save padded sequences using NumPy's np.savez."""
    np.savez(path, X_train_padded=train, X_val_padded=val, X_test_padded=test)

def load_padded_sequences(path):
    """Load padded sequences."""
    data = np.load(path)
    return data["X_train_padded"], data["X_val_padded"], data["X_test_padded"]

def save_npz_file(filepath, **kwargs):
    """Save multiple arrays into a compressed .npz file."""
    np.savez_compressed(filepath, **kwargs)

def load_npz_file(filepath):
    """Load a .npz file and return as a dictionary."""
    return np.load(filepath, allow_pickle=True)



def process_data_for_model_input(word2vec_path, X_train, X_val, X_test,
                                 MAX_NUM_WORDS=75000, EMBEDDING_DIM=50, MAX_SEQUENCE_LENGTH=100):
    # Load Word2Vec Model and Compute Reduced Embeddings
    print("Computing Word2Vec embeddings...")
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    svd = TruncatedSVD(n_components=EMBEDDING_DIM)
    reduced_embeddings = svd.fit_transform(word2vec_model.vectors)
    word2vec_reduced = {word: reduced_embeddings[i] for i, word in enumerate(word2vec_model.index_to_key)}
    save_pickle(word2vec_reduced, "../models/word2vec_reduced.pkl")
    print("Word2Vec embeddings saved!")
    
    # Compute Tokenizer
    print("Computing tokenizer...")
    full_texts = list(X_train) + list(X_val) + list(X_test)
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(full_texts)
    save_pickle(tokenizer, "../models/tokenizer.pkl")
    print("Tokenizer saved!")
    
    # Compute Embedding Matrix
    print("Computing embedding matrix...")
    input_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((input_size, EMBEDDING_DIM))

    oov_index = tokenizer.word_index.get("<OOV>", None)
    if oov_index:
        # Assign a random vector to the OOV token
        embedding_matrix[oov_index] = np.random.uniform(-0.25, 0.25, EMBEDDING_DIM)

    for word, i in tokenizer.word_index.items():
        if word in word2vec_reduced:
            embedding_matrix[i] = word2vec_reduced[word]
        
    # Compute Padded Sequences
    print("Computing padded sequences...")
    X_train_sequences = tokenizer.texts_to_sequences(X_train)
    X_val_sequences = tokenizer.texts_to_sequences(X_val)
    X_test_sequences = tokenizer.texts_to_sequences(X_test)

    X_train_padded = pad_sequences(X_train_sequences, padding="post", maxlen=MAX_SEQUENCE_LENGTH)
    X_val_padded = pad_sequences(X_val_sequences, padding="post", maxlen=MAX_SEQUENCE_LENGTH)
    X_test_padded = pad_sequences(X_test_sequences, padding="post", maxlen=MAX_SEQUENCE_LENGTH)
    
    print("\nTokenizer vocab size:", input_size)  # Check new vocab size

    print("\nData processing complete.")
    return X_train_padded, X_val_padded, X_test_padded, tokenizer, embedding_matrix





# Define a dictionary of chat word mappings
chat_words = {
    "7K": "Sick:-D Laugher",
    "A3": "Anytime, Anywhere, Anyplace",
    "ADIH": "Another day in hell",
    "AFAIK": "As Far As I Know",
    "AFK": "Away From Keyboard",
    "ASAP": "As Soon As Possible",
    "ASL": "Age, Sex, Location",
    "ATK": "At The Keyboard",
    "ATM": "At The Moment",
    "BAE": "Before anyone else",
    "BAK": "Back At Keyboard",
    "BBS": "Be Back Soon",
    "BBL": "Be Back Later",
    "BFN": "Bye For Now",
    "BFF": "Best friends forever",
    "B4": "Before",
    "B4N": "Bye For Now",
    "BRB": "Be Right Back",
    "BRT": "Be Right There",
    "BRUH": "Bro (casual slang)",
    "BSAAW": "Big smile and a wink",
    "BWL": "Bursting with laughter",
    "BTW": "By The Way",
    "CYA": "See You",
    "CU": "See You",
    "CUL8R": "See You Later",
    "CSL": "Can't stop laughing",
    "DM": "Direct Message",
    "FAQ": "Frequently Asked Questions",
    "FC": "Fingers Crossed",
    "FIMH": "Forever in my heart",
    "FINNA": "Going to (ex: 'I'm finna eat')",
    "FOMO": "Fear of Missing Out",
    "FWIW": "For What It's Worth",
    "FYI": "For Your Information",
    "GAL": "Get A Life",
    "GG": "Good Game",
    "GMTA": "Great Minds Think Alike",
    "GN": "Good Night",
    "GOAT": "Greatest of All Time",
    "GR8": "Great!",
    "G9": "Genius",
    "HIGHKEY": "Definitely, openly",
    "ICYMI": "In Case You Missed It",
    "IC": "I See",
    "ICQ": "I Seek you (also a chat program)",
    "IDK": "I Don't Know",
    "IDC": "I don't care",
    "IFYP": "I feel your pain",
    "ILU": "I Love You",
    "ILY": "I love you",
    "IMHO": "In My Honest/Humble Opinion",
    "IMO": "In My Opinion",
    "IMU": "I miss you",
    "IOW": "In Other Words",
    "IRL": "In Real Life",
    "JK": "Just kidding",
    "KISS": "Keep It Simple, Stupid",
    "LMK": "Let Me Know",
    "LDR": "Long Distance Relationship",
    "LIT": "Amazing, exciting, cool",
    "LOL": "Laughing Out Loud",
    "LMAO": "Laugh My A.. Off",
    "LOWKEY": "Kind of, secretly",
    "LTNS": "Long Time No See",
    "L8R": "Later",
    "M8": "Mate",
    "MFW": "My face when",
    "MRW": "My reaction when",
    "MTE": "My Thoughts Exactly",
    "NO CAP": "No lie / I'm serious",
    "NRN": "No Reply Necessary",
    "OIC": "Oh I See",
    "PITA": "Pain In The A..",
    "PRT": "Party",
    "PRW": "Parents Are Watching",
    "QPSA?": "Que Pasa?",
    "RN": "Right Now",
    "ROFL": "Rolling On The Floor Laughing",
    "ROFLOL": "Rolling On The Floor Laughing Out Loud",
    "ROTFLMAO": "Rolling On The Floor Laughing My A.. Off",
    "SK8": "Skate",
    "SMH": "Shaking My Head",
    "SUS": "Suspicious (used in gaming, ex: 'That's sus')",
    "STATS": "Your sex and age",
    "TBH": "To Be Honest",
    "TBT": "Throwback Thursday",
    "TFW": "That feeling when",
    "THX": "Thank You",
    "TIL": "Today I Learned",
    "TIME": "Tears in my eyes",
    "TL;DR": "Too Long; Didn't Read",
    "TNTL": "Trying not to laugh",
    "TTFN": "Ta-Ta For Now!",
    "TTYL": "Talk To You Later",
    "U": "You",
    "U2": "You Too",
    "U4E": "Yours For Ever",
    "WB": "Welcome Back",
    "WTF": "What The F...",
    "WTG": "Way To Go!",
    "WUF": "Where Are You From?",
    "W8": "Wait...",
    "WYWH": "Wish you were here",
    "YEET": "To throw something (or express excitement)",
    "ZZZ": "Sleeping, bored, tired"
}

