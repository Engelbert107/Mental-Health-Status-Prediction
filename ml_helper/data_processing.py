import re
import emoji
import pickle
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import StratifiedShuffleSplit



def find_and_replace_chat_words(text, chat_words):
    """Replaces chat words in the given text with their full meanings."""
    text = str(text)  # Ensure input is string
    words = text.split()  # Split text into words
    
    # Replace words with dictionary values (case insensitive lookup)
    words = [chat_words.get(word.upper(), word) for word in words]
    
    return ' '.join(words)


def replace_emojis(text):
    """Replaces emojis with descriptive text."""
    return emoji.demojize(text).replace(":", " ").replace("_", " ")


def preprocess_text(text):
    """Preprocesses text by cleaning, normalizing, and tokenizing."""
    if not isinstance(text, str):  # Handle NaN or non-string values
        return ""
    
    text = replace_emojis(text)  # Convert emojis to words
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove special characters and punctuation (keep only alphanumeric characters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Convert to lowercase for NLP processing
    text = text.lower()

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stop words and apply stemming
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    cleaned_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]

    return ' '.join(cleaned_tokens)



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

