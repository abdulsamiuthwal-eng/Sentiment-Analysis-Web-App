import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure necessary NLTK datasets are downloaded
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """
    Cleans raw text by:
    1. Converting to lowercase
    2. Removing punctuation
    3. Removing numbers
    4. Removing URLs
    5. Removing non-ASCII / special characters
    6. Stripping extra whitespaces
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove numbers
    text = re.sub(r"\d+", "", text)
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove non-ascii / special characters
    text = re.sub(r"[^\w\s]", "", text)
    # Trim and clean up whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_text(text):
    """
    Tokenizes, removes stopwords, and lemmatizes the cleaned text.
    """
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    # Filter stopwords and apply lemmatization
    processed = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return " ".join(processed)
