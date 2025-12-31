import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk import pos_tag
from nltk.corpus import wordnet
from typing import List, Optional

class TextPreprocessor:
    def __init__(self, use_stemming: bool = False, use_lemmatization: bool = True, 
                 remove_stopwords: bool = True, min_token_length: int = 2):
        # Ensure resources are downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'], quiet=True)
            
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = SnowballStemmer('english')
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords
        self.min_length = min_token_length
    
    def _get_wordnet_pos(self, tag):
        if tag.startswith('J'): return wordnet.ADJ
        elif tag.startswith('V'): return wordnet.VERB
        elif tag.startswith('R'): return wordnet.ADV
        return wordnet.NOUN
    
    def preprocess(self, text: str) -> List[str]:
        if not isinstance(text, str):
            return []
            
        # Clean text
        text = text.lower()
        text = re.sub(r'http\S+|www\.\S+', '', text)  # Remove URLs
        text = re.sub(r'<[^>]*>', '', text)           # Remove HTML
        text = re.sub(r'[^\w\s]', '', text)           # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()      # Normalize whitespace
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
        
        # Normalize (lemmatization preferred for quality; stemming for speed)
        if self.use_lemmatization:
            pos_tags = pos_tag(tokens)
            tokens = [self.lemmatizer.lemmatize(w, self._get_wordnet_pos(p)) for w, p in pos_tags]
        elif self.use_stemming:
            tokens = [self.stemmer.stem(t) for t in tokens]
        
        return [t for t in tokens if len(t) >= self.min_length]


