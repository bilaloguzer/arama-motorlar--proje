import re
from nltk.stem import PorterStemmer
from typing import List

class ProposalAlignedPreprocessor:
    """
    Preprocessing pipeline matching project proposal:
    - Tokenization
    - Stopword removal  
    - Stemming
    - Text normalization
    """
    def __init__(self):
        self.stemmer = PorterStemmer()
        
        # Standard English stopwords as per proposal
        self.stop_words = set([
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'this', 'but', 'or', 'not', 'been',
            'have', 'had', 'do', 'does', 'did', 'can', 'could', 'would', 'should'
        ])
    
    def clean(self, text: str) -> str:
        """
        Complete preprocessing pipeline as specified in proposal:
        1. Normalization (lowercase, remove special chars)
        2. Tokenization (word-level)
        3. Stopword removal
        4. Stemming (Porter Stemmer)
        """
        if not isinstance(text, str):
            return ""
        
        # Step 1: Text normalization
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()   # Normalize whitespace
        
        # Step 2: Tokenization
        tokens = text.split()
        
        # Step 3: Stopword removal
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        
        # Step 4: Stemming (as per proposal)
        tokens = [self.stemmer.stem(t) for t in tokens]
        
        return " ".join(tokens)

