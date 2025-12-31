# Preprocessing Implementation

As specified in our project proposal, we implemented a complete text preprocessing pipeline:

## Pipeline Steps

### 1. Tokenization
- Word-level tokenization
- Split on whitespace and punctuation

### 2. Stopword Removal
- Removed common English stopwords (45 words)
- Examples: "the", "is", "at", "which", "on"
- Preserves semantic content while reducing noise

### 3. Stemming
- **Algorithm**: Porter Stemmer
- **Purpose**: Conflates related word forms to their root
- **Examples**:
  - "running", "runs", "ran" → "run"
  - "retrieval", "retrieve", "retrieved" → "retriev"
  - "searching", "searches" → "search"

### 4. Normalization
- Lowercase conversion
- Removal of punctuation and special characters
- Whitespace normalization

## Implementation Details

```python
from nltk.stem import PorterStemmer

class PreprocessorWithStemming:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set([...])  # 45 common English stopwords
    
    def clean(self, text):
        # 1. Normalize
        text = text.lower()
        text = remove_punctuation(text)
        
        # 2. Tokenize
        tokens = text.split()
        
        # 3. Remove stopwords
        tokens = [t for t in tokens if t not in self.stop_words]
        
        # 4. Apply stemming
        tokens = [self.stemmer.stem(t) for t in tokens]
        
        return " ".join(tokens)
```

## Example Transformation

**Original text:**
> "Information retrieval systems are searching through databases of documents"

**After preprocessing:**
> "inform retriev system search databas document"

## Impact on Performance

Stemming provides:
- ✅ Better term matching (conflates morphological variants)
- ✅ Reduced vocabulary size
- ✅ Improved recall (more term matches)
- ⚠️ Slight precision reduction (over-stemming can conflate unrelated terms)

## Consistency

The same preprocessing pipeline is applied to:
- All document text
- All query text
- Ensures fair comparison across models

