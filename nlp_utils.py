
import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class NLPUtils():

    def tokenize(self, text):
        """Text tokenization

        Parameters
        ----------
        text: string
            Text to tokenize

        Returns
        -------
        text: Tokenized text.
        """
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9]", " ", text)
        words = word_tokenize(text)
        words = [w for w in words if w not in stopwords.words("english")]
        words = [WordNetLemmatizer().lemmatize(w, pos='v') for w in words]
        words = [PorterStemmer().stem(w) for w in words]
        return words