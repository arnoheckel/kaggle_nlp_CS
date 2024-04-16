import transformers
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer



class BertEncoder:

    def __init__(self):
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.config = transformers.DistilBertConfig()
        self.dbert_pt = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased', config=self.config)

    def fit_transform(self, X=None, y=None):
        pass

    def transform(self, X):
        embeddings = []
        for sentence in X:
            inputs = self.tokenizer(sentence, return_tensors="pt")
            outputs = self.dbert_pt(**inputs)
            embeddings.append(outputs.last_hidden_state[0][0].tolist())

        return np.vstack(embeddings)




class TFIDFEncoder():

    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit_transform(self, X):
        return self.vectorizer.fit_transform(X)
    
    def transform(self, X):
        return self.vectorizer.transform(X)

    