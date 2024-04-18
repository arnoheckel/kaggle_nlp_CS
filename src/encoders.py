import transformers
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

class BertEncoder:

    def __init__(self):
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.config = transformers.DistilBertConfig()
        self.dbert_pt = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased', config=self.config)

    def fit_transform(self, X=None, y=None):
        return self.transform(X)

    def transform(self, X):
        embeddings = []
        for sentence in tqdm(X):
            inputs = self.tokenizer(sentence,padding='max_length', max_length = 150, truncation=True, return_tensors='pt')
            outputs = self.dbert_pt(**inputs)
            embeddings.append(outputs.last_hidden_state[0][0].tolist())

        return np.vstack(embeddings)
    
    # USE this when enought RAM
    # def transform(self, X):
    #     # Let's tokenize the input sentences
    #     tokenized_sentences = self.tokenizer(X, padding='max_length', max_length = 150, truncation=True, return_tensors='pt')
    #     # Let's get the embeddings
    #     outputs = self.dbert_pt(**tokenized_sentences)
    #     # Let's get the embeddings
    #     embeddings = outputs.last_hidden_state[:,0,:].detach().numpy()

    #     return embeddings
    
class DebertaEncoder:

    def __init__(self):
        self.tokenizer = transformers.DebertaTokenizer.from_pretrained('microsoft/deberta-base')
        self.config = transformers.DebertaConfig()
        self.dbert_pt = transformers.DebertaModel.from_pretrained('microsoft/deberta-base', config=self.config)

    def fit_transform(self, X=None, y=None):
        return self.transform(X)

    def transform(self, X):
        embeddings = []
        for sentence in tqdm(X):
            inputs = self.tokenizer(sentence,padding='max_length', max_length = 150, truncation=True, return_tensors='pt')
            outputs = self.dbert_pt(**inputs)
            embeddings.append(outputs.last_hidden_state[0][0].tolist())

        return np.vstack(embeddings)
    
    # USE this when enought RAM
    # def transform(self, X):
    #     # Let's tokenize the input sentences
    #     tokenized_sentences = self.tokenizer(X, padding='max_length', max_length = 150, truncation=True, return_tensors='pt')
    #     # Let's get the embeddings
    #     outputs = self.dbert_pt(**tokenized_sentences)
    #     # Let's get the embeddings
    #     embeddings = outputs.last_hidden_state[:,0,:].detach().numpy()

    #     return embeddings

class Debertav2Encoder:

    def __init__(self):
        self.tokenizer = transformers.DebertaTokenizer.from_pretrained('microsoft/deberta-v2-xlarge')
        self.config = transformers.DebertaConfig()
        self.dbert_pt = transformers.DebertaModel.from_pretrained('microsoft/deberta-v2-xlarge', config=self.config)

    def fit_transform(self, X=None, y=None):
        return self.transform(X)
    
    def transform(self, X):
        embeddings = []
        for sentence in tqdm(X):
            inputs = self.tokenizer(sentence,padding='max_length', max_length = 150, truncation=True, return_tensors='pt')
            outputs = self.dbert_pt(**inputs)
            embeddings.append(outputs.last_hidden_state[0][0].tolist())

        return np.vstack(embeddings)
    
    # USE this when enought RAM
    # def transform(self, X):
    #     # Let's tokenize the input sentences
    #     tokenized_sentences = self.tokenizer(X, padding='max_length', max_length = 150, truncation=True, return_tensors='pt')
    #     # Let's get the embeddings
    #     outputs = self.dbert_pt(**tokenized_sentences)
    #     # Let's get the embeddings
    #     embeddings = outputs.last_hidden_state[:,0,:].detach().numpy()

    #     return embeddings

    

class RobertaEncoder:

    def __init__(self):
        self.tokenizer = transformers.RobertaTokenizer.from_pretrained('FacebookAI/roberta-base')
        self.roberta_pt = transformers.RobertaModel.from_pretrained("FacebookAI/roberta-base")

    def fit_transform(self, X=None, y=None):
        return self.transform(X)
    
    def transform(self, X):
        embeddings = []
        for sentence in tqdm(X):
            inputs = self.tokenizer(sentence,padding='max_length', max_length = 150, truncation=True, return_tensors='pt')
            outputs = self.roberta_pt(**inputs)
            embeddings.append(outputs.last_hidden_state[0][0].tolist())

        return np.vstack(embeddings)
    


class TFIDFEncoder():

    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit_transform(self, X):
        return self.vectorizer.fit_transform(X).toarray()
    
    def transform(self, X):
        return self.vectorizer.transform(X).toarray()

    