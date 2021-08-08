import torchtext
import dask.bag as db
import json 
import pandas as pd

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>", }
        self.stoi = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>":3, }
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
        return tokenizer(text)

    def build_vocabulary(self, source):
        frequencies = {}
        idx = 4
        for abstract in source:
            for word in self.tokenizer_eng(abstract):
                frequencies[word] += 1 
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text): 
        tokenized_text = self.tokenizer_eng(text)
        return [self.stoi[_str] if _str in self.stoi else self.stoi["<UNK>"] for _str in tokenized_text]

def save_vocab(vocab, path):
    with open(path, 'w+') as f:     
        for token, index in vocab.stoi.items():
            f.write(f'{index}\t{token}')
            
if __name__ == '__main__':
    data = db.read_text("arxiv-metadata-oai-snapshot.json").map(json.loads).compute()
    print('Loaded.')
    data = data[200000:250001]
    print('Shrunken')
    df = pd.DataFrame(data)
    print('Converted to dataframe')
    df = df["abstract"]
    print('Reduced to abstracts')
    vocab = Vocabulary(100)
    vocab.build_vocabulary()
    print('Vocabulary built')
    save_vocab(vocab, 'vocab_list.txt')
    print('Vocabulary written.')
