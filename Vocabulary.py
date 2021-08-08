"""Class for creating the vocabulary set based on frequently used words across 50000 Arxiv abstracts"""

#import statements bring relevant packages in scope
import torchtext
import dask.bag as db
import json 
import pandas as pd

class Vocabulary:
    def __init__(self, freq_threshold):
        #dictionary that will house indexes as keys and strings as values
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>", }
        #dictionary that functions as opposite of itos (strings as keys, numbers as values)
        self.stoi = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>":3, }
        #threshold for adding word to self.itos/self.stoi
        self.freq_threshold = freq_threshold

    def __len__(self):
        #returns length of the dictionary
        return len(self.itos)

    @staticmethod
    #method to get a tokenizer to use when numericalizing words
    def tokenizer_eng(text):
        print(text)
        tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
        return tokenizer(text)
    
    #this is the meat of the class: initialises index at 4 and checks through its source
    #if a word is not in the frequencies list, it is added, and if it occurs 
    #threshold number of times, it is added to the stoi and itos dictionaries
    #and idx is incremented for the next word
    def build_vocabulary(self, source):
        frequencies = {}
        idx = 4
        for abstract in source:
            for word in self.tokenizer_eng(abstract):
                if word not in frequencies:
                    frequencies[word] = 1
                else: 
                    frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    #tokenizes the input text, and returns a list of numbers corresponding to words or unknowns (if characters not in vocab)
    #that represents the string
    def numericalize(self, text): 
        tokenized_text = self.tokenizer_eng(text)
        return [self.stoi[_str] if _str in self.stoi else self.stoi["<UNK>"] for _str in tokenized_text]

def save_vocab(vocab, path):
    #(I borrowed this one from a stackoverflow page, as I wasn't super sure on how to write the right parameters to a file in a useful format)
    try:
        #writes the vocab list to a file in format: <word><value in Vocabulary.stoi corresponding to key: word>
        with open(path, 'w+') as f:     
            for token, index in vocab.stoi.items():
                f.write(f'{index}\t{token}')
    except:
        #moves on if the file cannot be written
        return None 
            
