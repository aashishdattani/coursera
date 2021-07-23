import tensorflow as tf
import csv
import random

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


import re
REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\|)|(\()|(\))|(\[)|(\])|(\%)|(\$)|(\>)|(\<)|(\{)|(\})")
REPLACE_WITH_SPACE = re.compile("(<br\s/><br\s/?)|(-)|(/)|(:).")

import preprocessor

embedding_dim = 100
max_length = 16
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size=1600000
test_portion=.1

num_sentences = 0
corpus = []
with open("training_cleaned.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        list_item = []
        line = preprocessor.clean(row[5])
        line = REPLACE_NO_SPACE.sub("", line.lower())
        line = REPLACE_WITH_SPACE.sub(" ", line)
        list_item.append(line)
        list_item.append(0 if row[0] == '0' else 1)
        num_sentences = num_sentences + 1
        corpus.append(list_item)
        if(num_sentences % 100000 == 0):
            print("Completed:", num_sentences)

sentences = []
labels = []
random.shuffle(corpus)
for x in range(training_size):
    sentences.append(corpus[x][0])
    labels.append(corpus[x][1])



