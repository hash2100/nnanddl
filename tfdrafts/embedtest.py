import os
os.chdir('d:/projects/python/nnanddl/')

import re
import os
import collections
import numpy as np
import random
import math

TOKEN_REGEX = re.compile(r'[A-Za-z]+|[!?.:,()]')
source_path = 'd:/projects/python/nnanddl/books'
vocabulary_size = 0

words = []
for file in os.listdir(source_path):
    with open(os.path.join(source_path, file), 'rb') as fin:
        for line in fin:
            words_on_line = TOKEN_REGEX.findall(line.decode().strip())
            words.extend(words_on_line)

    
def build_dataset(words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    counter = collections.Counter(words)
    count.extend(counter.most_common(len(counter.keys()) - 1))
    global vocabulary_size
    vocabulary_size = len(counter.keys()) - 1
        
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
            
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


data, count, dictionary, reverse_dictionary = build_dataset(words)
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0

def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
        if data_index == len(data):
            buffer[:] = data[:span]
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])