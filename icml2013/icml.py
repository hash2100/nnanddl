import numpy as np
import matplotlib.pyplot as plt
import csv



def show_sample(data, label):
    # Make those columns into a array of 8-bits pixels
    # The pixel intensity values are integers from 0 to 255
    pixels = np.array(data, dtype='uint8')

    # Reshape the array into 28 x 28 array (2-dimensional array)
    n = int(np.sqrt(len(pixels)))
    assert n**2 == len(pixels)
    pixels = pixels.reshape(n, n)

    # Plot
    plt.title('Label is {label}'.format(label=label))
    plt.imshow(pixels, cmap='gray')
    plt.show()
    
    
def one_hot(x, size):
    list = [0] * size
    list[x] = 1
    return list



class DataSet(object):
        
    def __init__(self, fileName):
        self._fileName = fileName
        self._train = dict()
        self._validate = dict()
        self._test = dict()
        
        
    def create_sets(self, ds = None, oneHot = True):
        self._train['labels'] = np.array([ r['emotion'] for r in ds 
                    if r['usage'] == 'Training' ])
        self._train['images'] = np.array([ r['pixels'] for r in ds 
                    if r['usage'] == 'Training' ])
        
        self._validate['labels'] = np.array([ r['emotion'] for r in ds 
                    if r['usage'] == 'PublicTest' ])
        self._validate['images'] = np.array([ r['pixels'] for r in ds 
                    if r['usage'] == 'PublicTest' ])
        
        self._test['labels'] = np.array([ r['emotion'] for r in ds 
                    if r['usage'] == 'PrivateTest' ])
        self._test['images'] = np.array([ r['pixels'] for r in ds 
                    if r['usage'] == 'PrivateTest' ])
    
        if oneHot:
            maxVal = max(
                    max(self._train['labels']),
                    max(self._validate['labels']),
                    max(self._test['labels'])
            )
            self._train['labels'] = np.array(
                    [one_hot(x, maxVal + 1) for x in self._train['labels']])
            self._validate['labels'] = np.array(
                    [one_hot(x, maxVal + 1) for x in self._validate['labels']])
            self._test['labels'] = np.array(
                    [one_hot(x, maxVal + 1) for x in self._test['labels']])
        
        
    def read_data(self):
        csvFile = open(self._fileName, 'rt')
        picReader = csv.reader(csvFile, delimiter=',')

        line = 0
        dataset = []
        for row in picReader:
            line += 1
            
            # skip the header
            if line == 1:
                # print(row)
                continue
            
            d = dict()
            d['emotion'] = np.array(int(row[0]))
            d['pixels'] = np.array([int(x) for x in row[1].split()])
            d['usage'] = np.array(row[2])
            
            dataset.append(d)
            
        csvFile.close()   
        self.create_sets(dataset)
        

    @property
    def dataset(self):
        return self._dataset
    
    
    @property
    def train(self):
        return self._train
    
    
    @property
    def validate(self):
        return self._validate
    
    
    @property
    def test(self):
        return self._test



class Batch(object):
        
    def __init__(self, num_samples, batch_size, shuffle = True):
        self._num_samples = num_samples
        self._start = 0
        self._perm = None
        self._shuffle = shuffle
        self._batch_size = batch_size
        
    
    def next_batch(self, samples, labels):
        assert len(samples) == self._num_samples
        assert len(labels) == self._num_samples
                  
        if self._start == 0:
            self._perm = np.arange(self._num_samples)
            if self._shuffle:
                np.random.shuffle(self._perm)
                
        window = None
                
        # check where there are sufficient samples here
        if self._start + self._batch_size > self._num_samples:
            rest_num_samples = self._num_samples - self._start
            window = self._perm[self._start:self._num_samples]
            
            # prepare to wrap around
            if self._shuffle:
                np.random.shuffle(self._perm)
                
            # add the remaining of the batch
            self._start = 0
            remaining = self._batch_size - rest_num_samples
            window = np.concatenate((
                    window, 
                    self._perm[self._start:remaining]), 
                axis=0)
    
            self._start += remaining
        else:
            # there is no danger of wrapping around
            nextpos = self._start + self._batch_size
            window = self._perm[self._start:nextpos]
            
            self._start = nextpos
        
        return samples[window], labels[window]
    


#ds = DataSet('fer2013/fer2013.csv')
#ds.read_data()

#show_sample(ds.train['images'][0], ds.train['labels'][0])

