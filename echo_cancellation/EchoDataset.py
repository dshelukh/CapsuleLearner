'''
@author: Dmitry
'''

import wave
import sys
sys.path.insert(0, '../')

from os import listdir
from os.path import isfile, join

from Trainer import *
import numpy as np
from scipy import signal

MAX_ABS_SHORT = 32768

def remove_echo(data, delay, level, framerate = 16000):
    echo_delay = int(framerate * delay)
    result = []
    for i in range(len(data)):
        if (i < echo_delay):
            result.append(data[i])
        else:
            echo = level * result[i - echo_delay]
            result.append(data[i] - echo)
    return np.array(result)

def add_echo(orig, echo_delay, echo_level, framerate = 16000):
    echo_start = int(echo_delay * framerate)
    echo = echo_level * np.pad(orig, (echo_start, 0), mode = 'constant')[:-echo_start]
    echoed = orig + echo
    return echoed, echo

class EchoCancellationDataset(Dataset):
    def __init__(self, base, *args, step_size = 1):
        Dataset.__init__(self, base, *args)
        self.nchannels = 1
        self.sampwidth = 2
        self.framerate = 16000

        self.step_size = step_size
        self.sample_len = 3 * self.framerate # 3 seconds
        self.echo_delay_min = 0.15
        self.echo_delay_max = 0.7
        self.echo_level_min = 0.1
        self.echo_level_max = 0.65

    def add_echo(self, orig):
        echo_time = np.random.uniform(self.echo_delay_min, self.echo_delay_max)
        echo_level = np.random.uniform(self.echo_level_min, self.echo_level_max)

        echoed, echo = add_echo(orig, echo_time, echo_level, self.framerate)
        return echoed, echo_time, echo_level, echo

    def get_batch(self, dataset, num, batch_size, shuffle = True):
        # TODO implement shuffled get_batch in base class
        start, end = batch_size * num, batch_size * (num + 1)
        ind = np.arange(len(dataset.images))
        if (shuffle):
            np.random.shuffle(ind)

        files = np.array(dataset.images)[ind[start:end]]

        # every file should have sample_len number of samples. Samples should be converted to floats in range (-1; 1)
        # echo should be added to input
        X = []
        y = []

        for filename in files:
            with wave.open(filename, 'rb') as wave_file:
                nframes = wave_file.getnframes()
                orig = np.frombuffer(wave_file.readframes(nframes), dtype = np.short)
                if nframes < self.sample_len:
                    startfrom = 0
                    sized = np.pad(orig, (self.sample_len - nframes, 0), mode = 'constant')
                else:
                    startfrom = np.random.randint(nframes - self.sample_len + 1)
                    sized = orig[startfrom:startfrom + self.sample_len]
                result = sized.astype(np.float32) / MAX_ABS_SHORT
                echoed, time, level, echo = self.add_echo(result)
                echo_start = startfrom + int(time * self.framerate)
                #print(time, level)
                f, times, Zxx = signal.stft(echoed)
                X.append(echoed) #np.rollaxis(np.abs(Zxx)**2, axis = -1).astype(np.float32))
                y.append(result) #[[1.0, level] if t >= echo_start else [0.0, 0.0] for t in times])
        # make 3D numpy arrays
        #f, t, Zxx = signal.stft(X)
        #print(f)
        #print('')
        #print(t)
        #print('')
        #print((np.abs(Zxx)**2).shape)
        #X = np.rollaxis(np.abs(Zxx)**2, axis = -1, start = 1).astype(np.float32)
        
        X = np.reshape(X, [-1, self.sample_len // self.step_size, self.step_size])
        y = np.reshape(y, [-1, self.sample_len // self.step_size, self.step_size])
        
        #fft = np.fft.rfft(X, axis = -1)
        #X = np.concatenate((np.abs(fft), np.angle(fft)), axis = -1).astype(np.float32)
        #fft = np.fft.rfft(y, axis = -1)
        #y = np.concatenate((np.abs(fft), np.angle(fft)), axis = -1).astype(np.float32)
        abs_logits = np.abs(X)
        #print(np.mean(np.abs(y)))
        return np.array(X), np.array(y)

class EchoDataset(DownloadableDataset):
    def __init__(self, train_split = 0.75, val_split = 0.1, data_dir = 'data/'):
        DownloadableDataset.__init__(self, 'http://festvox.org/cmu_arctic/cmu_arctic/packed/', ['cmu_us_bdl_arctic-0.95-release.zip'], data_dir, True)
        self.train_split = train_split
        self.val_split = val_split
        self.loadDataset()

    def loadDataset(self):
        audio_path = self.data_dir + 'cmu_us_bdl_arctic/wav'
        audiofiles = [join(audio_path, f) for f in listdir(audio_path) if isfile(join(audio_path, f))]

        num_files = len(audiofiles)
        train_index = int(self.train_split * num_files)
        val_index = train_index + int(self.val_split * num_files)

        train_files = audiofiles[:train_index]
        val_files = audiofiles[train_index:val_index]
        test_files = audiofiles[val_index:]

        # all preprocessing should be in runtime
        self.train = DatasetBase(train_files, train_files)
        self.val = DatasetBase(val_files, val_files)
        self.test = DatasetBase(test_files, test_files)

        print('Train: inputs - ', len(train_files))
        print('Val  : inputs - ', len(val_files))
        print('Test : inputs - ', len(test_files))

    def get_dataset_for_trainer(self, step_size = 1):
        return EchoCancellationDataset((self.train, self.val, self.test), step_size = step_size)

def save_wav(sound, name):
    with wave.open(name, 'wb') as wave_out:
        wave_out.setparams((1, 2, 16000, 1, 'NONE', 'not compressed'))
        sound = np.reshape(sound, [-1, *sound.shape[1:]])
        wave_out.writeframes((np.squeeze(sound) * MAX_ABS_SHORT).astype(np.short))

if __name__ == "__main__":
    dataset = EchoDataset().get_dataset_for_trainer(step_size = 160)
    X, y = dataset.get_batch(dataset.val, 0, 1, False)
    print(X[0].shape, y.shape)
    
    save_wav(X[0], './echoed.wav')
    save_wav(y[0], './original.wav')



