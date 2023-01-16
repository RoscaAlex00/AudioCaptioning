import csv

import numpy as np
import resampy
import soundfile as sf
import tensorflow_hub as hub
import tensorflow as tf

yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)


def yamnet_classify(file_name):
    wav_data, sr = sf.read(file_name, dtype=np.int16)
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype

    waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]
    waveform = waveform.astype('float32')

    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)

    if sr != 16000:
        waveform = resampy.resample(waveform, sr, 16000)
    scores, embeddings, spectrogram = yamnet_model(waveform)

    return scores.numpy()


def class_names_from_csv():
    """Returns the class map corresponding to score vector."""
    with tf.io.gfile.GFile(yamnet_model.class_map_path().numpy()) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        num_classes = sum(1 for l in csvfile) - 1
        class_map = [[] for l in range(num_classes)]
        csvfile.seek(0)
        _ = next(reader)
        for r in reader:
            c_ = ' '.join(r[2:])
            c_ = c_.replace('etc.', '')
            if '(' in c_:
                c_ = c_[:c_.index('(') - 1]
            assert '.' not in c_
            class_map[int(r[0])] = c_

    return class_map
