import numpy as np
import resampy
import soundfile as sf
import tensorflow_hub as hub

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
