import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#define a fixed time step for all spectrograms
fixed_time_steps = 32
scaler = MinMaxScaler(feature_range=(0, 1))


def compute_mel_spectrogram(raw_values, sr=16000, n_fft=256, hop_length=16, n_mels=64):
    """
    Compute a mel spectrogram for raw EEG

    Args:
        raw_values (array-like): 1D array of raw signal data.
        sr (int): Sampling rate (default: 16000 Hz).
        n_fft (int): Number of FFT components (default: 1024).
        hop_length (int): Number of samples between successive frames (default: 512).
        n_mels (int): Number of mel bands to generate (default: 128).

    Returns:
        np.ndarray: Mel spectrogram (shape: [n_mels, time]).
    """
    # Compute the Short-Time Fourier Transform (STFT)
    #print(raw_values)
    #print(len(raw_values))
    stft = librosa.stft(np.array(raw_values), n_fft=n_fft, hop_length=hop_length)
    mel_spec = librosa.feature.melspectrogram(S=np.abs(stft)**2, sr=sr, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    return log_mel_spec

def process_spectrogram(mel_spec, fixed_time_steps):
    #normalize
    mel_spec_normalized = scaler.fit_transform(mel_spec)

    #pad to fixed time steps
    if mel_spec_normalized.shape[1] > fixed_time_steps:
        mel_spec_normalized = mel_spec_normalized[:, :fixed_time_steps]
    else:
        mel_spec_normalized = np.pad(
            mel_spec_normalized,
            ((0, 0), (0, fixed_time_steps - mel_spec_normalized.shape[1])),
            mode="constant",
        )
    return mel_spec_normalized

def compute_process_and_plot(raw_signal):
    mel_spec                  = compute_mel_spectrogram(raw_signal)
    processed_mel_spectrogram = process_spectrogram(mel_spec, fixed_time_steps)
    mel_input_array           = np.array(processed_mel_spectrogram)
    plt.figure()
    plt.imshow(mel_input_array, aspect='auto', origin='lower', cmap='viridis')
    plt.title('Mel Spectrogram')
    plt.colorbar()
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')
    plt.show()
    
