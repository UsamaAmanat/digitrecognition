from flask import Flask, request, jsonify
import os
import torch
import pickle
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define DigitClassifier class first
class DigitClassifier(torch.nn.Module):
    def __init__(self, input_size=12, hidden_size=64, num_layers=2, num_classes=10):
        super(DigitClassifier, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# Load the existing model and scaler
model = DigitClassifier()
model.load_state_dict(torch.load('digit_model.pth', weights_only=True))
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
model.eval()

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'wav'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# MFCC computation function
def compute_mfcc(file_path, num_ceps=12, num_filt=40, nfft=512, noise_level=0.0):
    sample_rate, signal = wavfile.read(file_path)
    if len(signal.shape) > 1:
        signal = signal[:, 0]
    if sample_rate != 8000:
        return None, f"Invalid sample rate: {sample_rate}, expected 8000"
    if noise_level > 0:
        signal = add_noise(signal, noise_level)
    preemphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - preemphasis * signal[:-1])
    frame_size, frame_stride = 0.025, 0.01
    frame_length, frame_step = int(round(frame_size * sample_rate)), int(round(frame_stride * sample_rate))
    signal_length = len(emphasized_signal)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) + 1
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z)
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    frames *= np.hamming(frame_length)
    mag_frames = np.absolute(np.fft.rfft(frames, nfft))
    pow_frames = ((1.0 / nfft) * (mag_frames ** 2))
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, num_filt + 2)
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))
    bin = np.floor((nfft + 1) * hz_points / sample_rate)
    fbank = np.zeros((num_filt, int(np.floor(nfft / 2 + 1))))
    for m in range(1, num_filt + 1):
        f_m_minus, f_m, f_m_plus = int(bin[m - 1]), int(bin[m]), int(bin[m + 1])
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :num_ceps]
    return mfcc, None

def add_noise(signal, noise_level=0.005):
    noise = np.random.randn(len(signal)) * noise_level * np.max(np.abs(signal))
    return signal + noise

# Prediction function
def predict(file_path, model, scaler, noise_level=0.0, max_frames=100):
    mfcc, error = compute_mfcc(file_path, noise_level=noise_level)
    if error:
        return None, error
    if len(mfcc) > max_frames:
        mfcc = mfcc[:max_frames]
    elif len(mfcc) < max_frames:
        mfcc = np.pad(mfcc, ((0, max_frames - len(mfcc)), (0, 0)), mode='constant')
    mfcc = scaler.transform(mfcc)
    mfcc = torch.from_numpy(mfcc).float().unsqueeze(0)
    with torch.no_grad():
        output = model(mfcc)
        probabilities = torch.softmax(output, dim=1).squeeze().numpy()
        _, pred = torch.max(output, 1)
    return pred.item(), probabilities.tolist()

@app.route('/predict', methods=['POST'])
def predict_digit():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(file_path)
        try:
            pred, probs_or_error = predict(file_path, model, scaler)
            os.remove(file_path)  # Clean up
            if pred is None:
                return jsonify({'error': probs_or_error}), 400
            return jsonify({
                'predicted_digit': pred,
                'probabilities': probs_or_error
            }), 200
        except Exception as e:
            os.remove(file_path)  # Clean up on error
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Invalid file format, only WAV files allowed'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))