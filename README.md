# Spoken Digit Recognition

A Flask-based API and web frontend for recognizing spoken digits (0-9) from WAV audio files using an LSTM neural network model. The frontend allows users to record a 2-second audio clip of a spoken digit, which is sent to the API for prediction. The API processes the audio, extracts MFCC features, and returns the predicted digit.

## Features

- **API**: Hosted on Render, processes 8 kHz, mono WAV files to predict spoken digits (0-9).
- **Frontend**: Records audio via the browser, trims silence, and displays the predicted digit.
- **CORS Support**: Configured to allow requests from `https://digitalrecog.techviq.com`.
- **Deployment**: API runs on Render's free tier, with a note about potential cold start delays.

## Project Structure

- `app.py`: Flask API with the `/predict` endpoint for digit recognition.
- `digit_model.pth`: Pre-trained LSTM model for digit classification.
- `scaler.pkl`: StandardScaler for normalizing MFCC features.
- `requirements.txt`: Python dependencies for the API.
- `index.html`: Frontend UI for recording audio and displaying predictions.

## Requirements

- **API**:
  - Python 3.8+
  - Dependencies listed in `requirements.txt` (e.g., Flask, PyTorch, NumPy, SciPy, scikit-learn, flask-cors).
  - Audio files must be 8 kHz, mono WAV files.
- **Frontend**:
  - Modern browser with Web Audio API support (e.g., Chrome, Firefox).
  - Hosted at `https://digitalrecog.techviq.com` or locally via a web server (e.g., XAMPP).

## Setup

### API

1. Clone the repository:
   ```bash
   git clone https://github.com/UsamaAmanat/digitrecognition.git
   cd digitrecognition
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the API locally:
   ```bash
   python app.py
   ```
   Or use `gunicorn` for production:
   ```bash
   gunicorn app:app
   ```
4. Test the API:
   ```bash
   curl -X POST -F "file=@path/to/audio.wav" http://localhost:5000/predict
   ```

### Frontend

1. Serve `index.html` via a web server:
   - **Local (XAMPP)**: Copy `index.html` to `C:\xampp\htdocs\ai` and access at `http://localhost/ai/index.html`.
   - **Hosted**: Deploy to a static site host (e.g., Render, GitHub Pages) or a server (e.g., cPanel for `https://digitalrecog.techviq.com`).
2. Open the URL in a browser, click "Record Digit," and speak a digit (0-9) for 2 seconds.

## Deployment

### API (Render)

- **URL**: `https://digitrecognition-lplk.onrender.com/predict`
- **Setup**:
  - Deployed as a Web Service on Render, linked to `https://github.com/UsamaAmanat/digitrecognition`.
  - **Build Command**: `pip install -r requirements.txt`
  - **Start Command**: `python app.py` (recommended: `gunicorn app:app` for production).
- **Note**: Hosted on Render's free tier, which may shut down if idle. Try multiple times if the API doesn't respond immediately.

### Frontend

- **URL**: `https://digitalrecog.techviq.com`
- **Setup**:
  - Deploy `index.html` as a static site (e.g., Render, GitHub Pages) or via a web server (e.g., cPanel, XAMPP).
  - Configured to send requests to `https://digitrecognition-lplk.onrender.com/predict`.

## Usage

1. Visit `https://digitalrecog.techviq.com`.
2. Click "Record Digit" and speak a digit (0-9) for 2 seconds.
3. Wait for the prediction (e.g., "Predicted Digit: 5"). A loading indicator shows during the API call.
4. Check the browser console (F12) for debug info (e.g., audio duration, probabilities).
5. **Note**: If the API is slow due to Render's free tier, try recording again.

## Troubleshooting

- **CORS Errors**:
  - Ensure `app.py` includes:
    ```python
    from flask_cors import CORS
    CORS(app, resources={r"/predict": {"origins": "https://digitalrecog.techviq.com"}})
    ```
  - Check Render logs for `flask-cors` installation.
- **API Errors**:
  - Verify audio is 8 kHz, mono WAV using:
    ```python
    from scipy.io import wavfile
    sample_rate, signal = wavfile.read("path/to/audio.wav")
    print(f"Sample rate: {sample_rate}, Channels: {signal.shape}")
    ```
  - Check Render logs for missing files (`digit_model.pth`, `scaler.pkl`) or runtime errors.
- **Cold Starts**:
  - Render's free tier may cause delays. Try multiple requests if the API is unresponsive.

## Future Enhancements

- Add a probability chart using Chart.js (toggleable).
- Implement API key authentication for security.
- Switch to `gunicorn` for production stability.
- Add a download button for recorded WAV files.


## Contact

For issues or suggestions, open an issue on GitHub or contact at usamaamanat38@gmail.com
