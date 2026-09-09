import { useState } from "react";
import axios from "axios";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFile = (selectedFile) => {
    if (!selectedFile) return;

    if (!selectedFile.type.startsWith("image/")) {
      setError("Please select an image file.");
      return;
    }

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setError("");
  };

  const handleChange = (event) => {
    handleFile(event.target.files[0]);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    handleFile(event.dataTransfer.files[0]);
  };

  const predictDisease = async () => {
    if (!file) {
      setError("Please select a leaf image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await axios.post(
        `${API_URL}/predict`,
        formData
      );

      setResult(response.data);
    } catch (err) {
      console.error(err);
      setError(
        "Could not connect to the AI backend. Make sure FastAPI is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError("");
  };

  return (
    <div className="app">
      <nav className="navbar">
        <div className="brand">
          <span className="brand-icon">🌿</span>
          <span>AI Plant Doctor</span>
        </div>

        <div className="status">
          <span className="status-dot"></span>
          AI Model Online
        </div>
      </nav>

      <main className="container">
        <section className="hero">
          <div className="badge">🧠 Deep Learning • EfficientNet-B0</div>

          <h1>
            Your Plant's
            <span> AI Doctor</span>
          </h1>

          <p>
            Upload a leaf image and let our deep learning model identify
            plant diseases in seconds.
          </p>
        </section>

        <section className="card">
          {!preview ? (
            <div
              className="upload-area"
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
            >
              <div className="upload-icon">📷</div>

              <h2>Upload a leaf image</h2>

              <p>
                Drag & drop an image here, or choose one from your device.
              </p>

              <label className="choose-button">
                Choose Image
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleChange}
                  hidden
                />
              </label>

              <small>JPG, JPEG, PNG • Recommended: clear leaf image</small>
            </div>
          ) : (
            <div className="prediction-section">
              <div className="image-container">
                <img src={preview} alt="Selected leaf" />
              </div>

              <div className="actions">
                <button
                  className="predict-button"
                  onClick={predictDisease}
                  disabled={loading}
                >
                  {loading ? "🧠 Analyzing..." : "🔍 Analyze Plant"}
                </button>

                <button className="reset-button" onClick={reset}>
                  Choose Another Image
                </button>
              </div>
            </div>
          )}

          {error && <div className="error">⚠️ {error}</div>}
        </section>

        {result && (
          <section className="result-card">
            <div className="result-header">
              <span className="success-icon">✓</span>

              <div>
                <p className="result-label">AI Diagnosis</p>
                <h2>{result.disease.replaceAll("_", " ")}</h2>
              </div>
            </div>

            <div className="result-grid">
              <div className="result-box">
                <span>🌱 Plant</span>
                <strong>{result.plant}</strong>
              </div>

              <div className="result-box">
                <span>🎯 Confidence</span>
                <strong>{Number(result.confidence).toFixed(2)}%</strong>

                <div className="confidence-bar">
                  <div
                    style={{
                      width: `${Math.min(result.confidence, 100)}%`,
                    }}
                  ></div>
                </div>
              </div>

              <div className="result-box">
                <span>🧠 Model</span>
                <strong>{result.model}</strong>
              </div>
            </div>

            <div className="info-box">
  <h3>🩺 Treatment Recommendation</h3>

  {result?.treatment ? (
    <>
      <p>
        <strong>Severity:</strong>{" "}
        {result.treatment.severity || "Unknown"}
      </p>

      <h4>🌿 Treatment</h4>
      <ul>
        {Array.isArray(result.treatment.treatment) &&
          result.treatment.treatment.map((item, index) => (
            <li key={index}>{item}</li>
          ))}
      </ul>

      <h4>🛡️ Prevention</h4>
      <ul>
        {Array.isArray(result.treatment.prevention) &&
          result.treatment.prevention.map((item, index) => (
            <li key={index}>{item}</li>
          ))}
      </ul>
    </>
  ) : (
    <p>No treatment information available for this disease yet.</p>
  )}
</div>
 <div className="info-box">
              ...your treatment section...
            </div>

            {result.gradcam_url && (
              <div className="info-box">
                <h3>🔥 Grad-CAM Explanation</h3>

                <p>
                  The highlighted areas show which parts of the leaf influenced
                  the AI prediction.
                </p>

                <img
                  src={`http://127.0.0.1:8000${result.gradcam_url}?t=${Date.now()}`}
                  alt="Grad-CAM explanation"
                  style={{
                    width: "100%",
                    maxWidth: "600px",
                    borderRadius: "12px",
                    display: "block",
                    margin: "16px auto"
                  }}
                />
              </div>
            )}
          </section>
        )}
      </main>

      <footer>
        <p>
          AI Plant Doctor • Deep Learning Plant Disease Detection
        </p>
        <span>EfficientNet-B0 • 38 Classes • PyTorch</span>
      </footer>
    </div>
  );
}

export default App;
