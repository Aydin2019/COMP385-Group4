import React, { useState } from 'react';
import './App.css';
import PredictionForm from './components/PredictionForm';
import ResultCard from './components/ResultCard';
import Header from './components/Header';

function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);

  const handlePredict = async (formData) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      const data = await response.json();
      if (data.error) {
        setError(data.error);
      } else {
        setResult(data);
        setHistory(prev => [{ ...data, studentName: formData.studentName || 'Student', timestamp: new Date().toLocaleTimeString() }, ...prev.slice(0, 4)]);
      }
    } catch (err) {
      setError('Could not connect to the prediction server. Make sure the Flask API is running.');
    }
    setLoading(false);
  };

  return (
    <div className="app">
      <Header />
      <div className="main-content">
        <div className="left-panel">
          <PredictionForm onPredict={handlePredict} loading={loading} />
        </div>
        <div className="right-panel">
          {error && <div className="error-box">{error}</div>}
          {result && <ResultCard result={result} />}
          {history.length > 0 && (
            <div className="history-section">
              <h3>Recent Predictions</h3>
              {history.map((item, i) => (
                <div key={i} className={`history-item risk-${item.risk_level.toLowerCase()}`}>
                  <span className="history-name">{item.studentName}</span>
                  <span className="history-prediction">{item.prediction}</span>
                  <span className="history-time">{item.timestamp}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
