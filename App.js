import React, { useState } from 'react';
import './App.css';

function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);

  const handlePredict = async () => {
    const response = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    const data = await response.json();
    setResult(data);
  };

  return (
    <div className="App">
      <h2>Sentiment Analysis</h2>
      <textarea
        rows="4"
        cols="50"
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <br />
      <button onClick={handlePredict}>Predict</button>
      {result && (
        <div>
          <p><strong>Label:</strong> {result.label}</p>
          <p><strong>Score:</strong> {result.score.toFixed(3)}</p>
        </div>
      )}
    </div>
  );
}

export default App;
