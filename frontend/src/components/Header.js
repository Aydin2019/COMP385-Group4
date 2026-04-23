import React from 'react';

function Header() {
  return (
    <div className="header">
      <div className="header-title">
        <h1>Student Risk Prediction Dashboard</h1>
        <p>AI-Driven Academic Outcome Prediction System — COMP385 Group 4</p>
      </div>
      <div className="header-badge">
        <span className="badge">Gradient Boosting Model</span>
        <span className="badge">Macro F1: 0.70</span>
        <span className="badge">ROC-AUC: 0.89</span>
      </div>
    </div>
  );
}

export default Header;
