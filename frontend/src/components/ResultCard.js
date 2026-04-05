import React from 'react';

function ResultCard({ result }) {
  const riskColors = {
    High: { bg: '#fff0f0', border: '#e74c3c', text: '#c0392b', label: 'HIGH RISK' },
    Medium: { bg: '#fffbf0', border: '#f39c12', text: '#d68910', label: 'MEDIUM RISK' },
    Low: { bg: '#f0fff4', border: '#27ae60', text: '#1e8449', label: 'LOW RISK' }
  };

  const colors = riskColors[result.risk_level];

  const getBarWidth = (value) => `${(value * 100).toFixed(1)}%`;

  const barColors = {
    Dropout: '#e74c3c',
    Enrolled: '#f39c12',
    Graduate: '#27ae60'
  };

  return (
    <div className="result-card" style={{ borderLeft: `6px solid ${colors.border}`, background: colors.bg }}>
      <div className="result-header">
        <div>
          <div className="result-prediction">{result.prediction}</div>
          <div className="result-subtitle">Predicted Outcome</div>
        </div>
        <div className="risk-badge" style={{ background: colors.border, color: 'white' }}>
          {colors.label}
        </div>
      </div>

      <div className="confidence-section">
        <div className="confidence-title">Prediction Confidence</div>
        {Object.entries(result.confidence).map(([cls, val]) => (
          <div key={cls} className="confidence-row">
            <span className="confidence-label">{cls}</span>
            <div className="confidence-bar-bg">
              <div
                className="confidence-bar-fill"
                style={{ width: getBarWidth(val), background: barColors[cls] }}
              />
            </div>
            <span className="confidence-pct">{(val * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>

      <div className="result-advice" style={{ borderTop: `1px solid ${colors.border}` }}>
        {result.risk_level === 'High' && (
          <p>This student is predicted to drop out. Immediate advisor intervention is recommended. Review attendance, academic support options, and financial aid eligibility.</p>
        )}
        {result.risk_level === 'Medium' && (
          <p>This student's outcome is uncertain. Schedule a check-in with the student to assess academic progress and identify any support needs before the semester ends.</p>
        )}
        {result.risk_level === 'Low' && (
          <p>This student is on track to graduate. No immediate intervention required. Continue monitoring standard academic milestones.</p>
        )}
      </div>
    </div>
  );
}

export default ResultCard;
