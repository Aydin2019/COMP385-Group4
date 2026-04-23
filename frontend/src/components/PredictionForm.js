import React, { useState } from 'react';

const defaultForm = {
  studentName: '',
  'Age at enrollment': 20,
  'Gender': 1,
  'Scholarship holder': 0,
  'Debtor': 0,
  'Tuition fees up to date': 1,
  'International': 0,
  'Curricular units 1st sem (enrolled)': 6,
  'Curricular units 1st sem (approved)': 5,
  'Curricular units 1st sem (grade)': 12.0,
  'Curricular units 2nd sem (enrolled)': 6,
  'Curricular units 2nd sem (approved)': 5,
  'Curricular units 2nd sem (grade)': 12.0,
  'Displaced': 0,
  'Educational special needs': 0
};

function PredictionForm({ onPredict, loading }) {
  const [form, setForm] = useState(defaultForm);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm(prev => ({ ...prev, [name]: name === 'studentName' ? value : parseFloat(value) }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const { studentName, ...features } = form;
    onPredict({ ...features, studentName });
  };

  const handleReset = () => setForm(defaultForm);

  return (
    <div className="form-card">
      <h2>Student Information</h2>
      <p className="form-subtitle">Enter the student's academic data to generate a risk prediction.</p>
      <form onSubmit={handleSubmit}>

        <div className="form-group">
          <label>Student Name (for reference)</label>
          <input type="text" name="studentName" value={form.studentName} onChange={handleChange} placeholder="e.g. John Smith" />
        </div>

        <div className="form-section-title">Demographics</div>
        <div className="form-row">
          <div className="form-group">
            <label>Age at Enrollment</label>
            <input type="number" name="Age at enrollment" value={form['Age at enrollment']} onChange={handleChange} min="17" max="70" />
          </div>
          <div className="form-group">
            <label>Gender</label>
            <select name="Gender" value={form['Gender']} onChange={handleChange}>
              <option value={1}>Male</option>
              <option value={0}>Female</option>
            </select>
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label>Scholarship Holder</label>
            <select name="Scholarship holder" value={form['Scholarship holder']} onChange={handleChange}>
              <option value={0}>No</option>
              <option value={1}>Yes</option>
            </select>
          </div>
          <div className="form-group">
            <label>International Student</label>
            <select name="International" value={form['International']} onChange={handleChange}>
              <option value={0}>No</option>
              <option value={1}>Yes</option>
            </select>
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label>Tuition Fees Up To Date</label>
            <select name="Tuition fees up to date" value={form['Tuition fees up to date']} onChange={handleChange}>
              <option value={1}>Yes</option>
              <option value={0}>No</option>
            </select>
          </div>
          <div className="form-group">
            <label>Has Outstanding Debt</label>
            <select name="Debtor" value={form['Debtor']} onChange={handleChange}>
              <option value={0}>No</option>
              <option value={1}>Yes</option>
            </select>
          </div>
        </div>

        <div className="form-section-title">1st Semester Performance</div>
        <div className="form-row">
          <div className="form-group">
            <label>Units Enrolled</label>
            <input type="number" name="Curricular units 1st sem (enrolled)" value={form['Curricular units 1st sem (enrolled)']} onChange={handleChange} min="0" max="26" />
          </div>
          <div className="form-group">
            <label>Units Approved</label>
            <input type="number" name="Curricular units 1st sem (approved)" value={form['Curricular units 1st sem (approved)']} onChange={handleChange} min="0" max="26" />
          </div>
          <div className="form-group">
            <label>Average Grade (0–20)</label>
            <input type="number" name="Curricular units 1st sem (grade)" value={form['Curricular units 1st sem (grade)']} onChange={handleChange} min="0" max="20" step="0.1" />
          </div>
        </div>

        <div className="form-section-title">2nd Semester Performance</div>
        <div className="form-row">
          <div className="form-group">
            <label>Units Enrolled</label>
            <input type="number" name="Curricular units 2nd sem (enrolled)" value={form['Curricular units 2nd sem (enrolled)']} onChange={handleChange} min="0" max="26" />
          </div>
          <div className="form-group">
            <label>Units Approved</label>
            <input type="number" name="Curricular units 2nd sem (approved)" value={form['Curricular units 2nd sem (approved)']} onChange={handleChange} min="0" max="26" />
          </div>
          <div className="form-group">
            <label>Average Grade (0–20)</label>
            <input type="number" name="Curricular units 2nd sem (grade)" value={form['Curricular units 2nd sem (grade)']} onChange={handleChange} min="0" max="20" step="0.1" />
          </div>
        </div>

        <div className="form-actions">
          <button type="submit" className="btn-predict" disabled={loading}>
            {loading ? 'Predicting...' : 'Generate Prediction'}
          </button>
          <button type="button" className="btn-reset" onClick={handleReset}>Reset</button>
        </div>
      </form>
    </div>
  );
}

export default PredictionForm;
