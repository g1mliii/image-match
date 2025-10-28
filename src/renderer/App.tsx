import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

const App: React.FC = () => {
  return (
    <Router>
      <div className="app">
        <header className="app-header">
          <h1>Product Matching System</h1>
        </header>
        <main className="app-main">
          <Routes>
            <Route path="/" element={<div>Welcome to Product Matching System</div>} />
          </Routes>
        </main>
      </div>
    </Router>
  );
};

export default App;
