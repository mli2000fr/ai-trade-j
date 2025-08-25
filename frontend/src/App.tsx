import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import ChatGptPage from './ChatGptPage';
import AnalyseActionPage from './AnalyseActionPage';
import logo from './logo.svg';
import './App.css';

function Home() {
  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Bienvenue sur l’interface ChatGPT. Rendez-vous sur <b>/chatgpt</b> pour interroger le modèle.
        </p>
      </header>
    </div>
  );
}

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<AnalyseActionPage />} />
        <Route path="/chatgpt" element={<ChatGptPage />} />
        <Route path="/analyse-action" element={<AnalyseActionPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
