import React from 'react';
import { useQuery } from '@tanstack/react-query';
import logo from './logo.svg';
import './App.css';

function App() {
  const { data: message, isLoading, isError } = useQuery({
    queryKey: ['hello-message'],
    queryFn: async () => {
      const res = await fetch('/api/hello');
      if (!res.ok) throw new Error('Erreur de connexion au backend');
      return res.text();
    },
  });

  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          {isLoading ? 'Chargement...' : isError ? 'Erreur de connexion au backend' : message}
        </p>
      </header>
    </div>
  );
}

export default App;
