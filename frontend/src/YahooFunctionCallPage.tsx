import React, { useState } from 'react';

const FUNCTION_NAMES = [
  'TIME_SERIES_INTRADAY',
  'TIME_SERIES_DAILY',
  'GLOBAL_QUOTE',
];

const YahooFunctionCallPage: React.FC = () => {
  const [symbol, setSymbol] = useState('');
  const [functionName, setFunctionName] = useState('TIME_SERIES_DAILY');
  const [result, setResult] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleCall = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await fetch('/api/chatgpt/function-call', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          functionName: 'getInfosAction',
          arguments: { symbol, functionName }
        })
      });
      if (!response.ok) {
        throw new Error('Erreur lors de la requête');
      }
      const data = await response.text();
      setResult(data);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 500, margin: '2rem auto', padding: 20, border: '1px solid #ccc', borderRadius: 8 }}>
      <h2>Interroger Alpha Vantage via Function Calling</h2>
      <div style={{ marginBottom: 12 }}>
        <label style={{ marginRight: 8 }}>Fonction :</label>
        <select value={functionName} onChange={e => setFunctionName(e.target.value)}>
          {FUNCTION_NAMES.map(fn => (
            <option key={fn} value={fn}>{fn}</option>
          ))}
        </select>
      </div>
      <input
        type="text"
        placeholder="Symbole boursier (ex: GLE.PA)"
        value={symbol}
        onChange={e => setSymbol(e.target.value)}
        style={{ width: '70%', marginRight: 8 }}
      />
      <button onClick={handleCall} disabled={loading || !symbol}>
        {loading ? 'Chargement...' : 'Obtenir le cours'}
      </button>
      {error && <div style={{ color: 'red', marginTop: 16 }}>{error}</div>}
      {result && (
        <pre style={{ marginTop: 16, background: '#f7f7f7', padding: 10, borderRadius: 4 }}>{result}</pre>
      )}
    </div>
  );
};

export default YahooFunctionCallPage;
