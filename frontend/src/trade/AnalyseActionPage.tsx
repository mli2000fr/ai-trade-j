import React, { useState } from 'react';

const AnalyseActionPage: React.FC = () => {
  const [symbol, setSymbol] = useState('GLE.PA');
  const [result, setResult] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyse = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await fetch(`/api/trade/analyse-action?symbol=${encodeURIComponent(symbol)}`);
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
      <h2>Analyse d'une action (ChatGPT)</h2>
      <div style={{ marginBottom: 12 }}>
        <input
          type="text"
          placeholder="Symbole boursier (ex: GLE.PA)"
          value={symbol}
          onChange={e => setSymbol(e.target.value)}
          style={{ width: '60%', marginRight: 8 }}
        />
        <button onClick={handleAnalyse} disabled={loading || !symbol}>
          {loading ? 'Analyse en cours...' : 'Analyser'}
        </button>
      </div>
      {error && <div style={{ color: 'red', marginTop: 16 }}>{error}</div>}
      {result && (
        <pre style={{ marginTop: 16, background: '#f7f7f7', padding: 10, borderRadius: 4, whiteSpace: 'pre-wrap' }}>{result}</pre>
      )}
    </div>
  );
};

export default AnalyseActionPage;
