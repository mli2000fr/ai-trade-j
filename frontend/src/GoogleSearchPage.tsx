import React, { useState } from 'react';

const GoogleSearchPage: React.FC = () => {
  const [query, setQuery] = useState('');
  const [result, setResult] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await fetch(`/api/search?query=${encodeURIComponent(query)}`);
      if (!response.ok) {
        throw new Error('Erreur lors de la requête');
      }
      const text = await response.text();
      setResult(text);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 600, margin: '0 auto', padding: 20 }}>
      <h2>Recherche Google Custom Search</h2>
      <input
        type="text"
        value={query}
        onChange={e => setQuery(e.target.value)}
        placeholder="Entrez votre requête"
        style={{ width: '70%', marginRight: 8 }}
      />
      <button onClick={handleSearch} disabled={loading || !query}>
        {loading ? 'Recherche...' : 'Search'}
      </button>
      {error && <div style={{ color: 'red', marginTop: 10 }}>{error}</div>}
      {result && (
        <pre style={{ marginTop: 20, background: '#f4f4f4', padding: 10, borderRadius: 4 }}>
          {result}
        </pre>
      )}
    </div>
  );
};

export default GoogleSearchPage;

