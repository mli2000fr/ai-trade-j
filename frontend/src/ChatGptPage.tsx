import React, { useState } from 'react';

const ChatGptPage: React.FC = () => {
  const [question, setQuestion] = useState('');
  const [response, setResponse] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleAsk = async () => {
    setLoading(true);
    setResponse(null);
    setError(null);
    try {
      const res = await fetch('/api/chatgpt/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        // Correction : envoyer un objet JSON avec le champ "prompt"
        body: JSON.stringify({ prompt: question }),
      });
      const data = await res.json();
      if (res.ok && data.message) {
        setResponse(data.message);
      } else {
        setError(data.error || 'Erreur inconnue');
      }
    } catch (e: any) {
      setError('Erreur réseau ou serveur.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 600, margin: '40px auto', padding: 24, border: '1px solid #ccc', borderRadius: 8 }}>
      <h2>Interroger ChatGPT</h2>
      <input
        type="text"
        value={question}
        onChange={e => setQuestion(e.target.value)}
        placeholder="Votre question..."
        style={{ width: '80%', padding: 8, marginRight: 8 }}
        onKeyDown={e => { if (e.key === 'Enter') handleAsk(); }}
        disabled={loading}
      />
      <button onClick={handleAsk} disabled={loading || !question.trim()}>
        {loading ? 'Envoi...' : 'Envoyer'}
      </button>
      <div style={{ marginTop: 24, minHeight: 40 }}>
        {response && (
          <div style={{ color: 'green' }}><b>Réponse :</b> {response}</div>
        )}
        {error && (
          <div style={{ color: 'red' }}><b>Erreur :</b> {error}</div>
        )}
      </div>
    </div>
  );
};

export default ChatGptPage;
