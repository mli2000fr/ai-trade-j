import React, { useState } from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import TextField from '@mui/material/TextField';
import Radio from '@mui/material/Radio';
import RadioGroup from '@mui/material/RadioGroup';
import FormControlLabel from '@mui/material/FormControlLabel';
import Button from '@mui/material/Button';
import Alert from '@mui/material/Alert';

interface TestSignalResult {
  signal: boolean;
}

const TestSignalForm: React.FC = () => {
  const [closePrices, setClosePrices] = useState<string>('');
  const [isEntry, setIsEntry] = useState<boolean>(true);
  const [result, setResult] = useState<TestSignalResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      // Supporte séparateur virgule, espace ou retour à la ligne
      const prices = closePrices
        .split(/[\s,;]+/)
        .map(s => s.trim())
        .filter(Boolean)
        .map(Number)
        .filter(n => !isNaN(n));
      if (prices.length < 2) {
        setError('Veuillez saisir au moins 2 prix de clôture valides.');
        setLoading(false);
        return;
      }
      const response = await fetch('/api/trade/strategies/test-signal', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ closePrices: prices, isEntry }),
      });
      if (!response.ok) throw new Error('Erreur API');
      const data: TestSignalResult = await response.json();
      setResult(data);
    } catch (err: any) {
      setError('Erreur lors du test du signal.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card sx={{ mb: 2, borderRadius: 2, boxShadow: 1, border: '1px solid #e0e0e0', bgcolor: 'background.paper' }}>
      <CardContent>
        <Typography variant="h6" sx={{ mb: 2 }}>Test du signal combiné</Typography>
        <form onSubmit={handleSubmit}>
          <Stack spacing={2}>
            <TextField
              label="Prix de clôture (séparés par virgule, espace ou retour à la ligne)"
              value={closePrices}
              onChange={e => setClosePrices(e.target.value)}
              multiline
              minRows={4}
              fullWidth
              size="small"
              placeholder="ex: 123.4, 124.1, 125.0, 126.2"
              disabled={loading}
            />
            <RadioGroup
              row
              value={isEntry ? 'entry' : 'exit'}
              onChange={e => setIsEntry(e.target.value === 'entry')}
            >
              <FormControlLabel value="entry" control={<Radio />} label="Signal d'entrée (achat)" disabled={loading} />
              <FormControlLabel value="exit" control={<Radio />} label="Signal de sortie (vente)" disabled={loading} />
            </RadioGroup>
            <Button type="submit" variant="contained" color="primary" disabled={loading}>
              {loading ? 'Test en cours...' : 'Tester le signal'}
            </Button>
            {result && (
              <Alert severity={result.signal ? 'success' : 'warning'} sx={{ fontWeight: 'bold' }}>
                Signal combiné : {result.signal ? 'VALIDÉ (true)' : 'NON VALIDÉ (false)'}
              </Alert>
            )}
            {error && <Alert severity="error">{error}</Alert>}
          </Stack>
        </form>
      </CardContent>
    </Card>
  );
};

export default TestSignalForm;
