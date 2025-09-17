import React, { useState } from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import Alert from '@mui/material/Alert';
import Button from '@mui/material/Button';
import CircularProgress from '@mui/material/CircularProgress';

const StrategyManagerBlock: React.FC = () => {
  const [loading, setLoading] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const callApi = async (endpoint: string, label: string) => {
    setLoading(label);
    setSuccess(null);
    setError(null);
    try {
      const res = await fetch(`${endpoint}`);
      if (!res.ok) throw new Error('Erreur serveur');
      setSuccess(`${label} : succès !`);
    } catch (e) {
      setError(`${label} : échec`);
    } finally {
      setLoading(null);
    }
  };

  return (
    <Card sx={{ mb: 2 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Actions rapides sur la base de données
        </Typography>
        <Stack direction="row" spacing={2}>
          <Button
            variant="contained"
            color="primary"
            onClick={() => callApi('/api/stra/strategies/db/update-assets', 'Update Assets')}
            disabled={loading !== null}
          >
            {loading === 'Update Assets' ? <CircularProgress size={24} /> : 'Update Assets'}
          </Button>
          <Button
            variant="contained"
            color="secondary"
            onClick={() => callApi('/api/stra/strategies/db/update-daily-valu', 'Update Daily Value')}
            disabled={loading !== null}
          >
            {loading === 'Update Daily Value' ? <CircularProgress size={24} /> : 'Update Daily Value'}
          </Button>
        </Stack>
        {success && <Alert severity="success" sx={{ mt: 2 }}>{success}</Alert>}
        {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}
      </CardContent>
    </Card>
  );
};

export default StrategyManagerBlock;
