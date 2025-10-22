import React, { useEffect, useState } from 'react';
import { Box, Typography, Button, CircularProgress, Paper, Divider, Chip } from '@mui/material';
import LinearProgress from '@mui/material/LinearProgress';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import HourglassTopIcon from '@mui/icons-material/HourglassTop';

const API_MONITOR_URL = '/api/result/symbol-buy/monitor';
const API_RUN_URL = '/api/result/getSymbolBuy';

const SymbolBuyMonitorPage: React.FC = () => {
  const [executedCount, setExecutedCount] = useState<number>(0);
  const [loading, setLoading] = useState<boolean | string>(false);
  const [result, setResult] = useState<string>('');
  const [error, setError] = useState<string>('');

  const fetchMonitor = async () => {
    try {
      const res = await fetch(API_MONITOR_URL);
      const count = await res.json();
      setExecutedCount(count);
    } catch (e) {
      setExecutedCount(0);
    }
  };

  useEffect(() => {
    fetchMonitor();
    const interval = setInterval(fetchMonitor, 10000);
    return () => clearInterval(interval);
  }, []);

  const runSymbolBuy = async () => {
    setLoading(true);
    setError('');
    setResult('');
    try {
      const res = await fetch(API_RUN_URL);
      if (!res.ok) throw new Error('Erreur serveur');
      const data = await res.text();
      setResult(data);
    } catch (e: any) {
      setError(e.message || 'Erreur inconnue');
    }
    setLoading(false);
    fetchMonitor();
  };

  return (
    <Box maxWidth={600} mx="auto" mt={4}>
      <Paper elevation={3} sx={{ p: 3 }}>
        <Typography variant="h5" gutterBottom>
          Monitoring Symbol Buy
        </Typography>
        <Divider sx={{ mb: 2 }} />
        <Box mb={2}>
          <Typography variant="body1">
            Nombre de symboles exécutés : <b>{executedCount}</b>
          </Typography>
        </Box>
        <Box display="flex" justifyContent="center" alignItems="center" mb={2}>
          <Button variant="contained" color="primary" onClick={runSymbolBuy} disabled={!!loading} sx={{ minWidth: 220, minHeight: 48 }}>
            {loading ? <CircularProgress size={24} /> : 'Lancer getSymbolBuy()'}
          </Button>
        </Box>
        <Box mt={3} display="flex" justifyContent="center" alignItems="center">
          {result && (
            <Chip icon={<CheckCircleIcon color="success" />} label={`Résultat : ${result}`} color="success" />
          )}
          {error && (
            <Chip icon={<ErrorIcon color="error" />} label={`Erreur : ${error}`} color="error" />
          )}
        </Box>
      </Paper>
    </Box>
  );
};

export default SymbolBuyMonitorPage;
