import React, { useEffect, useState } from 'react';
import { Box, Typography, LinearProgress, Paper, Divider, Button, CircularProgress } from '@mui/material';

const API_PROGRESS_URL = '/api/best-combination/monitor';

const MixStrategiesMonitorPage: React.FC = () => {
  const [progress, setProgress] = useState<any>(null);
  const [loading, setLoading] = useState<boolean | string>(true);

  const fetchProgress = async () => {
    setLoading(true);
    try {
      const res = await fetch(API_PROGRESS_URL);
      const data = await res.json();
      setProgress(data);
    } catch (e) {
      setProgress(null);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchProgress();
    const interval = setInterval(fetchProgress, 10000);
    return () => clearInterval(interval);
  }, []);

  const formatDuration = (start: number, end: number) => {
    if (!start) return '-';
    const ms = (end ? end : Date.now()) - start;
    const sec = Math.floor(ms / 1000) % 60;
    const min = Math.floor(ms / 60000) % 60;
    const hr = Math.floor(ms / 3600000);
    return `${hr}h ${min}m ${sec}s`;
  };
const callApi = async (endpoint: string, label: string) => {
    setLoading(label);
    try {
      const res = await fetch(`${endpoint}`);
      if (!res.ok) throw new Error('Erreur serveur');
    } catch (e) {
    } finally {
    }
  };
  if (!progress) return(<Button
            variant="contained"
            color="info"
            onClick={() => callApi('/api/best-combination/calcul', 'Calcule mix strategies')}
          >
            {loading === 'Calcule mix strategies' ? <CircularProgress size={24} /> : 'Calcule mix strategies'}
          </Button>);

  const percent = progress.totalSymbols > 0 ? Math.round(100 * progress.processedSymbols / progress.totalSymbols) : 0;

  return (
    <Box p={4}>
      <Paper elevation={3} sx={{ p: 3, maxWidth: 600, margin: '0 auto' }}>

        <Typography variant="h5" gutterBottom>Suivi du calcul des stratégies mixtes</Typography>
        <Divider sx={{ mb: 2 }} />
        <Typography>Status : <b>{progress.status}</b></Typography>
        <Typography>Progression : <b>{progress.processedSymbols} / {progress.totalSymbols}</b></Typography>
        <LinearProgress variant="determinate" value={percent} sx={{ my: 2 }} />
        <Typography>{percent}%</Typography>
        <Typography>Inserts réussis : <b>{progress.nbInsert}</b></Typography>
        <Typography>Erreurs : <b>{progress.error}</b></Typography>
        <Typography>Dernier symbole traité : <b>{progress.lastSymbol || '-'}</b></Typography>
        <Typography>Début : {progress.startTime ? new Date(progress.startTime).toLocaleString() : '-'}</Typography>
        <Typography>Fin : {progress.endTime ? new Date(progress.endTime).toLocaleString() : '-'}</Typography>
        <Typography>Durée : {formatDuration(progress.startTime, progress.endTime)}</Typography>
      </Paper>
    </Box>
  );
};

export default MixStrategiesMonitorPage;

