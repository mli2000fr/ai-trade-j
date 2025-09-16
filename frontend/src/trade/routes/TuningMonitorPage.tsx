import React, { useEffect, useState } from 'react';
import { Box, Typography, Button, CircularProgress, Paper, Divider, Chip } from '@mui/material';
import DownloadIcon from '@mui/icons-material/Download';
import LinearProgress from '@mui/material/LinearProgress';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import HourglassTopIcon from '@mui/icons-material/HourglassTop';

const API_PROGRESS_URL = '/api/tuning/progress';

const TuningMonitorPage: React.FC = () => {
  const [progress, setProgress] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean | string>(true);

  const fetchAll = async () => {
    setLoading(true);
    try {
      const [progressRes] = await Promise.all([
        fetch(API_PROGRESS_URL)
      ]);
      const progressData = await progressRes.json();
      setProgress(progressData);
    } catch (e) {
      setProgress([]);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchAll();
    const interval = setInterval(fetchAll, 10000);
    return () => clearInterval(interval);
  }, []);

  const callApi = async (endpoint: string, label: string) => {
    setLoading(label);
    try {
      const res = await fetch(`${endpoint}`);
      if (!res.ok) throw new Error('Erreur serveur');
    } catch (e) {
    } finally {
    }
  };
  const formatDuration = (start: number, end: number) => {
    if (!start) return '-';
    const ms = (end ? end : Date.now()) - start;
    const sec = Math.floor(ms / 1000) % 60;
    const min = Math.floor(ms / 60000) % 60;
    const hr = Math.floor(ms / 3600000);
    return `${hr}h ${min}m ${sec}s`;
  };

  // Vérifie si un tuning est en cours
  const isTuningEnCours = progress.some((row: any) => row.status === 'en_cours');

  if (!isTuningEnCours) {
    if(loading){
        return (
            <Box p={4} textAlign="center">
                <CircularProgress size={40} />
            </Box>
        );
    }else{
        return (
            <Box p={4} textAlign="center">
                <Typography variant="h6" gutterBottom>Aucun tuning en cours</Typography>
                <Button
                    variant="contained"
                    color="info"
                    onClick={() => callApi('/api/lstm/tuneAllSymbols', 'Calcule hyper params LSTM')}
                    disabled={loading === 'Calcule hyper params LSTM'}
                    sx={{ mt: 2 }}
                >
                    {loading === 'Calcule hyper params LSTM' ? <CircularProgress size={24} /> : 'Lancer le tuning LSTM'}
                </Button>
            </Box>
        );
    }
  }

  return (
    <Box p={4}>
      <Paper elevation={3} sx={{ p: 3, maxWidth: 700, margin: '0 auto' }}>
        <Box display="flex" alignItems="center" gap={1} mb={2}>
          {progress[0]?.status === 'termine' ? <CheckCircleIcon color="success" /> : progress[0]?.status === 'erreur' ? <ErrorIcon color="error" /> : <HourglassTopIcon color="info" />}
          <Typography variant="h5" gutterBottom>Suivi du tuning LSTM</Typography>
          <Chip label={progress[0]?.status || ''} color={progress[0]?.status === 'termine' ? 'success' : progress[0]?.status === 'erreur' ? 'error' : 'info'} size="small" sx={{ ml: 2 }} />
        </Box>
        <Divider sx={{ mb: 2 }} />
        {progress.map((row: any, idx: number) => {
          const percent = row.totalConfigs > 0 ? Math.round(100 * row.testedConfigs / row.totalConfigs) : 0;
          return (
            <Box key={row.symbol} mb={3}>
              <Typography variant="subtitle1" gutterBottom><b>{row.symbol}</b></Typography>
              <LinearProgress variant="determinate" value={percent} sx={{ height: 10, borderRadius: 5 }} />
              <Typography align="center" mt={1}>{percent}%</Typography>
              <Box display="flex" justifyContent="space-between" mt={2} mb={1}>
                <Typography>Avancement :</Typography>
                <Typography><b>{row.testedConfigs} / {row.totalConfigs}</b></Typography>
              </Box>
              <Box display="flex" justifyContent="space-between" mb={1}>
                <Typography>Statut :</Typography>
                <Chip label={row.status === 'en_cours' ? 'En cours' : row.status === 'termine' ? 'Terminé' : 'Erreur'} color={row.status === 'termine' ? 'success' : row.status === 'erreur' ? 'error' : 'info'} size="small" />
              </Box>
              <Box display="flex" justifyContent="space-between" mb={1}>
                <Typography>Durée :</Typography>
                <Typography>{formatDuration(row.startTime, row.status === 'termine' ? row.endTime : row.lastUpdate)}</Typography>
              </Box>
              <Divider sx={{ my: 2 }} />
            </Box>
          );
        })}
        {(progress[0]?.status === 'termine' || progress[0]?.status === 'erreur') && (
          <Button
            variant="contained"
            color="info"
            onClick={() => callApi('/api/lstm/tuneAllSymbols', 'Relancer le tuning')}
            sx={{ mt: 2 }}
          >
            {loading === 'Relancer le tuning' ? <CircularProgress size={24} /> : 'Relancer le tuning'}
          </Button>
        )}
      </Paper>
    </Box>
  );
};

export default TuningMonitorPage;
