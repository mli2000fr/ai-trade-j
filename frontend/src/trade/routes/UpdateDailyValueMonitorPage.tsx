import React, { useEffect, useState } from 'react';
import { Box, Typography, Button, CircularProgress, Paper, Divider, Chip } from '@mui/material';
import LinearProgress from '@mui/material/LinearProgress';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import HourglassTopIcon from '@mui/icons-material/HourglassTop';

const API_PROGRESS_URL = '/api/stra/strategies/daily_value/progress';

const UpdateDailyValueMonitorPage: React.FC = () => {
  const [progress, setProgress] = useState<any>({});
  const [loading, setLoading] = useState<boolean | string>(true);

  const fetchAll = async () => {
    setLoading(true);
    try {
      const res = await fetch(API_PROGRESS_URL);
      const progressData = await res.json();
      setProgress(progressData || {});
    } catch (e) {
      setProgress({});
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
      // Le polling est déjà géré par useEffect
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

  const isUpdateEnCours = progress.status === 'en_cours';

  return (
    <>
      {!isUpdateEnCours && (
        <Box p={4} textAlign="center">
          <Typography variant="h6" gutterBottom>Aucune mise à jour en cours</Typography>
          <Box display="flex" justifyContent="center" gap={2}>
            <Button
              variant="contained"
              color="primary"
              disabled={true}
              /*disabled={loading !== true && loading !== false}*/
              onClick={() => callApi('/api/stra/strategies/db/update-assets', 'Update Assets')}
            >
              {loading === 'Update Assets' ? <CircularProgress size={24} /> : 'Update Assets'}
            </Button>
            <Button
              variant="contained"
              color="secondary"
              disabled={true}
              onClick={() => callApi('/api/stra/strategies/db/update-daily-valu', 'Update Daily Value')}
              /*disabled={loading !== true && loading !== false}*/
            >
              {loading === 'Update Daily Value' ? <CircularProgress size={24} /> : 'Lancer la mise à jour Daily Value'}
            </Button>
          </Box>
        </Box>
      )}
      <Box p={4}>
        <Paper elevation={3} sx={{ p: 3, maxWidth: 700, margin: '0 auto' }}>
          <Box display="flex" alignItems="center" gap={1} mb={2}>
            {progress.status === 'termine' ? <CheckCircleIcon color="success" /> : progress.status === 'erreur' ? <ErrorIcon color="error" /> : <HourglassTopIcon color="info" />}
            <Typography variant="h5" gutterBottom>Suivi de la mise à jour Daily Value</Typography>
          </Box>
          <Divider sx={{ mb: 2 }} />
          <Box mb={3}>
            <Typography variant="subtitle1" gutterBottom><b>{progress.symbol || progress.name || 'Tâche'}</b></Typography>
            <LinearProgress variant="determinate" value={progress.totalItems > 0 ? Math.round(100 * progress.updatedItems / progress.totalItems) : 0} sx={{ height: 10, borderRadius: 5 }} />
            <Typography align="center" mt={1}>{progress.totalItems > 0 ? Math.round(100 * progress.updatedItems / progress.totalItems) : 0}%</Typography>
            <Box display="flex" justifyContent="space-between" mt={2} mb={1}>
              <Typography>Avancement :</Typography>
              <Typography><b>{progress.updatedItems} / {progress.totalItems}</b></Typography>
            </Box>
            <Box display="flex" justifyContent="space-between" mb={1}>
              <Typography>Statut :</Typography>
              <Chip label={progress.status === 'en_cours' ? 'En cours' : progress.status === 'termine' ? 'Terminé' : progress.status === 'erreur' ? 'Erreur' : ''} color={progress.status === 'termine' ? 'success' : progress.status === 'erreur' ? 'error' : 'info'} size="small" />
            </Box>
            <Box display="flex" justifyContent="space-between" mb={1}>
              <Typography>Durée :</Typography>
              <Typography>{formatDuration(progress.startTime, progress.status === 'termine' ? progress.endTime : progress.lastUpdate)}</Typography>
            </Box>
            <Divider sx={{ my: 2 }} />
          </Box>
        </Paper>
      </Box>
    </>
  );
};

export default UpdateDailyValueMonitorPage;
