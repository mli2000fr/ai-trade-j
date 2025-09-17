import React, { useEffect, useState } from 'react';
import { Box, Typography, Button, CircularProgress, Paper, Divider, Chip } from '@mui/material';
import LinearProgress from '@mui/material/LinearProgress';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import HourglassTopIcon from '@mui/icons-material/HourglassTop';

const API_PROGRESS_URL = '/api/stra/strategies/croised/progress';

const CroisedStrategiesMonitorPage: React.FC = () => {
  const [progress, setProgress] = useState<any>({});
  const [loading, setLoading] = useState<boolean | string>(true);

  const fetchAll = async () => {
    setLoading(true);
    try {
      const [progressRes] = await Promise.all([
        fetch(API_PROGRESS_URL)
      ]);
      const progressData = await progressRes.json();
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
      // On ne relance pas le polling ici, il est déjà géré par useEffect
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

  // Vérifie si un calcul croisé est en cours
  const isCroisedEnCours = progress.status === 'en_cours';

  return (<>
             {!isCroisedEnCours && (<Box p={4} textAlign="center">
                  <Typography variant="h6" gutterBottom>Aucun calcul croisé en cours</Typography>
                  <Button
                      variant="contained"
                      color="info"
                      onClick={() => callApi('/api/stra/strategies/calcul_croised_strategies', 'Calcule croised strategies')}
                      sx={{ mt: 2 }}
                  >
                      {loading === 'Calcule croised strategies' ? <CircularProgress size={24} /> : 'Lancer le calcul croisé'}
                  </Button>
              </Box>)}
              <Box p={4}>
                    <Paper elevation={3} sx={{ p: 3, maxWidth: 700, margin: '0 auto' }}>
                      <Box display="flex" alignItems="center" gap={1} mb={2}>
                        {progress.status === 'termine' ? <CheckCircleIcon color="success" /> : progress.status === 'erreur' ? <ErrorIcon color="error" /> : <HourglassTopIcon color="info" />}
                        <Typography variant="h5" gutterBottom>Suivi du calcul croisé des stratégies</Typography>
                      </Box>
                      <Divider sx={{ mb: 2 }} />
                      <Box mb={3}>
                        <Typography variant="subtitle1" gutterBottom><b>{progress.symbol || progress.name || 'Tâche'}</b></Typography>
                        <LinearProgress variant="determinate" value={progress.totalConfigs > 0 ? Math.round(100 * progress.testedConfigs / progress.totalConfigs) : 0} sx={{ height: 10, borderRadius: 5 }} />
                        <Typography align="center" mt={1}>{progress.totalConfigs > 0 ? Math.round(100 * progress.testedConfigs / progress.totalConfigs) : 0}%</Typography>
                        <Box display="flex" justifyContent="space-between" mt={2} mb={1}>
                          <Typography>Avancement :</Typography>
                          <Typography><b>{progress.testedConfigs} / {progress.totalConfigs}</b></Typography>
                        </Box>
                        <Box display="flex" justifyContent="space-between" mb={1}>
                          <Typography>Statut :</Typography>
                          <Chip label={progress.status === 'en_cours' ? 'En cours' : progress.status === 'termine' ? 'Terminé' : ''} color={progress.status === 'termine' ? 'success' : progress.status === 'erreur' ? 'error' : 'info'} size="small" />
                        </Box>
                        <Box display="flex" justifyContent="space-between" mb={1}>
                          <Typography>Durée :</Typography>
                          <Typography>{formatDuration(progress.startTime, progress.status === 'termine' ? progress.endTime : progress.lastUpdate)}</Typography>
                        </Box>
                      </Box>
                    </Paper>
                  </Box>
                  </>
          );

  return (
    <Box p={4}>
      <Paper elevation={3} sx={{ p: 3, maxWidth: 700, margin: '0 auto' }}>
        <Box display="flex" alignItems="center" gap={1} mb={2}>
          {progress.status === 'termine' ? <CheckCircleIcon color="success" /> : progress.status === 'erreur' ? <ErrorIcon color="error" /> : <HourglassTopIcon color="info" />}
          <Typography variant="h5" gutterBottom>Suivi du calcul croisé des stratégies</Typography>
          <Chip label={progress.status || ''} color={progress.status === 'termine' ? 'success' : progress.status === 'erreur' ? 'error' : 'info'} size="small" sx={{ ml: 2 }} />
        </Box>
        <Divider sx={{ mb: 2 }} />
        <Box mb={3}>
          <Typography variant="subtitle1" gutterBottom><b>{progress.symbol || progress.name || 'Tâche'}</b></Typography>
          <LinearProgress variant="determinate" value={progress.totalConfigs > 0 ? Math.round(100 * progress.testedConfigs / progress.totalConfigs) : 0} sx={{ height: 10, borderRadius: 5 }} />
          <Typography align="center" mt={1}>{progress.totalConfigs > 0 ? Math.round(100 * progress.testedConfigs / progress.totalConfigs) : 0}%</Typography>
          <Box display="flex" justifyContent="space-between" mt={2} mb={1}>
            <Typography>Avancement :</Typography>
            <Typography><b>{progress.testedConfigs} / {progress.totalConfigs}</b></Typography>
          </Box>
          <Box display="flex" justifyContent="space-between" mb={1}>
            <Typography>Statut :</Typography>
            <Chip label={progress.status === 'en_cours' ? 'En cours' : progress.status === 'termine' ? 'Terminé' : 'Erreur'} color={progress.status === 'termine' ? 'success' : progress.status === 'erreur' ? 'error' : 'info'} size="small" />
          </Box>
          <Box display="flex" justifyContent="space-between" mb={1}>
            <Typography>Durée :</Typography>
            <Typography>{formatDuration(progress.startTime, progress.status === 'termine' ? progress.endTime : progress.lastUpdate)}</Typography>
          </Box>
          <Divider sx={{ my: 2 }} />
        </Box>
        {(progress.status === 'termine' || progress.status === 'erreur') && (
          <Button
            variant="contained"
            color="success"
            onClick={() => callApi('/api/stra/strategies/calcul_croised_strategies', 'Relancer le calcul croisé')}
            sx={{ mt: 2 }}
          >
            {loading === 'Relancer le calcul croisé' ? <CircularProgress size={24} /> : 'Relancer le calcul croisé'}
          </Button>
        )}
      </Paper>
    </Box>
  );
};

export default CroisedStrategiesMonitorPage;
