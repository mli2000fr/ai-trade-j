import React, { useEffect, useState } from 'react';
import { Box, Typography, LinearProgress, Paper, Divider, Button, CircularProgress, Chip } from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import HourglassTopIcon from '@mui/icons-material/HourglassTop';

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

  let percent = 0;
  let statusColor: "info" | "success" | "error" | "primary" | "secondary" | "warning" | "default" = 'info';
  let statusIcon = <HourglassTopIcon color="info" />;
  let timeSinceUpdate = null;
  let estimatedTotal = null;
  let estimatedRemaining = null;

    if(progress?.status !== undefined && progress?.status !== undefined){
      percent = progress?.totalSymbols > 0 ? Math.round(100 * progress?.processedSymbols / progress?.totalSymbols) : 0;
      statusColor = progress?.status === 'termine' ? 'success' : progress.status === 'erreur' ? 'error' : 'info';
      statusIcon = progress?.status === 'termine' ? <CheckCircleIcon color="success" /> : progress?.status === 'erreur' ? <ErrorIcon color="error" /> : <HourglassTopIcon color="info" />;
      timeSinceUpdate = progress?.lastUpdate ? Math.round((Date.now() - progress?.lastUpdate) / 1000) : null;
      estimatedTotal = progress?.processedSymbols > 0 ? Math.round((Date.now() - progress?.startTime) / progress?.processedSymbols * progress?.totalSymbols) : null;
      estimatedRemaining = estimatedTotal ? estimatedTotal - (Date.now() - progress.startTime) : null;
    }

return (<>
                 {!progress && (<Box p={4} textAlign="center">
                       <Typography variant="h6" gutterBottom>Aucun calcul en cours</Typography>
                       <Button
                         variant="contained"
                         color="info"
                         onClick={() => callApi('/api/best-combination/calcul', 'Calcule mix strategies')}
                         sx={{ mt: 2 }}
                       >
                         {loading === 'Calcule mix strategies' ? <CircularProgress size={24} /> : 'Lancer le calcul mix strategies'}
                       </Button>
                     </Box>)}
               <Box p={4}>
                    <Paper elevation={3} sx={{ p: 3, maxWidth: 600, margin: '0 auto' }}>
                      <Box display="flex" alignItems="center" gap={1} mb={2}>
                        {statusIcon}
                        <Typography variant="h5" gutterBottom>Suivi du calcul des stratégies mixtes</Typography>
                        {progress && progress.status !== undefined && (
                          <Chip label={progress.status} color={statusColor} size="small" sx={{ ml: 2 }} />
                        )}
                      </Box>
                      <Divider sx={{ mb: 2 }} />
                      <Box mb={2}>
                        <LinearProgress variant="determinate" value={percent} sx={{ height: 10, borderRadius: 5 }} />
                        <Typography align="center" mt={1}>{percent}%</Typography>
                      </Box>
                      <Box display="flex" justifyContent="space-between" mb={2}>
                        <Typography>Progression :</Typography>
                         {progress && progress.processedSymbols !== undefined && (
                             <Typography><b>{progress.processedSymbols} / {progress.totalSymbols}</b></Typography>
                         )}
                        </Box>
                      <Box display="flex" justifyContent="space-between" mb={2}>
                        <Typography>Inserts réussis :</Typography>
                        {progress && progress.nbInsert !== undefined && (
                            <Typography color="success.main"><b>{progress.nbInsert}</b></Typography>
                        )}
                      </Box>
                      <Box display="flex" justifyContent="space-between" mb={2}>
                        <Typography>Erreurs :</Typography>
                        {progress && progress.error !== undefined && (
                          <Chip label={progress.error} color={progress.error > 0 ? 'error' : 'default'} size="small" />
                        )}
                      </Box>
                      <Box display="flex" justifyContent="space-between" mb={2}>
                        <Typography>Dernier symbole traité :</Typography>
                        {progress && progress.lastSymbol !== undefined && (
                          <Typography><b>{progress.lastSymbol || '-'}</b></Typography>
                        )}
                      </Box>
                      <Box display="flex" justifyContent="space-between" mb={2}>
                        <Typography>Début :</Typography>
                        {progress && progress.lastSymbol !== undefined && (
                          <Typography>{progress.lastSymbol ? new Date(progress.startTime).toLocaleString() : '-'}</Typography>
                        )}
                      </Box>
                      <Box display="flex" justifyContent="space-between" mb={2}>
                        <Typography>Fin :</Typography>
                        {progress && progress.endTime !== undefined && (
                          <Typography>{progress.endTime ? new Date(progress.endTime).toLocaleString() : '-'}</Typography>
                        )}
                      </Box>
                      <Box display="flex" justifyContent="space-between" mb={2}>
                        <Typography>Durée :</Typography>
                        {progress && progress.startTime !== undefined && (
                          <Typography>{formatDuration(progress.startTime, progress.endTime)}</Typography>
                        )}
                      </Box>
                      <Box display="flex" justifyContent="space-between" mb={2}>
                        <Typography>Dernière mise à jour :</Typography>
                        <Typography>{timeSinceUpdate !== null ? `${timeSinceUpdate}s` : '-'}</Typography>
                      </Box>
                      <Box display="flex" justifyContent="space-between" mb={2}>
                        <Typography>Estimation temps restant :</Typography>
                        <Typography>{estimatedRemaining && estimatedRemaining > 0 ? formatDuration(0, estimatedRemaining) : '-'}</Typography>
                      </Box>
                    </Paper>
                  </Box>
                 </>
              );
};

export default MixStrategiesMonitorPage;
