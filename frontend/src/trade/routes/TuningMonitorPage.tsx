import React, { useEffect, useState } from 'react';
import { Box, Typography, Table, TableHead, TableRow, TableCell, TableBody, Button, CircularProgress } from '@mui/material';
import DownloadIcon from '@mui/icons-material/Download';
import LinearProgress from '@mui/material/LinearProgress';

const API_URL = '/api/tuning/metrics';
const API_PROGRESS_URL = '/api/tuning/progress';

const TuningMonitorPage: React.FC = () => {
  const [metrics, setMetrics] = useState<any[]>([]);
  const [progress, setProgress] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean | string>(true);

  const fetchAll = async () => {
    setLoading(true);
    try {
      const [metricsRes, progressRes] = await Promise.all([
        fetch(API_URL),
        fetch(API_PROGRESS_URL)
      ]);
      const metricsData = await metricsRes.json();
      const progressData = await progressRes.json();
      setMetrics(metricsData);
      setProgress(progressData);
    } catch (e) {
      setMetrics([]);
      setProgress([]);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchAll();
    const interval = setInterval(fetchAll, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleExport = () => {
    window.open('/api/tuning/metrics/csv', '_blank');
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

  return (
    <Box sx={{ p: 4 }}>
      <Typography variant="h4" gutterBottom>Monitoring du tuning LSTM</Typography>

          <Button
            variant="contained"
            color="info"
            onClick={() => callApi('/api/lstm/tuneAllSymbols', 'Calcule hyper params LSTM')}
            disabled={isTuningEnCours || loading === 'Calcule hyper params LSTM'}
          >
            {loading === 'Calcule hyper params LSTM' ? <CircularProgress size={24} /> : 'Calcule hyper params LSTM'}
          </Button>
      <Typography variant="h6" sx={{ mt: 2, mb: 1 }}>Progression en temps réel</Typography>
      <Table sx={{ mb: 4 }}>
                <TableHead>
                  <TableRow>
                    <TableCell>Symbole</TableCell>
                    <TableCell>Avancement</TableCell>
                    <TableCell>Statut</TableCell>
                    <TableCell>Durée</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {progress.map((row: any, idx: number) => {
                    const percent = row.totalConfigs > 0 ? Math.round(100 * row.testedConfigs / row.totalConfigs) : 0;
                    return (
                      <TableRow key={row.symbol}>
                        <TableCell>{row.symbol}</TableCell>
                        <TableCell>
                          <Box sx={{ minWidth: 120 }}>
                            <LinearProgress variant="determinate" value={percent} sx={{ height: 8, borderRadius: 4 }} />
                            <Typography variant="body2" sx={{ mt: 0.5 }}>{row.testedConfigs} / {row.totalConfigs} ({percent}%)</Typography>
                          </Box>
                        </TableCell>
                        <TableCell>{row.status === 'en_cours' ? 'En cours' : row.status === 'termine' ? 'Terminé' : 'Erreur'}</TableCell>
                        <TableCell>{formatDuration(row.startTime, row.status === 'termine' ? row.endTime : row.lastUpdate)}</TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
      <Typography variant="h6" sx={{ mt: 2, mb: 1 }}>Historique des métriques</Typography>
      <Table>
        <TableHead>
          <TableRow>
            <TableCell>Symbole</TableCell>
            <TableCell>Statut</TableCell>
            <TableCell>MSE</TableCell>
            <TableCell>RMSE</TableCell>
            <TableCell>Direction</TableCell>
            <TableCell>Date</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {metrics.slice(0, 10).map((row, idx) => (
            <TableRow key={idx}>
              <TableCell>{row.symbol}</TableCell>
              <TableCell>{row.status || 'Terminé'}</TableCell>
              <TableCell>{row.mse}</TableCell>
              <TableCell>{row.rmse}</TableCell>
              <TableCell>{row.direction}</TableCell>
              <TableCell>{row.tested_date}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Box>
  );
};

export default TuningMonitorPage;
