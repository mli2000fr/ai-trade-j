import React, { useEffect, useState } from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import Checkbox from '@mui/material/Checkbox';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import FormGroup from '@mui/material/FormGroup';
import FormControlLabel from '@mui/material/FormControlLabel';
import Alert from '@mui/material/Alert';
import Button from '@mui/material/Button';
import CircularProgress from '@mui/material/CircularProgress';

interface StrategyListDto {
  allStrategies: string[];
  activeStrategies: string[];
  combinationMode: string;
  descriptions?: Record<string, string>;
  logs?: string[];
}

const MODES = [
  { value: 'MAJORITY', label: 'Majorité' },
  { value: 'ALL', label: 'Tous' },
  { value: 'ANY', label: 'Au moins un' },
  { value: 'SINGLE', label: 'Unique' },
];

const STRATEGY_DESCRIPTIONS: Record<string, string> = {
  'SMA Crossover': 'Achat quand la moyenne mobile courte croise au-dessus de la longue. Vente dans le cas inverse.',
  'RSI': 'Achat si le RSI est en survente (<30), vente en surachat (>70).',
  'MACD': 'Achat quand la ligne MACD croise au-dessus du signal, vente dans le cas inverse.',
  'Breakout': 'Achat si le prix casse la résistance (plus haut récent), vente si le prix casse le support.',
  'Mean Reversion': 'Achat si le prix est sous la moyenne mobile, vente s’il est au-dessus.',
  'Trend Following': 'Achat si le prix casse le plus haut sur une longue période, vente sur le plus bas.'
};

const StrategyManagerBlock: React.FC = () => {
  const [strategies, setStrategies] = useState<StrategyListDto | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [mode, setMode] = useState<string>('MAJORITY');
  const [loading, setLoading] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Charger la liste des stratégies et le mode courant
  const fetchStrategies = async () => {
    setLoading('fetch');
    setError(null);
    try {
      const res = await fetch('/api/stra/strategies');
      if (!res.ok) throw new Error('Erreur API');
      const data: StrategyListDto = await res.json();
      setStrategies(data);
      setSelected(data.activeStrategies);
      setMode(data.combinationMode);
    } catch (e) {
      setError('Erreur lors du chargement des stratégies');
    } finally {
      setLoading(null);
    }
  };

  useEffect(() => { fetchStrategies(); }, []);

  // Changer les stratégies actives
  const handleChangeActive = async (name: string) => {
    let newSelected: string[];
    if (selected.includes(name)) {
      newSelected = selected.filter(s => s !== name);
    } else {
      newSelected = [...selected, name];
    }
    setSelected(newSelected);
    setLoading('active');
    setError(null);
    try {
      await fetch('/api/stra/strategies/active', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ strategyNames: newSelected }),
      });
      await fetchStrategies();
    } catch (e) {
      setError('Erreur lors de la modification des stratégies actives');
    } finally {
      setLoading(null);
    }
  };

  // Changer le mode de combinaison
  const handleChangeMode = async (newMode: string) => {
    setMode(newMode);
    setLoading('mode');
    setError(null);
    try {
      await fetch('/api/stra/strategies/mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ combinationMode: newMode }),
      });
      await fetchStrategies();
    } catch (e) {
      setError('Erreur lors de la modification du mode');
    } finally {
      setLoading(null);
    }
  };

  const callApi = async (endpoint: string, label: string) => {
    setLoading(label);
    setSuccess(null);
    setError(null);
    try {
      const res = await fetch(`/api/stra${endpoint}`);
      if (!res.ok) throw new Error('Erreur serveur');
      setSuccess(`${label} : succès !`);
    } catch (e) {
      setError(`${label} : échec`);
    } finally {
      setLoading(null);
    }
  };

  if (!strategies) return <div>Chargement des stratégies...</div>;

  return (
    <>
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Actions rapides sur la base de données
          </Typography>
          <Stack direction="row" spacing={2}>
            <Button
              variant="contained"
              color="primary"
              onClick={() => callApi('/strategies/db/update-assets', 'Update Assets')}
              disabled={loading !== null}
            >
              {loading === 'Update Assets' ? <CircularProgress size={24} /> : 'Update Assets'}
            </Button>
            <Button
              variant="contained"
              color="secondary"
              onClick={() => callApi('/strategies/db/update-daily-valu', 'Update Daily Value')}
              disabled={loading !== null}
            >
              {loading === 'Update Daily Value' ? <CircularProgress size={24} /> : 'Update Daily Value'}
            </Button>
            <Button
              variant="contained"
              color="success"
              onClick={() => callApi('/strategies/optimise-param', 'Optimise Param')}
              disabled={loading !== null}
            >
              {loading === 'Optimise Param' ? <CircularProgress size={24} /> : 'Optimise Param'}
            </Button>
          </Stack>
          {success && <Alert severity="success" sx={{ mt: 2 }}>{success}</Alert>}
          {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}
        </CardContent>
      </Card>
      <Card sx={{ mb: 2, borderRadius: 2, boxShadow: 1, border: '1px solid #e0e0e0', bgcolor: 'background.paper' }}>
        <CardContent>
          <Typography variant="h6" sx={{ mb: 2 }}>Gestion des stratégies de trading</Typography>
          <Stack spacing={2}>
            <div>
              <Typography variant="subtitle1" sx={{ mb: 1 }}>Stratégies disponibles :</Typography>
              <FormGroup>
                {strategies.allStrategies.map(name => (
                  <FormControlLabel
                    key={name}
                    control={
                      <Checkbox
                        checked={selected.includes(name)}
                        onChange={() => handleChangeActive(name)}
                        disabled={loading !== null}
                      />
                    }
                    label={
                      <Stack direction="column" alignItems="flex-start" spacing={0.2}>
                        <Typography fontWeight={selected.includes(name) ? 'bold' : 'normal'}>{name}</Typography>
                        <Typography variant="caption" color="text.secondary">
                          {STRATEGY_DESCRIPTIONS[name] || 'Aucune description.'}
                        </Typography>
                      </Stack>
                    }
                    sx={{ alignItems: 'center', mb: 1 }}
                  />
                ))}
              </FormGroup>
            </div>
            <FormControl fullWidth size="small">
              <InputLabel id="combination-mode-label">Mode de combinaison</InputLabel>
              <Select
                labelId="combination-mode-label"
                value={mode}
                label="Mode de combinaison"
                onChange={e => handleChangeMode(e.target.value)}
                disabled={loading !== null}
              >
                {MODES.map(m => (
                  <MenuItem key={m.value} value={m.value}>{m.label}</MenuItem>
                ))}
              </Select>
            </FormControl>
            <Typography variant="body2" color="text.secondary">
              Stratégies actives : {strategies.activeStrategies.join(', ') || 'Aucune'}<br />
              Mode actuel : {MODES.find(m => m.value === strategies.combinationMode)?.label || strategies.combinationMode}
            </Typography>
            {strategies.logs && strategies.logs.length > 0 && (
              <div>
                <Typography variant="subtitle2" sx={{ mb: 1 }}>Logs récents :</Typography>
                <Stack spacing={0.5} sx={{ bgcolor: '#f7f7f7', borderRadius: 1, p: 1 }}>
                  {strategies.logs.map((log, i) => (
                    <Typography key={i} variant="caption" color="text.secondary">{log}</Typography>
                  ))}
                </Stack>
              </div>
            )}
            {error && <Alert severity="error">{error}</Alert>}
          </Stack>
        </CardContent>
      </Card>
    </>
  );
};

export default StrategyManagerBlock;
