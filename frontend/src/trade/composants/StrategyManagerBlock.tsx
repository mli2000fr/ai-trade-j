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
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Charger la liste des stratégies et le mode courant
  const fetchStrategies = async () => {
    setLoading(true);
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
      setLoading(false);
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
    setLoading(true);
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
      setLoading(false);
    }
  };

  // Changer le mode de combinaison
  const handleChangeMode = async (newMode: string) => {
    setMode(newMode);
    setLoading(true);
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
      setLoading(false);
    }
  };

  if (!strategies) return <div>Chargement des stratégies...</div>;

  return (
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
                      disabled={loading}
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
              disabled={loading}
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
  );
};

export default StrategyManagerBlock;
