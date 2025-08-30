import React from 'react';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import CircularProgress from '@mui/material/CircularProgress';
import Alert from '@mui/material/Alert';

interface Compte {
  id: number;
  nom: string;
  alias?: string;
  real?: boolean;
}

interface TradeCompteSelectProps {
  comptes: Compte[];
  selectedCompteId: number | null;
  loading: boolean;
  error: string | null;
  onSelect: (id: number) => void;
}

const TradeCompteSelect: React.FC<TradeCompteSelectProps> = ({ comptes, selectedCompteId, loading, error, onSelect }) => (
  <FormControl fullWidth variant="outlined" size="small" sx={{ mb: 2 }}>
    <InputLabel id="compte-select-label">Compte</InputLabel>
    {loading ? (
      <CircularProgress size={24} sx={{ mt: 1, mb: 1 }} />
    ) : error ? (
      <Alert severity="error">{error}</Alert>
    ) : (
      <Select
        labelId="compte-select-label"
        value={selectedCompteId ?? ''}
        label="Compte"
        onChange={e => onSelect(Number(e.target.value))}
      >
        {comptes.map(compte => (
          <MenuItem key={compte.id} value={compte.id}>
            {compte.nom}
            {compte.alias ? ` (${compte.alias})` : ''}
            {compte.real === true ? ' [REAL]' : compte.real === false ? ' [PAPER]' : ''}
          </MenuItem>
        ))}
      </Select>
    )}
  </FormControl>
);

export default TradeCompteSelect;
