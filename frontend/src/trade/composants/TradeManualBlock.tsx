import React from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import CircularProgress from '@mui/material/CircularProgress';

interface TradeManualBlockProps {
  action: 'buy' | 'sell' | 'trade-ai';
  symbol: string;
  quantity: number;
  ownedSymbols: string[];
  isExecuting: boolean;
  onChangeAction: (action: 'buy' | 'sell' | 'trade-ai') => void;
  onChangeSymbol: (symbol: string) => void;
  onChangeQuantity: (quantity: number) => void;
  onTrade: () => void;
}

const TradeManualBlock: React.FC<TradeManualBlockProps> = ({
  action,
  symbol,
  quantity,
  ownedSymbols,
  isExecuting,
  onChangeAction,
  onChangeSymbol,
  onChangeQuantity,
  onTrade
}) => (
  <Box sx={{ mb: 3, p: 2, border: '1px solid #eee', borderRadius: 2 }}>
    <Typography variant="h6" sx={{ mb: 2 }}>Trade Manuel</Typography>
    <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap', mb: 2 }}>
      <FormControl size="small" sx={{ minWidth: 120 }}>
        <InputLabel id="action-select-label">Action</InputLabel>
        <Select
          labelId="action-select-label"
          value={action}
          label="Action"
          onChange={e => onChangeAction(e.target.value as 'buy' | 'sell' | 'trade-ai')}
        >
          <MenuItem value="buy">Acheter</MenuItem>
          <MenuItem value="sell">Vendre</MenuItem>
          <MenuItem value="trade-ai">Trade AI</MenuItem>
        </Select>
      </FormControl>
      {(action === 'buy' || action === 'trade-ai') ? (
        <TextField
          label="Symbole"
          value={symbol}
          onChange={e => onChangeSymbol(e.target.value.toUpperCase())}
          inputProps={{ maxLength: 8 }}
          size="small"
          sx={{ minWidth: 120 }}
        />
      ) : (
        <FormControl size="small" sx={{ minWidth: 120 }}>
          <InputLabel id="symbol-select-label">Symbole</InputLabel>
          <Select
            labelId="symbol-select-label"
            value={symbol}
            label="Symbole"
            onChange={e => onChangeSymbol(e.target.value)}
          >
            {ownedSymbols.length === 0 ? (
              <MenuItem value="">Aucune position</MenuItem>
            ) : (
              ownedSymbols.map((sym: string) => (
                <MenuItem key={sym} value={sym}>{sym}</MenuItem>
              ))
            )}
          </Select>
        </FormControl>
      )}
      {action !== 'trade-ai' && (
        <TextField
          label="Quantité"
          type="number"
          inputProps={{ min: 0.01, step: 'any' }}
          value={quantity}
          onChange={e => onChangeQuantity(parseFloat(e.target.value))}
          size="small"
          sx={{ minWidth: 100 }}
        />
      )}
    </Box>
    <Box>
      <Button
        onClick={onTrade}
        disabled={isExecuting}
        variant="contained"
        size="large"
      >
        {isExecuting && <CircularProgress size={20} sx={{ mr: 1 }} />}
        {isExecuting ? 'Exécution...' : 'Exécuter'}
      </Button>
    </Box>
  </Box>
);

export default TradeManualBlock;
