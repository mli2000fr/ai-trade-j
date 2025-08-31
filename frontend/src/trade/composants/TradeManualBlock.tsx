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
import Checkbox from '@mui/material/Checkbox';
import FormControlLabel from '@mui/material/FormControlLabel';

interface TradeManualBlockProps {
  action: 'buy' | 'sell';
  symbol: string;
  quantity: number;
  ownedSymbols: string[];
  isExecuting: boolean;
  cancelOpposite: boolean;
  forceDayTrade: boolean;
  onChangeAction: (action: 'buy' | 'sell') => void;
  onChangeSymbol: (symbol: string) => void;
  onChangeQuantity: (quantity: number) => void;
  onTrade: () => void;
  onChangeCancelOpposite: (value: boolean) => void;
  onChangeForceDayTrade: (value: boolean) => void;
}

const TradeManualBlock: React.FC<TradeManualBlockProps> = ({
  action,
  symbol,
  quantity,
  ownedSymbols,
  isExecuting,
  cancelOpposite,
  forceDayTrade,
  onChangeAction,
  onChangeSymbol,
  onChangeQuantity,
  onTrade,
  onChangeCancelOpposite,
  onChangeForceDayTrade
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
          onChange={e => onChangeAction(e.target.value as 'buy' | 'sell')}
        >
          <MenuItem value="buy">Acheter</MenuItem>
          <MenuItem value="sell">Vendre</MenuItem>
        </Select>
      </FormControl>
      <TextField
        label="Symbole"
        value={symbol}
        onChange={e => onChangeSymbol(e.target.value.toUpperCase())}
        inputProps={{ maxLength: 8 }}
        size="small"
        sx={{ minWidth: 120 }}
      />
      <TextField
        label="Quantité"
        type="number"
        inputProps={{ min: 0.01, step: 'any' }}
        value={quantity}
        onChange={e => onChangeQuantity(parseFloat(e.target.value))}
        size="small"
        sx={{ minWidth: 100 }}
      />
      <Box sx={{ width: '100%', display: 'flex', gap: 2, alignItems: 'center', mt: 1 }}>
        <FormControlLabel
          control={
            <Checkbox
              checked={cancelOpposite}
              onChange={e => onChangeCancelOpposite(e.target.checked)}
              color="primary"
            />
          }
          label="cancel opposite"
        />
        <FormControlLabel
          control={
            <Checkbox
              checked={forceDayTrade}
              onChange={e => onChangeForceDayTrade(e.target.checked)}
              color="primary"
            />
          }
          label="force day trade"
        />
      </Box>
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
