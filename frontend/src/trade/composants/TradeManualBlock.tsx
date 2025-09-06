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
  disabled?: boolean;
  cancelOpposite: boolean;
  forceDayTrade: boolean;
  onChangeAction: (action: 'buy' | 'sell') => void;
  onChangeSymbol: (symbol: string) => void;
  onChangeQuantity: (quantity: number) => void;
  onTrade: () => void;
  onChangeCancelOpposite: (value: boolean) => void;
  onChangeForceDayTrade: (value: boolean) => void;
  stopLoss: number | '';
  takeProfit: number | '';
  onChangeStopLoss: (value: number | '') => void;
  onChangeTakeProfit: (value: number | '') => void;
  message?: string;
  positions: { symbol: string; qty: number }[];
  quantiteMaxResetTrigger?: number;
}

const TradeManualBlock: React.FC<TradeManualBlockProps> = ({
  action,
  symbol,
  quantity,
  ownedSymbols,
  isExecuting,
  disabled,
  cancelOpposite,
  forceDayTrade,
  onChangeAction,
  onChangeSymbol,
  onChangeQuantity,
  onTrade,
  onChangeCancelOpposite,
  onChangeForceDayTrade,
  stopLoss,
  takeProfit,
  onChangeStopLoss,
  onChangeTakeProfit,
  message,
  positions,
  quantiteMaxResetTrigger
}) => {
  const [quantiteMaxActive, setQuantiteMaxActive] = React.useState(false);

  // Reset la case quantité max si le trigger change
  React.useEffect(() => {
    setQuantiteMaxActive(false);
  }, [quantiteMaxResetTrigger]);

  React.useEffect(() => {
    if (quantiteMaxActive && action === 'sell') {
      const pos = positions.find(p => p.symbol === symbol);
      if (pos) {
        onChangeQuantity(pos.qty);
      }
    }
  }, [quantiteMaxActive, symbol, action, positions, onChangeQuantity]);

  // Si on change d'action, on désactive la case à cocher
  React.useEffect(() => {
    if (action !== 'sell' && quantiteMaxActive) {
      setQuantiteMaxActive(false);
    }
  }, [action]);

  // Désactivation croisée stop-loss / take-profit en mode vente
  const disableStopLoss = action === 'sell' && takeProfit !== '' && takeProfit !== undefined && takeProfit !== null && !isNaN(Number(takeProfit));
  const disableTakeProfit = action === 'sell' && stopLoss !== '' && stopLoss !== undefined && stopLoss !== null && !isNaN(Number(stopLoss));

  return (
      <>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
      <Typography variant="h6">Trade Manuel</Typography>
       </Box>
    <Box sx={{ mb: 3, p: 2, border: '1px solid #eee', borderRadius: 2, backgroundColor: '#f5f5f5' }}>

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
        {action === 'sell' ? (
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel id="symbol-select-label">Symbole</InputLabel>
            <Select
              labelId="symbol-select-label"
              value={symbol}
              label="Symbole"
              onChange={e => onChangeSymbol(e.target.value as string)}
            >
              {ownedSymbols.length === 0 ? (
                <MenuItem value="" disabled>Aucune position</MenuItem>
              ) : (
                ownedSymbols.map((s) => (
                  <MenuItem key={s} value={s}>{s}</MenuItem>
                ))
              )}
            </Select>
          </FormControl>
        ) : (
          <TextField
            label="Symbole"
            value={symbol}
            onChange={e => onChangeSymbol(e.target.value.toUpperCase())}
            inputProps={{ maxLength: 8 }}
            size="small"
            sx={{ minWidth: 120 }}
          />
        )}
        <TextField
          label="Quantité"
          type="number"
          inputProps={{ min: 0.01, step: 'any' }}
          value={quantity}
          onChange={e => onChangeQuantity(parseFloat(e.target.value))}
          size="small"
          sx={{ minWidth: 50 }}
          disabled={quantiteMaxActive}
        />
        <TextField
          label="Stop-Loss"
          type="number"
          inputProps={{ min: 0, step: 'any' }}
          value={stopLoss}
          onChange={e => onChangeStopLoss(e.target.value === '' ? '' : parseFloat(e.target.value))}
          size="small"
          sx={{ minWidth: 120 }}
          disabled={disableStopLoss}
        />
        <TextField
          label="Take-Profit"
          type="number"
          inputProps={{ min: 0, step: 'any' }}
          value={takeProfit}
          onChange={e => onChangeTakeProfit(e.target.value === '' ? '' : parseFloat(e.target.value))}
          size="small"
          sx={{ minWidth: 120 }}
          disabled={disableTakeProfit}
        />
        <Box sx={{ width: '100%', display: 'flex', gap: 2, alignItems: 'center', mt: 1 }}>
          {action === 'sell' && (
            <FormControlLabel
              control={
                <Checkbox
                  checked={quantiteMaxActive}
                  onChange={e => setQuantiteMaxActive(e.target.checked)}
                  color="primary"
                />
              }
              label="quantité max"
            />
          )}
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
      <Box sx={{ display: 'flex', justifyContent: 'center' }}>
        <Button
          onClick={onTrade}
          disabled={disabled || isExecuting || !symbol || quantity <= 0}
          variant="contained"
          size="large"
        >
          {isExecuting && <CircularProgress size={20} sx={{ mr: 1 }} />}
          {isExecuting ? 'Exécution...' : 'Exécuter'}
        </Button>
      </Box>
      {message && (
        <Typography variant="body2" color={message.toLowerCase().includes('erreur') ? 'error' : 'success.main'} sx={{ mt: 2 }}>
          {message}
        </Typography>
      )}
    </Box>
          </>
  );
};

export default TradeManualBlock;
