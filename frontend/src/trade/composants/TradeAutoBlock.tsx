import React, { useRef } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import CircularProgress from '@mui/material/CircularProgress';

interface TradeAutoBlockProps {
  autoSymbols: string;
  isExecuting: boolean;
  onChange: (value: string) => void;
  onTrade: () => void;
  analyseGptText: string;
  onAnalyseGptChange: (text: string) => void;
}

const TradeAutoBlock: React.FC<TradeAutoBlockProps> = ({ autoSymbols, isExecuting, onChange, onTrade, analyseGptText, onAnalyseGptChange }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  return (
    <Box sx={{ mb: 3, p: 2, border: '1px solid #eee', borderRadius: 2 }}>
      <Typography variant="h6" sx={{ mb: 2 }}>Trade Auto</Typography>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2, flexWrap: 'wrap' }}>
        <TextField
          label="Symboles"
          value={autoSymbols}
          onChange={e => onChange(e.target.value)}
          placeholder="AAPL,KO,NVDA,TSLA,AMZN,MSFT,AMD,META,SHOP,PLTR"
          size="small"
          sx={{ minWidth: 500 }}
        />
      </Box>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2, flexWrap: 'wrap' }}>
        <Button
          variant="outlined"
          component="label"
          size="small"
        >
          Charger une analyse GPT (.txt)
          <input
            ref={fileInputRef}
            type="file"
            accept=".txt"
            hidden
            onChange={e => {
              const file = e.target.files && e.target.files[0];
              if (file) {
                const reader = new FileReader();
                reader.onload = ev => onAnalyseGptChange(ev.target?.result as string || '');
                reader.readAsText(file);
              } else {
                onAnalyseGptChange('');
              }
            }}
          />
        </Button>
        {analyseGptText && (
          <Typography variant="caption" color="success.main">Fichier chargé</Typography>
        )}
      </Box>
      <Box>
        <Button
          onClick={onTrade}
          disabled={isExecuting || !autoSymbols.trim()}
          variant="contained"
          size="large"
        >
          {isExecuting && <CircularProgress size={20} sx={{ mr: 1 }} />}
          {isExecuting ? 'Exécution...' : 'Exécuter'}
        </Button>
      </Box>
    </Box>
  );
};

export default TradeAutoBlock;
