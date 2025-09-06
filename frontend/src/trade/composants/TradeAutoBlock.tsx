import React, { useRef } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import CircularProgress from '@mui/material/CircularProgress';
import TradeAIResults from './TradeAIResults';

interface TradeAutoBlockProps {
  autoSymbols: string;
  isExecuting: boolean;
  disabled?: boolean;
  onChange: (value: string) => void;
  onTrade: () => void;
  analyseGptText: string;
  onAnalyseGptChange: (text: string) => void;
  message?: string;
  aiJsonResult: any | null;
  aiTextResult: string | null;
  compteId?: string | number;
  onOrdersUpdate?: (orders: any[]) => void;
  idGpt?: string;
}

const TradeAutoBlock: React.FC<TradeAutoBlockProps> = ({ autoSymbols, isExecuting, disabled, onChange, onTrade, analyseGptText, onAnalyseGptChange, message, aiJsonResult, aiTextResult, compteId, onOrdersUpdate, idGpt }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  return (
       <>
       <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
       <Typography variant="h6" >Trade Auto</Typography>
        </Box>
    <Box sx={{ mb: 3, p: 2, border: '1px solid #eee', borderRadius: 2, backgroundColor: '#f5f5f5' }}>

      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2, flexWrap: 'wrap' }}>
        <TextField
          label="Symboles"
          value={autoSymbols}
          onChange={e => onChange(e.target.value)}
          placeholder=""
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
      <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, mb: 2 }}>
        <Button
          onClick={onTrade}
          disabled={disabled || isExecuting || !autoSymbols.trim()}
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
      <TradeAIResults
        aiJsonResult={aiJsonResult}
        aiTextResult={aiTextResult}
        compteId={compteId}
        onOrdersUpdate={onOrdersUpdate}
        idGpt={idGpt}
      />
    </Box>
    </>
  );
};

export default TradeAutoBlock;
