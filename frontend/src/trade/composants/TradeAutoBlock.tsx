import React, { useRef } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import CircularProgress from '@mui/material/CircularProgress';
import TradeAIResults from './TradeAIResults';
import Snackbar from '@mui/material/Snackbar';
import MuiAlert, { AlertColor } from '@mui/material/Alert';
import Radio from '@mui/material/Radio';
import RadioGroup from '@mui/material/RadioGroup';
import FormControlLabel from '@mui/material/FormControlLabel';
import FormLabel from '@mui/material/FormLabel';

interface TradeAutoBlockProps {
  autoSymbols: string;
  isExecuting: boolean;
  disabled?: boolean;
  onChange: (value: string) => void;
  onTrade: (agent: string) => void;
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
  const [loadingPrompt, setLoadingPrompt] = React.useState(false);
  const [snackbarOpen, setSnackbarOpen] = React.useState(false);
  const [snackbarMessage, setSnackbarMessage] = React.useState('');
  const [snackbarSeverity, setSnackbarSeverity] = React.useState<AlertColor>('success');
  const [agent, setAgent] = React.useState<'gpt' | 'deepseek'>('deepseek');

  const showSnackbar = (message: string, severity: AlertColor = 'success') => {
    setSnackbarMessage(message);
    setSnackbarSeverity(severity);
    setSnackbarOpen(true);
  };

  const handleGetPrompt = async () => {
    setLoadingPrompt(true);
    try {
      console.log('Get prompt click', { idCompte: compteId, symbols: autoSymbols });
      const params = new URLSearchParams({ idCompte: String(compteId), symbols: autoSymbols });
      const response = await fetch(`/api/trade/getPromptAnalyseSymbol?${params.toString()}`);
      if (!response.ok) {
        throw new Error('Erreur lors de la récupération du prompt');
      }
      const prompt = await response.text();
      navigator.clipboard.writeText(prompt).then(() => {
        onAnalyseGptChange(prompt);
        showSnackbar('Le prompt a été copié dans le presse-papier', 'success');
      }, () => {
        onAnalyseGptChange(prompt);
        showSnackbar('Le prompt a été généré, mais la copie dans le presse-papier a échoué', 'warning');
      });
    } catch (error) {
      console.error(error);
      showSnackbar('Une erreur est survenue lors de la récupération du prompt', 'error');
    } finally {
      setLoadingPrompt(false);
    }
  };

  return (
       <>
       <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
       <Typography variant="h6" >Trade Auto</Typography>
        </Box>
    <Box sx={{ mb: 3, p: 2, border: '1px solid #eee', borderRadius: 2, backgroundColor: '#f5f5f5' }}>

      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2, flexWrap: 'wrap' }}>
        <TextField
          label="Symbols"
          value={autoSymbols}
          onChange={e => onChange(e.target.value)}
          placeholder=""
          size="small"
          sx={{ minWidth: 500 }}
        />
        <Button
          onClick={handleGetPrompt}
          disabled={disabled || isExecuting || !autoSymbols.trim()}
          variant="outlined"
          size="small"
        >
          {loadingPrompt && <CircularProgress size={20} sx={{ mr: 1 }} />}
          {loadingPrompt ? 'Chargement...' : 'Get Prompt'}
        </Button>
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
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2, flexWrap: 'wrap' }}>
        <FormLabel component="legend">Agent</FormLabel>
        <RadioGroup row value={agent} onChange={e => setAgent(e.target.value as 'gpt' | 'deepseek')}>
          <FormControlLabel value="gpt" control={<Radio />} label="GPT" />
          <FormControlLabel value="deepseek" control={<Radio />} label="Deepseek" />
        </RadioGroup>
      </Box>
      <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, mb: 2 }}>
        <Button
          onClick={() => onTrade(agent)}
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
    <Snackbar
      open={snackbarOpen}
      autoHideDuration={5000}
      onClose={() => setSnackbarOpen(false)}
      anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
    >
      <MuiAlert onClose={() => setSnackbarOpen(false)} severity={snackbarSeverity} elevation={6} variant="filled">
        {snackbarMessage}
      </MuiAlert>
    </Snackbar>
    </>
  );
};

export default TradeAutoBlock;
