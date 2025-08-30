import React from 'react';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import '../TradePage.css';

interface TradeAITextResultProps {
  aiTextResult: string;
}

const TradeAITextResult: React.FC<TradeAITextResultProps> = ({ aiTextResult }) => {
  if (!aiTextResult) return null;
  return (
    <Paper sx={{ p: 2, mb: 2 }}>
      <Typography variant="subtitle1" sx={{ fontWeight: 'bold', mb: 1 }}>Résultat AI (texte)&nbsp;:</Typography>
      <Typography variant="body2" sx={{ whiteSpace: 'pre-line' }}>{aiTextResult}</Typography>
    </Paper>
  );
};

export default TradeAITextResult;
