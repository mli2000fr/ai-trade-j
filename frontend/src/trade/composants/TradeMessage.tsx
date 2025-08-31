import React from 'react';
import Alert from '@mui/material/Alert';
import '../TradePage.css';

interface TradeMessageProps {
  message: string;
  severity?: 'info' | 'error' | 'success' | 'warning';
}

const TradeMessage: React.FC<TradeMessageProps> = ({ message, severity = 'info' }) => {
  if (!message) return null;
  return <Alert severity={severity} sx={{ mt: 2 }}>{message}</Alert>;
};

export default TradeMessage;
