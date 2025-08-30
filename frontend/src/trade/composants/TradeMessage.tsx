import React from 'react';
import Alert from '@mui/material/Alert';
import '../TradePage.css';

interface TradeMessageProps {
  message: string;
}

const TradeMessage: React.FC<TradeMessageProps> = ({ message }) => {
  if (!message) return null;
  return <Alert severity="info" sx={{ mt: 2 }}>{message}</Alert>;
};

export default TradeMessage;
