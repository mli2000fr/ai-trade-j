import React from 'react';
import '../TradePage.css';

interface TradeMessageProps {
  message: string;
}

const TradeMessage: React.FC<TradeMessageProps> = ({ message }) => {
  if (!message) return null;
  return <div className="trade-message trade-message-custom">{message}</div>;
};

export default TradeMessage;

