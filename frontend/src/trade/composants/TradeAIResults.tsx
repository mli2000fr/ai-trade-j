import React from 'react';
import '../TradePage.css';
import TradeAIJsonResult from './TradeAIJsonResult';
import TradeAITextResult from './TradeAITextResult';
import TradeMessage from './TradeMessage';

interface TradeAIResultsProps {
  aiJsonResult: any | null;
  aiTextResult: string | null;
  message: string;
  compteId?: string | number;
  onOrdersUpdate?: (orders: any[]) => void;
}

const TradeAIResults: React.FC<TradeAIResultsProps> = ({ aiJsonResult, aiTextResult, message, compteId, onOrdersUpdate }) => {
  return (
    <>
      {aiJsonResult && <TradeAIJsonResult aiJsonResult={aiJsonResult} compteId={compteId} onOrdersUpdate={onOrdersUpdate} />}
      {aiTextResult && <TradeAITextResult aiTextResult={aiTextResult} />}
      {!aiJsonResult && !aiTextResult && message && <TradeMessage message={message} />}
    </>
  );
};

export default TradeAIResults;
