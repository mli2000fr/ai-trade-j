import React from 'react';
import '../TradePage.css';
import TradeAIJsonResult from './TradeAIJsonResult';
import TradeAITextResult from './TradeAITextResult';
import TradeMessage from './TradeMessage';

interface TradeAIResultsProps {
  aiJsonResult: any | null;
  aiTextResult: string | null;
  message: string;
}

const TradeAIResults: React.FC<TradeAIResultsProps> = ({ aiJsonResult, aiTextResult, message }) => {
  return (
    <>
      {aiJsonResult && <TradeAIJsonResult aiJsonResult={aiJsonResult} />}
      {aiTextResult && <TradeAITextResult aiTextResult={aiTextResult} />}
      {!aiJsonResult && !aiTextResult && message && <TradeMessage message={message} />}
    </>
  );
};

export default TradeAIResults;
