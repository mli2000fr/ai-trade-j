import React from 'react';
import '../TradePage.css';
import TradeAIJsonResult from './TradeAIJsonResult';
import TradeAITextResult from './TradeAITextResult';

interface TradeAIResultsProps {
  aiJsonResult: any | null;
  aiTextResult: string | null;
  compteId?: string | number;
  onOrdersUpdate?: (orders: any[]) => void;
  idGpt?: string;
}

const TradeAIResults: React.FC<TradeAIResultsProps> = ({ aiJsonResult, aiTextResult, compteId, onOrdersUpdate, idGpt }) => {
  return (
    <>
      {aiJsonResult && <TradeAIJsonResult aiJsonResult={aiJsonResult} compteId={compteId} onOrdersUpdate={onOrdersUpdate} idGpt={idGpt} />}
      {aiTextResult && <TradeAITextResult aiTextResult={aiTextResult} />}
    </>
  );
};

export default TradeAIResults;
