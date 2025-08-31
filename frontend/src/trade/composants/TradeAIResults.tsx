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
  idGpt?: string;
}

const TradeAIResults: React.FC<TradeAIResultsProps> = ({ aiJsonResult, aiTextResult, message, compteId, onOrdersUpdate, idGpt }) => {
  // Détection d'une erreur dans le message (commence par 'Erreur' ou 'error', insensible à la casse)
  const isError = message && /^erreur|error/i.test(message.trim());
  return (
    <>
      {aiJsonResult && <TradeAIJsonResult aiJsonResult={aiJsonResult} compteId={compteId} onOrdersUpdate={onOrdersUpdate} idGpt={idGpt} />}
      {aiTextResult && <TradeAITextResult aiTextResult={aiTextResult} />}
      {!aiJsonResult && !aiTextResult && message && <TradeMessage message={message} severity={isError ? 'error' : 'info'} />}
    </>
  );
};

export default TradeAIResults;
