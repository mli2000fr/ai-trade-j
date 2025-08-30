import React from 'react';
import '../TradePage.css';

interface TradeAITextResultProps {
  aiTextResult: string;
}

const TradeAITextResult: React.FC<TradeAITextResultProps> = ({ aiTextResult }) => {
  if (!aiTextResult) return null;
  return (
    <div className="trade-ai-text-result trade-ai-text-result-custom">
      {aiTextResult}
    </div>
  );
};

export default TradeAITextResult;

