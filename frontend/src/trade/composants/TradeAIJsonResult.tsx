import React from 'react';
import '../TradePage.css';

interface TradeAIJsonResultProps {
  aiJsonResult: any;
}

const TradeAIJsonResult: React.FC<TradeAIJsonResultProps> = ({ aiJsonResult }) => {
  if (!aiJsonResult) return null;
  if (Array.isArray(aiJsonResult)) {
    const hasPriceLimit = aiJsonResult.some((item: any) => item.price_limit !== undefined && item.price_limit !== null && item.price_limit !== '' || item.priceLimit !== undefined && item.priceLimit !== null && item.priceLimit !== '');
    return (
      <div className="trade-ai-json-result">
        <b className="trade-ai-json-title">Résultat AI :</b>
        <div className="overflow-x-auto">
          <table className="trade-ai-json-table">
            <thead>
              <tr>
                <th>Symbole</th>
                <th>Action</th>
                <th>Quantité</th>
                {hasPriceLimit && <th>Prix limite</th>}
                <th>Stop loss</th>
                <th>Take profit</th>
              </tr>
            </thead>
            <tbody>
              {aiJsonResult.map((item: any, idx: number) => (
                <tr key={idx} className={idx % 2 === 0 ? 'even-row' : 'odd-row'}>
                  <td className="trade-ai-json-symbol">{item.symbol}</td>
                  <td className="trade-ai-json-action">{item.action}</td>
                  <td className="trade-ai-json-qty">{item.quantity ?? item.qty ?? ''}</td>
                  {hasPriceLimit && (
                    <td className="trade-ai-json-price">{item.price_limit ?? item.priceLimit ? (item.price_limit ?? item.priceLimit) + ' $' : '-'}</td>
                  )}
                  <td className="trade-ai-json-stop">{item.stop_loss ?? item.stopLoss ? (item.stop_loss ?? item.stopLoss) + ' $' : '-'}</td>
                  <td className="trade-ai-json-take">{item.take_profit ?? item.takeProfit ? (item.take_profit ?? item.takeProfit) + ' $' : '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  }
  return (
    <div className="trade-ai-json-result">
      <b className="trade-ai-json-title">Résultat AI :</b>
      <table className="trade-ai-json-table trade-ai-json-table-single">
        <tbody>
          {Object.entries(aiJsonResult).map(([key, value]) => (
            <tr key={key}>
              <td className="trade-ai-json-key">{key}</td>
              <td className="trade-ai-json-value">{String(value)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default TradeAIJsonResult;

