import React from 'react';

interface TradeManualBlockProps {
  action: 'buy' | 'sell' | 'trade-ai';
  symbol: string;
  quantity: number;
  ownedSymbols: string[];
  isExecuting: boolean;
  onChangeAction: (action: 'buy' | 'sell' | 'trade-ai') => void;
  onChangeSymbol: (symbol: string) => void;
  onChangeQuantity: (quantity: number) => void;
  onTrade: () => void;
}

const TradeManualBlock: React.FC<TradeManualBlockProps> = ({
  action,
  symbol,
  quantity,
  ownedSymbols,
  isExecuting,
  onChangeAction,
  onChangeSymbol,
  onChangeQuantity,
  onTrade
}) => (
  <div className="trade-manual-block">
    <h2 className="trade-manual-title">Trade Manuel</h2>
    <div className="trade-manual-input-row">
      <label className="trade-manual-label">Action&nbsp;
        <select value={action} onChange={e => onChangeAction(e.target.value as 'buy' | 'sell' | 'trade-ai')}>
          <option value="buy">Acheter</option>
          <option value="sell">Vendre</option>
          <option value="trade-ai">Ttrade AI</option>
        </select>
      </label>
      {action === 'buy' || action === 'trade-ai' ? (
        <label className="trade-manual-label">Symbole&nbsp;&nbsp;
          <input value={symbol} onChange={e => onChangeSymbol(e.target.value.toUpperCase())} maxLength={8} className="trade-manual-input" />
        </label>
      ) : (
        <label className="trade-manual-label">Symbole&nbsp;
          <select value={symbol} onChange={e => onChangeSymbol(e.target.value)} className="trade-manual-input">
            {ownedSymbols.length === 0 ? (
              <option value="">Aucune position</option>
            ) : (
              ownedSymbols.map((sym: string) => (
                <option key={sym} value={sym}>{sym}</option>
              ))
            )}
          </select>
        </label>
      )}
      {action !== 'trade-ai' && (
        <label className="trade-manual-label">Quantité&nbsp;
          <input type="number" min={0.01} step="any" value={quantity} onChange={e => onChangeQuantity(parseFloat(e.target.value))} className="trade-manual-input" />
        </label>
      )}
    </div>
    <div className="trade-manual-btn-row">
      <button onClick={onTrade} disabled={isExecuting} className="trade-manual-btn">
        {isExecuting ? <span className="spinner trade-spinner"></span> : null}
        {isExecuting ? 'Exécution...' : 'Exécuter'}
      </button>
    </div>
  </div>
);

export default TradeManualBlock;

