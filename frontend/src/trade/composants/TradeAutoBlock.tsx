import React from 'react';

interface TradeAutoBlockProps {
  autoSymbols: string;
  isExecuting: boolean;
  onChange: (value: string) => void;
  onTrade: () => void;
}

const TradeAutoBlock: React.FC<TradeAutoBlockProps> = ({ autoSymbols, isExecuting, onChange, onTrade }) => (
  <div className="trade-auto-block">
    <h2 className="trade-auto-title">Trade Auto</h2>
    <div className="trade-auto-input-row">
      <label className="trade-auto-label" htmlFor="auto-symbols">Symboles&nbsp;</label>
      <input
        id="auto-symbols"
        type="text"
        className="trade-auto-input"
        value={autoSymbols}
        onChange={e => onChange(e.target.value)}
        placeholder="AAPL,KO,NVDA,TSLA,AMZN,MSFT,AMD,META,SHOP,PLTR"
      />
    </div>
    <div className="trade-auto-btn-row">
      <button onClick={onTrade} disabled={isExecuting || !autoSymbols.trim()} className="trade-auto-btn">
        {isExecuting ? <span className="spinner trade-spinner"></span> : null}
        {isExecuting ? 'Exécution...' : 'Exécuter'}
      </button>
    </div>
  </div>
);

export default TradeAutoBlock;

