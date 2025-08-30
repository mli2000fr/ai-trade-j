import React from 'react';

interface TradeAutoBlockProps {
  autoSymbols: string;
  isExecuting: boolean;
  onChange: (value: string) => void;
  onTrade: () => void;
  analyseGptText: string;
  onAnalyseGptChange: (text: string) => void;
}

const TradeAutoBlock: React.FC<TradeAutoBlockProps> = ({ autoSymbols, isExecuting, onChange, onTrade, analyseGptText, onAnalyseGptChange }) => (
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
    {/* Input pour fichier d'analyse GPT */}
    <div className="trade-auto-input-row" style={{ marginTop: 8 }}>
      <label className="trade-auto-label" htmlFor="analyse-gpt-file">Analyse GPT (optionnel, .txt)&nbsp;:</label>
      <input
        id="analyse-gpt-file"
        type="file"
        accept=".txt"
        onChange={e => {
          const file = e.target.files && e.target.files[0];
          if (file) {
            const reader = new FileReader();
            reader.onload = ev => onAnalyseGptChange(ev.target?.result as string || '');
            reader.readAsText(file);
          } else {
            onAnalyseGptChange('');
          }
        }}
      />
      {/* Affichage du nom du fichier sélectionné (optionnel) */}
      {analyseGptText && <span style={{ marginLeft: 8, color: '#888', fontSize: 12 }}>Fichier chargé</span>}
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
