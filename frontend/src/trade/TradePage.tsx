import React, { useState, useEffect } from 'react';
import './TradePage.css';

const TRADE_API_URL = '/api/trade/trade';

const TradePage: React.FC = () => {
  const [symbol, setSymbol] = useState('AAPL');
  const [action, setAction] = useState<'buy' | 'sell' | 'trade-ai'>('buy');
  const [quantity, setQuantity] = useState(1);
  const [message, setMessage] = useState('');
  const [portfolio, setPortfolio] = useState<{ positions: any[]; orders: any[]; account?: any } | null>(null);
  const [portfolioLoading, setPortfolioLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [isExecuting, setIsExecuting] = useState(false);
  const [pollingActive, setPollingActive] = useState(true);
  const [aiJsonResult, setAiJsonResult] = useState<any | null>(null);
  const [aiTextResult, setAiTextResult] = useState<string | null>(null);

  // Utilitaire pour charger le portefeuille (refactorisé avec temporisation)
  const loadPortfolio = async (showLoading: boolean = true) => {
    if (showLoading) setPortfolioLoading(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 1000)); // Pause de 1 seconde
      const res = await fetch('/api/trade/portfolio');
      const data = await res.json();
      setPortfolio(data);
      setLastUpdate(new Date());
    } catch {
      // Optionnel : afficher une erreur
    } finally {
      if (showLoading) setPortfolioLoading(false);
    }
  };

  // Rafraîchissement automatique du portefeuille toutes les minutes
  useEffect(() => {
    if (!pollingActive) return;
    const interval = setInterval(() => {
      loadPortfolio(false); // Mise à jour silencieuse
    }, 60000); // 60 000 ms = 1 minute
    return () => clearInterval(interval);
  }, [pollingActive]);

  // Récupération du portefeuille et des ordres récents
  useEffect(() => {
    loadPortfolio(true);
  }, []);

  // Soumission du trade
  const handleTrade = async () => {
    setMessage('');
    setAiJsonResult(null);
    setAiTextResult(null);
    setIsExecuting(true);
    setPollingActive(false);
    try {
      if (action === 'trade-ai') {
        const res = await fetch(`/api/trade/trade-ai?symbol=${encodeURIComponent(symbol)}`);
        const text = await res.text();
        const parts = text.split('===');
        if (parts.length >= 2) {
          try {
            setAiJsonResult(JSON.parse(parts[0].trim()));
          } catch {
            setAiJsonResult(null);
          }
          setAiTextResult(parts.slice(1).join('===').trim());
          setMessage('');
        } else {
          setAiJsonResult(null);
          setAiTextResult(null);
          setMessage(text);
        }
        await loadPortfolio(true);
        setIsExecuting(false);
        setPollingActive(true);
        return;
      }
      const res = await fetch(TRADE_API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol, action, quantity }),
      });
      const text = await res.text();
      setMessage(text);
      setAiJsonResult(null);
      setAiTextResult(null);
      await loadPortfolio(true);
    } catch (e) {
      setMessage('Erreur lors de la transaction');
      setAiJsonResult(null);
      setAiTextResult(null);
    } finally {
      setIsExecuting(false);
      setPollingActive(true);
    }
  };

  // Liste des statuts annulables Alpaca
  const cancellableStatuses = [
    'new', 'partially_filled', 'accepted', 'pending_new', 'pending_replace', 'pending_cancel'
  ];
  const [cancellingOrderId, setCancellingOrderId] = useState<string | null>(null);

  // Annulation d'un ordre
  const handleCancelOrder = async (orderId: string) => {
    setCancellingOrderId(orderId);
    setMessage('');
    try {
      const res = await fetch(`/api/trade/order/cancel/${orderId}`, { method: 'POST' });
      const text = await res.text();
      setMessage(text || 'Ordre annulé.');
      // Rafraîchir le portefeuille après annulation
      await loadPortfolio();
    } catch (e) {
      setMessage('Erreur lors de l\'annulation de l\'ordre.');
    }
    setCancellingOrderId(null);
  };

  // Liste des symboles détenus (pour la vente)
  const ownedSymbols = portfolio?.positions?.map((pos: any) => pos.symbol) || [];

  // Quand on passe en mode "sell", si le symbole courant n'est pas détenu, on sélectionne le premier symbole détenu
  useEffect(() => {
    if (action === 'sell' && ownedSymbols.length > 0 && !ownedSymbols.includes(symbol)) {
      setSymbol(ownedSymbols[0]);
    }
  }, [action, ownedSymbols]);

  return (
    <div className="trade-page">
      {/* Bloc portefeuille */}
      <div className="portfolio-block">
        <h3>Mon portefeuille</h3>
        {lastUpdate && (
          <div className="portfolio-last-update">
            Actualisé à {lastUpdate.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
          </div>
        )}
        {portfolioLoading && <div>Chargement...</div>}
        {!portfolioLoading && portfolio && (
          <>
            {portfolio.account && (
              <div className="portfolio-account portfolio-account-cards">
                <div className="account-card">
                  <div className="account-label">Valeur totale</div>
                  <div className="account-value">{Number(portfolio.account.equity).toLocaleString('fr-FR', {minimumFractionDigits: 2, maximumFractionDigits: 2})} $</div>
                </div>
                <div className="account-card">
                  <div className="account-label">Buying Power</div>
                  <div className="account-value">{Number(portfolio.account.buying_power).toLocaleString('fr-FR', {minimumFractionDigits: 2, maximumFractionDigits: 2})} $</div>
                </div>
                <div className="account-card">
                  <div className="account-label">Cash</div>
                  <div className="account-value">{Number(portfolio.account.cash).toLocaleString('fr-FR', {minimumFractionDigits: 2, maximumFractionDigits: 2})} $</div>
                </div>
                {portfolio.account.portfolio_value && (
                  <div className="account-card">
                    <div className="account-label">Portfolio Value</div>
                    <div className="account-value">{Number(portfolio.account.portfolio_value).toLocaleString('fr-FR', {minimumFractionDigits: 2, maximumFractionDigits: 2})} $</div>
                  </div>
                )}
                {portfolio.account.status && (
                  <div className="account-card">
                    <div className="account-label">Status</div>
                    <div className="account-value status-value">{portfolio.account.status}</div>
                  </div>
                )}
              </div>
            )}
            <div>
              <b>Positions&nbsp;:</b>
              {portfolio.positions.length === 0 ? (
                <div>Aucune position en cours.</div>
              ) : (
                <table className="orders-table" style={{ marginTop: 8 }}>
                  <thead>
                    <tr>
                      <th>Symbole</th>
                      <th>Prix d'achat</th>
                      <th>Prix actuel</th>
                      <th>Quantité</th>
                      <th>Gain/Perte</th>
                      <th>Total</th>
                    </tr>
                  </thead>
                  <tbody>
                    {portfolio.positions.map((pos, i) => (
                      <tr key={i}>
                        <td>{pos.symbol}</td>
                        <td>{pos.avg_entry_price !== undefined && pos.avg_entry_price !== null ? Number(pos.avg_entry_price).toFixed(2) + ' $' : '-'}</td>
                        <td>{pos.current_price !== undefined && pos.current_price !== null ? Number(pos.current_price).toFixed(2) + ' $' : '-'}</td>
                        <td>{pos.qty}</td>
                        <td>{pos.unrealized_plpc !== undefined && pos.unrealized_plpc !== null ? (Number(pos.unrealized_plpc) * 100).toFixed(3) + ' %' : '-'}</td>
                        <td>{pos.current_price !== undefined && pos.current_price !== null ? (Number(pos.qty) * Number(pos.current_price)).toFixed(2) + ' $' : '-'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
            <div style={{ marginTop: 8 }}>
              <b>Ordres récents&nbsp;:</b>
              {portfolio.orders.length === 0 ? (
                <div>Aucun ordre récent.</div>
              ) : (
                (() => {
                  // Vérifier s'il y a au moins une action annulable
                  const hasCancellable = portfolio.orders.some(order => order.id && cancellableStatuses.includes(order.status));
                  return (
                    <table className="orders-table">
                      <thead>
                        <tr>
                          <th>Action</th>
                          <th>Symbole</th>
                          <th>Quantité</th>
                          <th>Prix</th>
                          <th>Statut</th>
                          {hasCancellable && <th></th>}
                        </tr>
                      </thead>
                      <tbody>
                        {portfolio.orders.map((order, i) => (
                          <tr key={i} className={order.side === 'buy' ? 'buy-row' : order.side === 'sell' ? 'sell-row' : ''}>
                            <td className="status">{order.side}</td>
                            <td>{order.symbol}</td>
                            <td>{order.qty}</td>
                            <td>{order.filled_avg_price ? Number(order.filled_avg_price).toLocaleString('fr-FR', {minimumFractionDigits: 2, maximumFractionDigits: 2}) + ' $' : (order.limit_price ? Number(order.limit_price).toLocaleString('fr-FR', {minimumFractionDigits: 2, maximumFractionDigits: 2}) + ' $' : '-')}</td>
                            <td className="status">{order.status}</td>
                            {hasCancellable && (
                              <td>
                                {order.id && cancellableStatuses.includes(order.status) && (
                                  <button
                                    style={{ padding: '4px 10px', fontSize: '0.95em' }}
                                    disabled={cancellingOrderId === order.id}
                                    onClick={() => handleCancelOrder(order.id)}
                                  >
                                    {cancellingOrderId === order.id ? 'Annulation...' : 'Annuler'}
                                  </button>
                                )}
                              </td>
                            )}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  );
                })()
              )}
            </div>
          </>
        )}
      </div>
      <h2>Trade</h2>
      <div>
        <label>Action&nbsp;
          <select value={action} onChange={e => setAction(e.target.value as 'buy' | 'sell' | 'trade-ai')}>
            <option value="buy">Acheter</option>
            <option value="sell">Vendre</option>
            <option value="trade-ai">Ttrade AI</option>
          </select>
        </label>
      </div>
      {action === 'buy' ? (
        <div>
          <label>Symbole&nbsp;
            <input value={symbol} onChange={e => setSymbol(e.target.value.toUpperCase())} maxLength={8} />
          </label>
        </div>
      ) : action === 'sell' ? (
        <div>
          <label>Symbole&nbsp;
            <select value={symbol} onChange={e => setSymbol(e.target.value)}>
              {ownedSymbols.length === 0 ? (
                <option value="">Aucune position</option>
              ) : (
                ownedSymbols.map((sym: string) => (
                  <option key={sym} value={sym}>{sym}</option>
                ))
              )}
            </select>
          </label>
        </div>
      ) : (
        <div>
          <label>Symbole&nbsp;
            <input value={symbol} onChange={e => setSymbol(e.target.value.toUpperCase())} maxLength={8} />
          </label>
        </div>
      )}
      {action !== 'trade-ai' && (
        <div>
          <label>Quantité&nbsp;
            <input type="number" min={1} value={quantity} onChange={e => setQuantity(Number(e.target.value))} />
          </label>
        </div>
      )}
      <button onClick={handleTrade} disabled={isExecuting}>
        {isExecuting ? <span className="spinner" style={{marginRight: 8}}></span> : null}
        {isExecuting ? 'Exécution...' : 'Exécuter'}
      </button>
      {aiJsonResult && (
        <div className="trade-ai-json-result" style={{marginTop: 16}}>
          <b style={{fontSize: '1.1em'}}>Résultat AI :</b>
          <table style={{marginTop: 8, borderCollapse: 'collapse', background: '#f8fafd', border: '1px solid #e0e0e0', borderRadius: 6, minWidth: 220, boxShadow: '0 1px 4px #e0e0e0'}}>
            <tbody>
              {Object.entries(aiJsonResult).map(([key, value]) => (
                <tr key={key}>
                  <td style={{fontWeight: 'bold', padding: '6px 16px 6px 8px', borderBottom: '1px solid #e0e0e0', textTransform: 'capitalize', background: '#f0f4f8'}}>{key}</td>
                  <td style={{padding: '6px 12px', borderBottom: '1px solid #e0e0e0'}}>{String(value)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {aiTextResult && (
        <div className="trade-ai-text-result" style={{marginTop: 12, whiteSpace: 'pre-wrap', background: '#f6f6f6', padding: 10, borderRadius: 4}}>
          {aiTextResult}
        </div>
      )}
      {!aiJsonResult && !aiTextResult && message && <div className="trade-message">{message}</div>}
    </div>
  );
};

export default TradePage;
