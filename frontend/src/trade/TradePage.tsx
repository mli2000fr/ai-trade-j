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
  const [autoSymbols, setAutoSymbols] = useState<string>('');

  // --- Gestion des comptes ---
  const [comptes, setComptes] = useState<{ id: number; nom: string; alias?: string; real?: boolean }[]>([]);
  const [selectedCompteId, setSelectedCompteId] = useState<number | null>(null);
  const [comptesLoading, setComptesLoading] = useState(true);
  const [comptesError, setComptesError] = useState<string | null>(null);

  useEffect(() => {
    const fetchComptes = async () => {
      setComptesLoading(true);
      setComptesError(null);
      try {
        const res = await fetch('/api/trade/comptes');
        if (!res.ok) throw new Error('Erreur lors du chargement des comptes');
        const data = await res.json();
        setComptes(Array.isArray(data) ? data : []);
        if (Array.isArray(data) && data.length > 0) setSelectedCompteId(data[0].id);
      } catch (e) {
        setComptesError('Impossible de charger les comptes');
        setComptes([]);
      } finally {
        setComptesLoading(false);
      }
    };
    fetchComptes();
  }, []);

  // Utilitaire pour obtenir l'id du compte sélectionné
  const selectedCompte = comptes.find(c => c.id === selectedCompteId);
  const compteId = selectedCompte?.id || '';

  // Utilitaire pour charger le portefeuille (refactorisé avec temporisation)
  const loadPortfolio = async (showLoading: boolean = true) => {
    if (showLoading) setPortfolioLoading(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 1000));
      const res = await fetch(`/api/trade/portfolio?id=${encodeURIComponent(compteId)}`);
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
    if (!pollingActive || !selectedCompteId) return;
    const interval = setInterval(() => {
      loadPortfolio(false); // Mise à jour silencieuse
    }, 60000); // 60 000 ms = 1 minute
    return () => clearInterval(interval);
  }, [pollingActive, selectedCompteId]);

  // Charger le portefeuille et les ordres après sélection du compte (et donc après chargement des comptes)
  useEffect(() => {
    if (selectedCompteId !== null) {
      loadPortfolio(true);
      loadOrders();
      setAiJsonResult(null); // Vider le bloc Résultat AI
      setAiTextResult(null); // Vider le bloc Résultat AI
      setMessage(''); // Vider le message
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedCompteId]);

  // Met à jour autoSymbols quand le portefeuille change
  useEffect(() => {
    setAutoSymbols(portfolio?.positions?.map((pos: any) => pos.symbol).join(',') || '');
  }, [portfolio]);

  // Soumission du trade
  const handleTrade = async () => {
    setMessage('');
    setAiJsonResult(null);
    setAiTextResult(null);
    setIsExecuting(true);
    setPollingActive(false);
    try {
      if (action === 'trade-ai') {
        const res = await fetch('/api/trade/trade-ai', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ symbol, id: compteId }),
        });
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
        await loadOrders();
        setIsExecuting(false);
        setPollingActive(true);
        return;
      }
      const res = await fetch(TRADE_API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol, action, quantity, id: compteId }),
      });
      const text = await res.text();
      setMessage(text);
      setAiJsonResult(null);
      setAiTextResult(null);
      await loadPortfolio(true);
      await loadOrders();
    } catch (e) {
      setMessage('Erreur lors de la transaction');
      setAiJsonResult(null);
      setAiTextResult(null);
    } finally {
      setIsExecuting(false);
      setPollingActive(true);
    }
  };

  // Soumission du trade auto
  const handleTradeAuto = async () => {
    setMessage('');
    setAiJsonResult(null);
    setAiTextResult(null);
    setIsExecuting(true);
    setPollingActive(false);
    try {
      const symboles = autoSymbols.split(',').map(s => s.trim()).filter(Boolean);
      const res = await fetch('/api/trade/trade-ai-auto', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symboles, id: compteId }),
      });
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
      await loadOrders();
    } catch (e) {
      setMessage('Erreur lors de la transaction auto');
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
      const res = await fetch(`/api/trade/order/cancel/${compteId}/${orderId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      const text = await res.text();
      setMessage(text || 'Ordre annulé.');
      // Rafraîchir le portefeuille après annulation
      await loadPortfolio();
      // Rafraîchir le tableau des ordres après annulation
      await loadOrders();
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

  // Filtres pour les ordres récents
  const [filterSymbol, setFilterSymbol] = useState<string>('');
  const [filterCancelable, setFilterCancelable] = useState<boolean>(false);

  // Ajout état pour les ordres récupérés via l'API
  const [orders, setOrders] = useState<any[]>([]);
  const [ordersLoading, setOrdersLoading] = useState(false);

  // Fonction pour charger les ordres depuis l'API
  const loadOrders = async () => {
    setOrdersLoading(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 1000));
      const params = [];
      if (filterSymbol) params.push(`symbol=${encodeURIComponent(filterSymbol)}`);
      if (filterCancelable) params.push(`cancelable=true`);
      if (compteId) params.push(`id=${encodeURIComponent(compteId)}`);
      const url = '/api/trade/orders' + (params.length ? `?${params.join('&')}` : '');
      const res = await fetch(url);
      const data = await res.json();
      setOrders(Array.isArray(data) ? data : []);
    } catch {
      setOrders([]);
    } finally {
      setOrdersLoading(false);
    }
  };

  return (
    <div className="trade-page">
      {/* Alerte compte réel */}
      {selectedCompte && selectedCompte.real === true && (
        <div className="trade-alert-real">
          Attention : Vous êtes connecté à un <b>compte RÉEL</b> ! Toutes les opérations sont effectives sur le marché réel.
        </div>
      )}
      {/* Liste déroulante des comptes */}
      <div className="trade-comptes-select">
        <label className="trade-comptes-label">Compte&nbsp;:</label>
        {comptesLoading ? (
          <span>Chargement des comptes...</span>
        ) : comptesError ? (
          <span className="trade-comptes-error">{comptesError}</span>
        ) : (
          <select
            value={selectedCompteId ?? ''}
            onChange={e => setSelectedCompteId(Number(e.target.value))}
            className="trade-comptes-dropdown"
          >
            {comptes.map(compte => (
              <option key={compte.id} value={compte.id}>
                {compte.nom}
                {compte.alias ? ` (${compte.alias})` : ''}
                {compte.real === true ? ' [REAL]' : compte.real === false ? ' [PAPER]' : ''}
              </option>
            ))}
          </select>
        )}
      </div>
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
                <table className="orders-table portfolio-positions-table">
                  <thead>
                    <tr>
                      <th>Symbole</th>
                      <th>Prix d'achat</th>
                      <th>Prix actuel</th>
                      <th>Quantité</th>
                      <th>Total</th>
                      <th>P & L pc</th>
                      <th>P & L</th>
                    </tr>
                  </thead>
                  <tbody>
                    {portfolio.positions.map((pos, i) => (
                      <tr key={i}>
                        <td>{pos.symbol}</td>
                        <td>{pos.avg_entry_price !== undefined && pos.avg_entry_price !== null ? Number(pos.avg_entry_price).toFixed(2) + ' $' : '-'}</td>
                        <td>{pos.current_price !== undefined && pos.current_price !== null ? Number(pos.current_price).toFixed(2) + ' $' : '-'}</td>
                        <td>{pos.qty}</td>
                        <td>{pos.current_price !== undefined && pos.current_price !== null ? (Number(pos.qty) * Number(pos.current_price)).toFixed(2) + ' $' : '-'}</td>
                        <td>{pos.unrealized_plpc !== undefined && pos.unrealized_plpc !== null ? (Number(pos.unrealized_plpc) * 100).toFixed(3) + ' %' : '-'}</td>
                        <td>{pos.unrealized_pl !== undefined && pos.unrealized_pl !== null ? Number(pos.unrealized_pl).toFixed(2) + ' $' : '-'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
            <div className="trade-orders-recent">
              <b>Ordres récents&nbsp;:</b>
              {/* Filtres pour les ordres récents */}
              <div className="trade-orders-filter-bar">
                <label className="trade-orders-filter-label">
                  Filtrer par symbole&nbsp;
                  <select value={filterSymbol} onChange={e => setFilterSymbol(e.target.value)}>
                    <option value="">Tous</option>
                    {portfolio.positions
                      .map(pos => pos.symbol)
                      .filter((symbol, idx, arr) => arr.indexOf(symbol) === idx)
                      .map(symbol => (
                        <option key={symbol} value={symbol}>{symbol}</option>
                      ))}
                  </select>
                </label>
                <label className="trade-orders-filter-checkbox">
                  <input type="checkbox" checked={filterCancelable} onChange={e => setFilterCancelable(e.target.checked)} />
                  Annulables uniquement
                </label>
                <button onClick={loadOrders} disabled={ordersLoading} className="trade-orders-update-btn">
                  {ordersLoading ? 'Mise à jour...' : 'Update'}
                </button>
              </div>
              {/* Affichage du tableau d'ordres récupérés via l'API */}
              {orders.length === 0 && !ordersLoading ? (
                <div>Aucun ordre à afficher.</div>
              ) : ordersLoading ? (
                <div>Chargement des ordres...</div>
              ) : (
                (() => {
                  const hasCancellable = orders.some(order => order.id && cancellableStatuses.includes(order.status));
                  if (orders.length === 0) return <div>Aucun ordre ne correspond aux filtres.</div>;
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
                        {orders.map((order, i) => (
                          <tr key={i} className={order.side === 'buy' ? 'buy-row' : order.side === 'sell' ? 'sell-row' : ''}>
                            <td className="status">{order.side}</td>
                            <td>{order.symbol}</td>
                            <td>{order.qty}</td>
                            <td>{order.filledAvgPrice !== undefined && order.filledAvgPrice !== null ? Number(order.filledAvgPrice).toLocaleString('fr-FR', {minimumFractionDigits: 2, maximumFractionDigits: 2}) + ' $' : (order.limit_price !== undefined && order.limit_price !== null ? Number(order.limit_price).toLocaleString('fr-FR', {minimumFractionDigits: 2, maximumFractionDigits: 2}) + ' $' : '-')}</td>
                            <td className="status">{order.status}</td>
                            {hasCancellable && (
                              <td>
                                {order.id && cancellableStatuses.includes(order.status) && (
                                  <button
                                    className="trade-cancel-btn"
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
      <div className="trade-auto-block">
        <h2 className="trade-auto-title">Trade Auto</h2>
        <div className="trade-auto-input-row">
          <label className="trade-auto-label" htmlFor="auto-symbols">Symboles&nbsp;</label>
          <input
            id="auto-symbols"
            type="text"
            className="trade-auto-input"
            value={autoSymbols}
            onChange={e => setAutoSymbols(e.target.value)}
            placeholder="AAPL,KO,NVDA,TSLA,AMZN,MSFT,AMD,META,SHOP,PLTR"
          />
        </div>
        <div className="trade-auto-btn-row">
          <button onClick={handleTradeAuto} disabled={isExecuting || !autoSymbols.trim()} className="trade-auto-btn">
            {isExecuting ? <span className="spinner trade-spinner"></span> : null}
            {isExecuting ? 'Exécution...' : 'Exécuter'}
          </button>
        </div>
      </div>
      <div className="trade-manual-block">
        <h2 className="trade-manual-title">Trade Manuel</h2>
        <div className="trade-manual-input-row">
          <label className="trade-manual-label">Action&nbsp;
            <select value={action} onChange={e => setAction(e.target.value as 'buy' | 'sell' | 'trade-ai')}>
              <option value="buy">Acheter</option>
              <option value="sell">Vendre</option>
              <option value="trade-ai">Ttrade AI</option>
            </select>
          </label>
          {action === 'buy' || action === 'trade-ai' ? (
            <label className="trade-manual-label">Symbole&nbsp;&nbsp;
              <input value={symbol} onChange={e => setSymbol(e.target.value.toUpperCase())} maxLength={8} className="trade-manual-input" />
            </label>
          ) : (
            <label className="trade-manual-label">Symbole&nbsp;
              <select value={symbol} onChange={e => setSymbol(e.target.value)} className="trade-manual-input">
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
              <input type="number" min={0.01} step="any" value={quantity} onChange={e => setQuantity(parseFloat(e.target.value))} className="trade-manual-input" />
            </label>
          )}
        </div>
        <div className="trade-manual-btn-row">
          <button onClick={handleTrade} disabled={isExecuting} className="trade-manual-btn">
            {isExecuting ? <span className="spinner trade-spinner"></span> : null}
            {isExecuting ? 'Exécution...' : 'Exécuter'}
          </button>
        </div>
      </div>
      {aiJsonResult && (
        <div className="trade-ai-json-result">
          <b className="trade-ai-json-title">Résultat AI :</b>
          {Array.isArray(aiJsonResult) ? (
            (() => {
              // Vérifier si au moins un élément a une valeur pour price_limit ou priceLimit
              const hasPriceLimit = aiJsonResult.some((item: any) => item.price_limit !== undefined && item.price_limit !== null && item.price_limit !== '' || item.priceLimit !== undefined && item.priceLimit !== null && item.priceLimit !== '');
              return (
                <div style={{overflowX: 'auto'}}>
                  <table style={{marginTop: 8, borderCollapse: 'collapse', background: '#f8fafd', border: '1px solid #e0e0e0', borderRadius: 6, minWidth: 600, boxShadow: '0 1px 4px #e0e0e0', width: '100%'}}>
                    <thead style={{background: '#e3eaf3'}}>
                      <tr>
                        <th style={{padding: '8px 12px'}}>Symbole</th>
                        <th style={{padding: '8px 12px'}}>Action</th>
                        <th style={{padding: '8px 12px'}}>Quantité</th>
                        {hasPriceLimit && <th style={{padding: '8px 12px'}}>Prix limite</th>}
                        <th style={{padding: '8px 12px'}}>Stop loss</th>
                        <th style={{padding: '8px 12px'}}>Take profit</th>
                      </tr>
                    </thead>
                    <tbody>
                      {aiJsonResult.map((item: any, idx: number) => (
                        <tr key={idx} style={{background: idx % 2 === 0 ? '#f8fafd' : '#f0f4f8'}}>
                          <td style={{padding: '6px 12px', fontWeight: 500}}>{item.symbol}</td>
                          <td style={{padding: '6px 12px', textTransform: 'capitalize'}}>{item.action}</td>
                          <td style={{padding: '6px 12px', textAlign: 'right'}}>{item.quantity ?? item.qty ?? ''}</td>
                          {hasPriceLimit && (
                            <td style={{padding: '6px 12px', textAlign: 'right'}}>{item.price_limit ?? item.priceLimit ? (item.price_limit ?? item.priceLimit) + ' $' : '-'}</td>
                          )}
                          <td style={{padding: '6px 12px', textAlign: 'right'}}>{item.stop_loss ?? item.stopLoss ? (item.stop_loss ?? item.stopLoss) + ' $' : '-'}</td>
                          <td style={{padding: '6px 12px', textAlign: 'right'}}>{item.take_profit ?? item.takeProfit ? (item.take_profit ?? item.takeProfit) + ' $' : '-'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              );
            })()
          ) : (
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
          )}
        </div>
      )}
      {aiTextResult && (
        <div className="trade-ai-text-result" style={{marginTop: 12, whiteSpace: 'pre-wrap', background: '#f6f6f6', padding: 10, borderRadius: 4, width: '100%'}}>
          {aiTextResult}
        </div>
      )}
      {!aiJsonResult && !aiTextResult && message && <div className="trade-message" style={{width: '100%'}}>{message}</div>}

    </div>
  );
};

export default TradePage;
