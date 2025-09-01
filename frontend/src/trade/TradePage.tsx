import React, { useState, useEffect } from 'react';
import './TradePage.css';
import TradeAlertReal from './composants/TradeAlertReal';
import TradeCompteSelect from './composants/TradeCompteSelect';
import PortfolioBlock from './composants/PortfolioBlock';
import TradeAutoBlock from './composants/TradeAutoBlock';
import TradeManualBlock from './composants/TradeManualBlock';
import OrdersTable from './composants/OrdersTable';
import TradeAIResults from './composants/TradeAIResults';
import Box from '@mui/material/Box';

const TRADE_API_URL = '/api/trade/trade';

const TradePage: React.FC = () => {
  const [symbol, setSymbol] = useState('');
  const [action, setAction] = useState<'buy' | 'sell'>('buy');
  const [quantity, setQuantity] = useState(1);
  const [message, setMessage] = useState('');
  const [portfolio, setPortfolio] = useState<{ positions: any[]; orders: any[]; account?: any } | null>(null);
  const [portfolioLoading, setPortfolioLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [isExecuting, setIsExecuting] = useState(false);
  const [pollingActive, setPollingActive] = useState(true);
  const [aiJsonResult, setAiJsonResult] = useState<any | null>(null);
  const [aiTextResult, setAiTextResult] = useState<string | null>(null);
  const [idGpt, setIdGpt] = useState<string | null>(null);
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
      const res = await fetch(TRADE_API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol, action, quantity, id: compteId, cancelOpposite, forceDayTrade, stopLoss: action === 'buy' ? stopLoss || null : null, takeProfit: action === 'buy' ? takeProfit || null : null }),
      });
      const text = await res.text();
      setMessage(text);
      setAiJsonResult(null);
      setAiTextResult(null);
      setCancelOpposite(false);
      setForceDayTrade(false);
      setStopLoss('');
      setTakeProfit('');
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

  // Etats pour le trade auto
  const [analyseGptFile, setAnalyseGptFile] = useState<File | null>(null);
  const [analyseGptText, setAnalyseGptText] = useState<string>('');

  // Soumission du trade auto
  const handleTradeAuto = async () => {
    setMessage('');
    setAiJsonResult(null);
    setAiTextResult(null);
    setIsExecuting(true);
    setPollingActive(false);
    try {
      const symboles = autoSymbols.split(',').map(s => s.trim()).filter(Boolean);
      let analyseGpt = analyseGptText;
      if (analyseGptFile && !analyseGptText) {
        analyseGpt = await new Promise<string>(resolve => {
          const reader = new FileReader();
          reader.onload = ev => resolve(ev.target?.result as string || '');
          reader.readAsText(analyseGptFile);
        });
      }
      const res = await fetch('/api/trade/trade-ai-auto', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symboles, id: compteId, analyseGpt }),
      });
      if (!res.ok) throw new Error('Erreur lors de la transaction auto');
      const data = await res.json();
      setAiJsonResult(data.orders || null);
      setAiTextResult(data.analyseGpt || null);
      setIdGpt(data.idGpt || data.id || null); // Extraction de l'id GPT pour le trade auto
      setMessage('');
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
  const [ordersSize, setOrdersSize] = useState<number>(10);

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
      if (ordersSize) params.push(`sizeOrders=${ordersSize}`);
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

  // Etat pour gérer la case à cocher 'cancel opposite'
  const [cancelOpposite, setCancelOpposite] = useState(false);
  const [forceDayTrade, setForceDayTrade] = useState(false);
  const [stopLoss, setStopLoss] = useState<number | ''>('');
  const [takeProfit, setTakeProfit] = useState<number | ''>('');

  return (
    <Box sx={{ maxWidth: 1100, mx: 'auto', p: { xs: 1, sm: 2, md: 3 } }}>
      {/* Alerte compte réel */}
      {selectedCompte && selectedCompte.real === true && <TradeAlertReal />}
      {/* Liste déroulante des comptes */}
      <TradeCompteSelect
        comptes={comptes}
        selectedCompteId={selectedCompteId}
        loading={comptesLoading}
        error={comptesError}
        onSelect={setSelectedCompteId}
      />
      {/* Bloc portefeuille */}
      <PortfolioBlock
        portfolio={portfolio}
        lastUpdate={lastUpdate}
        loading={portfolioLoading}
      />
      {/* Tableau des ordres */}
      <OrdersTable
        orders={orders}
        loading={ordersLoading}
        filterSymbol={filterSymbol}
        filterCancelable={filterCancelable}
        onFilterSymbol={setFilterSymbol}
        onFilterCancelable={setFilterCancelable}
        onUpdate={loadOrders}
        onCancel={handleCancelOrder}
        cancellingOrderId={cancellingOrderId}
        cancellableStatuses={cancellableStatuses}
        positions={portfolio?.positions || []}
        ordersSize={ordersSize}
        onOrdersSizeChange={setOrdersSize}
      />
      {/* Bloc trade auto */}
      <TradeAutoBlock
        autoSymbols={autoSymbols}
        isExecuting={isExecuting}
        onChange={setAutoSymbols}
        onTrade={handleTradeAuto}
        analyseGptText={analyseGptText}
        onAnalyseGptChange={setAnalyseGptText}
      />
      {/* Bloc trade manuel */}
      <TradeManualBlock
        action={action}
        symbol={symbol}
        quantity={quantity}
        ownedSymbols={ownedSymbols}
        isExecuting={isExecuting}
        cancelOpposite={cancelOpposite}
        forceDayTrade={forceDayTrade}
        onChangeAction={setAction}
        onChangeSymbol={setSymbol}
        onChangeQuantity={setQuantity}
        onTrade={handleTrade}
        onChangeCancelOpposite={setCancelOpposite}
        onChangeForceDayTrade={setForceDayTrade}
        stopLoss={stopLoss}
        takeProfit={takeProfit}
        onChangeStopLoss={setStopLoss}
        onChangeTakeProfit={setTakeProfit}
      />
      {/* Résultats AI et messages */}
      <TradeAIResults
        aiJsonResult={aiJsonResult}
        aiTextResult={aiTextResult}
        message={message}
        compteId={compteId}
        onOrdersUpdate={async () => {
          await loadPortfolio(true);
          await loadOrders();
        }}
        idGpt={idGpt ?? undefined}
      />
    </Box>
  );
};

export default TradePage;
