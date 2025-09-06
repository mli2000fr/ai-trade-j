import React, { useEffect, useState } from 'react';
import PortfolioBlock from '../composants/PortfolioBlock';
import TradeAlertReal from '../composants/TradeAlertReal';
import TradeCompteSelect from '../composants/TradeCompteSelect';
import TradeManualBlock from '../composants/TradeManualBlock';
import TradeAutoBlock from '../composants/TradeAutoBlock';
import OrdersTable from '../composants/OrdersTable';
import { useSelectedCompte } from '../SelectedCompteContext';
import Button from '@mui/material/Button';
import Paper from '@mui/material/Paper';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';

const DashboardPage: React.FC = () => {
  const { comptes, selectedCompteId, setSelectedCompteId, comptesLoading, comptesError } = useSelectedCompte();
  // États pour le portefeuille
  const [portfolio, setPortfolio] = useState<any | null>(null);
  const [portfolioLoading, setPortfolioLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  // --- États pour trade manuel/auto ---
  const [action, setAction] = useState<'buy' | 'sell'>('buy');
  const [symbol, setSymbol] = useState('');
  const [quantity, setQuantity] = useState(1);
  const [messageManual, setMessageManual] = useState('');
  const [isExecutingManual, setIsExecutingManual] = useState(false);
  const [cancelOpposite, setCancelOpposite] = useState(false);
  const [forceDayTrade, setForceDayTrade] = useState(false);
  const [stopLoss, setStopLoss] = useState<number | ''>('');
  const [takeProfit, setTakeProfit] = useState<number | ''>('');
  const [autoSymbols, setAutoSymbols] = useState('');
  const [isExecutingAuto, setIsExecutingAuto] = useState(false);
  const [analyseGptText, setAnalyseGptText] = useState('');
  const [messageAuto, setMessageAuto] = useState('');
  const [aiJsonResult, setAiJsonResult] = useState<any | null>(null);
  const [aiTextResult, setAiTextResult] = useState<string | null>(null);
  const [idGpt, setIdGpt] = useState<string | null>(null);

  // --- États pour les ordres ---
  const [orders, setOrders] = useState<any[]>([]);
  const [ordersLoading, setOrdersLoading] = useState(false);
  const [filterSymbol, setFilterSymbol] = useState<string>('');
  const [filterCancelable, setFilterCancelable] = useState<boolean>(false);
  const [ordersSize, setOrdersSize] = useState<number>(10);
  const [cancellingOrderId, setCancellingOrderId] = useState<string | null>(null);
  const [cancelMessage, setCancelMessage] = useState<string>('');
  const [resetSellErrorKey, setResetSellErrorKey] = useState<number>(0);

  // --- Utilitaires pour trade manuel/auto ---
  const ownedSymbols = portfolio?.positions?.map((pos: any) => pos.symbol) || [];
  const positions = portfolio?.positions || [];

  // Met à jour autoSymbols quand le portefeuille change
  useEffect(() => {
    setAutoSymbols(portfolio?.positions?.map((pos: any) => pos.symbol).join(',') || '');
  }, [portfolio]);

  // Chargement du portefeuille (appelable manuellement)
  const loadPortfolio = async () => {
    if (!selectedCompteId) return;
    setPortfolioLoading(true);
    try {
      console.log('Appel backend portefeuille', selectedCompteId);
      const res = await fetch(`/api/trade/portfolio?id=${encodeURIComponent(selectedCompteId)}`);
      const data = await res.json();
      setPortfolio(data);
      setLastUpdate(new Date());
    } catch {
      setPortfolio(null);
    } finally {
      setPortfolioLoading(false);
    }
  };

  // Handler trade manuel
  const handleTradeManual = async () => {
    setMessageManual('');
    setIsExecutingManual(true);
    try {
      const res = await fetch('/api/trade/trade', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol, action, quantity, id: selectedCompteId, cancelOpposite, forceDayTrade, stopLoss, takeProfit }),
      });
      const text = await res.text();
      setMessageManual(text);
      setStopLoss('');
      setTakeProfit('');
      // Rafraîchir portefeuille
      if (selectedCompteId) {
        const res = await fetch(`/api/trade/portfolio?id=${encodeURIComponent(selectedCompteId)}`);
        const data = await res.json();
        setPortfolio(data);
        setLastUpdate(new Date());
      }
    } catch {
      setMessageManual('Erreur lors de la transaction');
    } finally {
      setIsExecutingManual(false);
    }
  };

  // Handler trade auto
  const handleTradeAuto = async () => {
    setMessageAuto('');
    setIsExecutingAuto(true);
    setAiJsonResult(null);
    setAiTextResult(null);
    try {
      const symboles = autoSymbols.split(',').map(s => s.trim()).filter(Boolean);
      const res = await fetch('/api/trade/trade-ai-auto', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symboles, id: selectedCompteId, analyseGpt: analyseGptText }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setAiJsonResult(data.orders || null);
      setAiTextResult(data.analyseGpt || null);
      setIdGpt(data.idGpt || data.id || null);
      setMessageAuto('Trade auto exécuté avec succès');
      // Rafraîchir portefeuille
      if (selectedCompteId) {
        const res = await fetch(`/api/trade/portfolio?id=${encodeURIComponent(selectedCompteId)}`);
      setAiTextResult(data.analyseGpt || null);
      setIdGpt(data.idGpt || data.id || null);
        setLastUpdate(new Date());
      }
    } catch (e: any) {
      setMessageAuto(e?.message || 'Erreur lors de la transaction auto');
      setAiJsonResult(null);
      setAiTextResult(null);
      setIdGpt(null);
    } finally {
      setIsExecutingAuto(false);
    }
  };

  // --- Fonctions pour les ordres ---
  const cancellableStatuses = [
    'new', 'partially_filled', 'accepted', 'pending_new', 'pending_replace', 'pending_cancel'
  ];
  const loadOrders = async () => {
    setOrdersLoading(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 500));
      const params = [];
      if (filterSymbol) params.push(`symbol=${encodeURIComponent(filterSymbol)}`);
      if (filterCancelable) params.push(`cancelable=true`);
      if (selectedCompteId) params.push(`id=${encodeURIComponent(selectedCompteId)}`);
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
  const handleCancelOrder = async (orderId: string) => {
    setCancelMessage('');
    setCancellingOrderId(orderId);
    try {
      const res = await fetch(`/api/trade/order/cancel/${selectedCompteId}/${orderId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      const text = await res.text();
      setCancelMessage(text || 'Ordre annulé.');
      await loadOrders();
    } catch (e) {
      setCancelMessage("Erreur lors de l'annulation de l'ordre.");
    }
    setCancellingOrderId(null);
  };
  useEffect(() => {
    if (selectedCompteId) {
      loadOrders();
    }
    // eslint-disable-next-line
  }, [filterSymbol, filterCancelable, ordersSize, selectedCompteId]);

  const selectedCompte = comptes.find(c => c.id === selectedCompteId);

  // useEffect pour charger le portefeuille après chargement des comptes et sélection du compte
  useEffect(() => {
    if (!comptesLoading && selectedCompteId) {
      loadPortfolio();
    }
    // eslint-disable-next-line
  }, [selectedCompteId, comptesLoading]);

  useEffect(() => {
    setAiJsonResult(null);
    setAiTextResult(null);
    setMessageManual('');
    setMessageAuto('');
    setCancelMessage('');
    setResetSellErrorKey(prev => prev + 1); // incrémente la clé à chaque changement de compte
  }, [selectedCompteId]);

  return (
    <Box sx={{
      bgcolor: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
      minHeight: '100vh',
      p: 3
    }}>
      <Box maxWidth={1100} mx="auto">
        <Paper elevation={3} sx={{ mb: 3, p: 2, borderRadius: 3 }}>
          <TradeCompteSelect
            comptes={comptes}
            selectedCompteId={selectedCompteId}
            loading={comptesLoading}
            error={comptesError}
            onSelect={setSelectedCompteId}
          />
          {selectedCompte?.real && <TradeAlertReal />}
        </Paper>
        <Paper elevation={3} sx={{ mb: 3, p: 2, borderRadius: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
            <Typography variant="h6">Mon portefeuille</Typography>
            <Button variant="outlined" size="small" onClick={loadPortfolio}>
              Rafraîchir
            </Button>
          </Box>
          <PortfolioBlock
            portfolio={portfolio}
            lastUpdate={lastUpdate}
            loading={portfolioLoading}
            compteId={selectedCompteId ? String(selectedCompteId) : null}
            resetSellErrorKey={resetSellErrorKey}
          />
        </Paper>
        <Paper elevation={3} sx={{ mb: 3, p: 2, borderRadius: 3 }}>
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
            positions={positions}
            ordersSize={ordersSize}
            onOrdersSizeChange={setOrdersSize}
            cancelMessage={cancelMessage}
          />
        </Paper>
        <Paper elevation={3} sx={{ mb: 3, p: 2, borderRadius: 3 }}>
          <TradeManualBlock
            action={action}
            symbol={symbol}
            quantity={quantity}
            ownedSymbols={ownedSymbols}
            isExecuting={isExecutingManual}
            cancelOpposite={cancelOpposite}
            forceDayTrade={forceDayTrade}
            onChangeAction={setAction}
            onChangeSymbol={setSymbol}
            onChangeQuantity={setQuantity}
            onTrade={handleTradeManual}
            onChangeCancelOpposite={setCancelOpposite}
            onChangeForceDayTrade={setForceDayTrade}
            stopLoss={stopLoss}
            takeProfit={takeProfit}
            onChangeStopLoss={setStopLoss}
            onChangeTakeProfit={setTakeProfit}
            message={messageManual}
            positions={positions}
          />
        </Paper>
        <Paper elevation={3} sx={{ mb: 3, p: 2, borderRadius: 3 }}>
          <TradeAutoBlock
            autoSymbols={autoSymbols}
            isExecuting={isExecutingAuto}
            onChange={setAutoSymbols}
            onTrade={handleTradeAuto}
            analyseGptText={analyseGptText}
            onAnalyseGptChange={setAnalyseGptText}
            message={messageAuto}
            aiJsonResult={aiJsonResult}
            aiTextResult={aiTextResult}
            compteId={selectedCompteId || ''}
            idGpt={idGpt || undefined}
          />
        </Paper>
      </Box>
    </Box>
  );
};

export default DashboardPage;
