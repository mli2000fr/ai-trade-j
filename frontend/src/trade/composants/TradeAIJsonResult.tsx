import React from 'react';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Button from '@mui/material/Button';
import Checkbox from '@mui/material/Checkbox';
import { useState } from 'react';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';

interface TradeAIJsonResultProps {
  aiJsonResult: any;
  compteId?: string | number;
  onOrdersUpdate?: (orders: any[]) => void;
  idGpt?: string;
}

  const FailedStatuses = [
    'canceled', 'rejected', 'expired', 'failed'
  ];

const TradeAIJsonResult: React.FC<TradeAIJsonResultProps> = ({ aiJsonResult, compteId, onOrdersUpdate, idGpt }) => {
  const [orders, setOrders] = useState<any[]>(Array.isArray(aiJsonResult) ? aiJsonResult.map((o: any) => ({ ...o })) : []);
  const [loading, setLoading] = useState(false);
  const [executed, setExecuted] = useState(false);
  const [alertMsg, setAlertMsg] = useState<string | null>(null);

  // Met à jour l'état local si aiJsonResult change
  React.useEffect(() => {
    if (Array.isArray(aiJsonResult)) {
      setOrders(aiJsonResult.map((o: any) => ({ ...o })));
    }
  }, [aiJsonResult]);

  // Validation à chaque modification d'un input stoploss/takeprofit
  React.useEffect(() => {
    if (!orders || orders.length === 0) {
      setAlertMsg(null);
      return;
    }
    for (const order of orders) {
      const stop = Number(order.stop_loss ?? order.stopLoss ?? 0);
      const take = Number(order.take_profit ?? order.takeProfit ?? 0);
      if ((stop > 0 && take === 0) || (take > 0 && stop === 0)) {
        setAlertMsg("Erreur : Si stoploss ou takeprofit est renseigné (>0), l'autre doit aussi être renseigné (>0) sur la même ligne.");
        return;
      }
    }
    setAlertMsg(null);
  }, [orders]);

  if (!Array.isArray(orders) || orders.length === 0) return null;
  const hasPriceLimit = orders.some((item: any) => item.price_limit !== undefined && item.price_limit !== null && item.price_limit !== '' || item.priceLimit !== undefined && item.priceLimit !== null && item.priceLimit !== '');
  //const hasStatut = orders.some((item: any) => item.statut !== undefined && item.statut !== null && item.statut !== '');

  const handleCheckboxChange = (idx: number) => (event: React.ChangeEvent<HTMLInputElement>) => {
    const newOrders = orders.map((order, i) => i === idx ? { ...order, executeNow: event.target.checked } : order);
    setOrders(newOrders);
  };

  // Handler pour modifier quantité, stoploss, takeprofit
  const handleOrderFieldChange = (idx: number, field: string) => (event: React.ChangeEvent<HTMLInputElement>) => {
    const value = event.target.value;
    setOrders(orders => orders.map((order, i) => i === idx ? { ...order, [field]: value } : order));
  };

  const handleExecute = async () => {
    if (!compteId) return;
    // Blocage si la règle n'est pas respectée
    if (alertMsg) return;
    setLoading(true);
    try {
      const res = await fetch('/api/trade/execute-orders', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: compteId, orders, idGpt }),
      });
      const data = await res.json();
      setOrders(data || []);
      setExecuted(true);
      if (onOrdersUpdate) onOrdersUpdate(data);
    } catch {
      // Optionnel : afficher une erreur
    } finally {
      setLoading(false);
    }
  };

  // Vérifie s'il y a au moins un ordre à exécuter
  const hasOrderToExecute = orders.some(order => order.executeNow !== false && Number(order.quantity ?? order.qty ?? 0) > 0);
  const hasSkippedDayTrade = orders.some((order: any) => order.statut === 'SKIPPED_DAYTRADE');

  // Calcul des totaux dynamiques (somme des montants = quantité × prix limite)
  // On exclut les lignes échouées (isFailed)
  const totalBuy = - orders
    .filter(order => order.executeNow && order.side === 'buy' && !(order.statut && FailedStatuses.includes(order.statut.toLowerCase())))
    .reduce((sum, order) => {
      const qty = Number(order.quantity ?? order.qty ?? 0);
      const price = Number(order.price_limit ?? order.priceLimit ?? 0);
      return sum + (qty * price);
    }, 0);
  const totalSell = orders
    .filter(order => order.executeNow && order.side === 'sell' && !(order.statut && FailedStatuses.includes(order.statut.toLowerCase())))
    .reduce((sum, order) => {
      const qty = Number(order.quantity ?? order.qty ?? 0);
      const price = Number(order.price_limit ?? order.priceLimit ?? 0);
      return sum + (qty * price);
    }, 0);
  const total = totalSell - totalBuy;

  return (
    <Paper sx={{ p: 2, mb: 2 }}>
      {/* Affichage des totaux au-dessus du tableau */}
      <Box sx={{ display: 'flex', gap: 4, mb: 1 }}>
        <Typography variant="subtitle2" color="primary">Total Buy : {totalBuy.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})} $</Typography>
        <Typography variant="subtitle2" color="secondary">Total Sell : {totalSell.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})} $</Typography>
        <Typography variant="subtitle2" color="text.primary">Total : {total.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})} $</Typography>
      </Box>
      <Typography variant="subtitle1" sx={{ mb: 1, fontWeight: 'bold' }}>Résultat AI :</Typography>
      <TableContainer>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Symbole</TableCell>
              <TableCell>Side</TableCell>
              <TableCell>Quantité</TableCell>
              {hasPriceLimit && <TableCell>Prix limite</TableCell>}
              <TableCell>Stop loss</TableCell>
              <TableCell>Take profit</TableCell>
              <TableCell>Exécuter</TableCell>
              <TableCell>Statut</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {orders.map((item: any, idx: number) => {
              const isOppositionFilled = item.oppositionOrder?.oppositionFilled === true;
              const isOppositionActived = item.oppositionOrder?.oppositionActived === true;
              const isFailed = item.statut && !FailedStatuses.includes(item.statut.toLowerCase());
              const isRed = isOppositionFilled || isOppositionActived || item.statut === 'FAILED_DAYTRADE' || item.statut === 'FAILED';
              return (
                <TableRow key={idx} style={isRed ? { background: '#ffcccc' } : (item.statut ? { background: '#e6ffe6' } : {})}>
                  <TableCell>{item.symbol}</TableCell>
                  <TableCell>{item.side}</TableCell>
                  <TableCell>
                    <input
                      type="number"
                      value={item.quantity ?? item.qty ?? ''}
                      min={0}
                      style={{ width: 70 }}
                      disabled={executed || isOppositionFilled}
                      onChange={handleOrderFieldChange(idx, item.quantity !== undefined ? 'quantity' : 'qty')}
                    />
                  </TableCell>
                  {hasPriceLimit && (
                    <TableCell>{item.price_limit ?? item.priceLimit ? (item.price_limit ?? item.priceLimit) + ' $' : '-'}</TableCell>
                  )}
                  <TableCell>
                    <input
                      type="number"
                      value={item.stop_loss ?? item.stopLoss ?? ''}
                      min={0}
                      style={{ width: 90 }}
                      disabled={executed || isOppositionFilled}
                      onChange={handleOrderFieldChange(idx, item.stop_loss !== undefined ? 'stop_loss' : 'stopLoss')}
                    />
                  </TableCell>
                  <TableCell>
                    <input
                      type="number"
                      value={item.take_profit ?? item.takeProfit ?? ''}
                      min={0}
                      style={{ width: 90 }}
                      disabled={executed || isOppositionFilled}
                      onChange={handleOrderFieldChange(idx, item.take_profit !== undefined ? 'take_profit' : 'takeProfit')}
                    />
                  </TableCell>
                  <TableCell>
                    <Checkbox
                      checked={item.executeNow !== false}
                      onChange={handleCheckboxChange(idx)}
                      color="primary"
                      disabled={executed || Number(item.quantity ?? item.qty ?? 0) === 0 || isOppositionFilled}
                    />
                  </TableCell>
                  <TableCell>{item.statut || '-'}</TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>
      <br />
      {hasSkippedDayTrade && (
              <Alert severity="warning" sx={{ mb: 2 }}>
                <div>- si l'ordre opposé en statut 'filled', l'exécution n'est pas possible</div>
                <div>- si l'ordre opposé en statut 'actived', l'exécution est possible (les ordres opposés en statut 'actived' seront annulés)</div>
              </Alert>
            )}
      {alertMsg && (
        <Alert severity="error" sx={{ mt: 2 }}>{alertMsg}</Alert>
      )}
      {!executed && hasOrderToExecute && (
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
          <Button
            variant="contained"
            color="primary"
            onClick={handleExecute}
            disabled={loading || !!alertMsg}
          >
            Exécuter
          </Button>
        </Box>
      )}
    </Paper>
  );
};

export default TradeAIJsonResult;
