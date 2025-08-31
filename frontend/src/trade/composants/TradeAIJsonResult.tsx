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

interface TradeAIJsonResultProps {
  aiJsonResult: any;
  compteId?: string | number;
  onOrdersUpdate?: (orders: any[]) => void;
}

const TradeAIJsonResult: React.FC<TradeAIJsonResultProps> = ({ aiJsonResult, compteId, onOrdersUpdate }) => {
  const [orders, setOrders] = useState<any[]>(Array.isArray(aiJsonResult) ? aiJsonResult.map((o: any) => ({ ...o })) : []);
  const [loading, setLoading] = useState(false);
  const [executed, setExecuted] = useState(false);

  // Met à jour l'état local si aiJsonResult change
  React.useEffect(() => {
    if (Array.isArray(aiJsonResult)) {
      setOrders(aiJsonResult.map((o: any) => ({ ...o })));
    }
  }, [aiJsonResult]);

  if (!Array.isArray(orders) || orders.length === 0) return null;
  const hasPriceLimit = orders.some((item: any) => item.price_limit !== undefined && item.price_limit !== null && item.price_limit !== '' || item.priceLimit !== undefined && item.priceLimit !== null && item.priceLimit !== '');
  const hasStatut = orders.some((item: any) => item.statut !== undefined && item.statut !== null && item.statut !== '');

  const handleCheckboxChange = (idx: number) => (event: React.ChangeEvent<HTMLInputElement>) => {
    const newOrders = orders.map((order, i) => i === idx ? { ...order, executeNow: event.target.checked } : order);
    setOrders(newOrders);
  };

  const handleExecute = async () => {
    if (!compteId) return;
    setLoading(true);
    try {
      const res = await fetch('/api/trade/execute-orders', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: compteId, orders }),
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

  return (
    <Paper sx={{ p: 2, mb: 2 }}>
      <Typography variant="subtitle1" sx={{ mb: 1, fontWeight: 'bold' }}>Résultat AI :</Typography>
      <TableContainer>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Symbole</TableCell>
              <TableCell>Action</TableCell>
              <TableCell>Quantité</TableCell>
              {hasPriceLimit && <TableCell>Prix limite</TableCell>}
              <TableCell>Stop loss</TableCell>
              <TableCell>Take profit</TableCell>
              <TableCell>Exécuter maintenant</TableCell>
              {hasStatut && <TableCell>Statut</TableCell>}
            </TableRow>
          </TableHead>
          <TableBody>
            {orders.map((item: any, idx: number) => (
              <TableRow key={idx} style={item.statut ? { background: '#e6ffe6' } : {}}>
                <TableCell>{item.symbol}</TableCell>
                <TableCell>{item.action}</TableCell>
                <TableCell>{item.quantity ?? item.qty ?? ''}</TableCell>
                {hasPriceLimit && (
                  <TableCell>{item.price_limit ?? item.priceLimit ? (item.price_limit ?? item.priceLimit) + ' $' : '-'}</TableCell>
                )}
                <TableCell>{item.stop_loss ?? item.stopLoss ? (item.stop_loss ?? item.stopLoss) + ' $' : '-'}</TableCell>
                <TableCell>{item.take_profit ?? item.takeProfit ? (item.take_profit ?? item.takeProfit) + ' $' : '-'}</TableCell>
                <TableCell>
                  <Checkbox
                    checked={item.executeNow !== false}
                    onChange={handleCheckboxChange(idx)}
                    color="primary"
                    disabled={executed || Number(item.quantity ?? item.qty ?? 0) === 0}
                  />
                </TableCell>
                {hasStatut && <TableCell>{item.statut || '-'}</TableCell>}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
      {!executed && (
        <Button
          variant="contained"
          color="primary"
          sx={{ mt: 2 }}
          onClick={handleExecute}
          disabled={loading}
        >
          Exécuter
        </Button>
      )}
    </Paper>
  );
};

export default TradeAIJsonResult;
