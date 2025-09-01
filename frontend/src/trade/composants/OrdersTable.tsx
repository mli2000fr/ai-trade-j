import React from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import Typography from '@mui/material/Typography';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import Checkbox from '@mui/material/Checkbox';
import Button from '@mui/material/Button';
import CircularProgress from '@mui/material/CircularProgress';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';

interface OrdersTableProps {
  orders: any[];
  loading: boolean;
  filterSymbol: string;
  filterCancelable: boolean;
  onFilterSymbol: (symbol: string) => void;
  onFilterCancelable: (checked: boolean) => void;
  onUpdate: () => void;
  onCancel: (orderId: string) => void;
  cancellingOrderId: string | null;
  cancellableStatuses: string[];
  positions: any[];
  ordersSize: number;
  onOrdersSizeChange: (size: number) => void;
  cancelMessage?: string;
  disabled?: boolean;
}

const OrdersTable: React.FC<OrdersTableProps> = ({
  orders,
  loading,
  filterSymbol,
  filterCancelable,
  onFilterSymbol,
  onFilterCancelable,
  onUpdate,
  onCancel,
  cancellingOrderId,
  cancellableStatuses,
  positions,
  ordersSize,
  onOrdersSizeChange,
  cancelMessage,
  disabled
}) => {
  const hasCancellable = orders.some(order => order.id && cancellableStatuses.includes(order.status));
  return (
      <Card sx={{ mb: 3, backgroundColor: '#f5f5f5' }}>
            <CardContent>
    <Box sx={{ mb: 3 }}>
      <Typography variant="h6" sx={{ mb: 2 }}>Ordres récents</Typography>
      <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', mb: 2, flexWrap: 'wrap' }}>
        <FormControl size="small" sx={{ minWidth: 120 }}>
          <InputLabel id="filter-symbol-label">Symbole</InputLabel>
          <Select
            labelId="filter-symbol-label"
            value={filterSymbol}
            label="Symbole"
            onChange={e => onFilterSymbol(e.target.value)}
          >
            <MenuItem value="">Tous</MenuItem>
            {positions
              .map(pos => pos.symbol)
              .filter((symbol, idx, arr) => arr.indexOf(symbol) === idx)
              .map(symbol => (
                <MenuItem key={symbol} value={symbol}>{symbol}</MenuItem>
              ))}
          </Select>
        </FormControl>
        <FormControl size="small" sx={{ minWidth: 100 }}>
          <InputLabel id="orders-size-label">Nombre</InputLabel>
          <Select
            labelId="orders-size-label"
            value={ordersSize}
            label="Nombre"
            onChange={e => onOrdersSizeChange(Number(e.target.value))}
          >
            {[10, 20, 30, 40, 50].map(size => (
              <MenuItem key={size} value={size}>{size}</MenuItem>
            ))}
          </Select>
        </FormControl>
        <FormControl size="small" sx={{ flexDirection: 'row', alignItems: 'center' }}>
          <Checkbox
            checked={filterCancelable}
            onChange={e => onFilterCancelable(e.target.checked)}
            sx={{ p: 0, mr: 1 }}
          />
          <Box component="span">Annulables uniquement</Box>
        </FormControl>
        <Button onClick={onUpdate} disabled={loading} variant="contained" size="small">
          Update
        </Button>
      </Box>
      {orders.length === 0 && !loading ? (
        <Alert severity="info">Aucun ordre à afficher.</Alert>
      ) : loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}><CircularProgress /></Box>
      ) : (
        <TableContainer component={Paper}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Side</TableCell>
                <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Symbole</TableCell>
                <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Quantité</TableCell>
                <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Prix</TableCell>
                <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Stop-loss</TableCell>
                <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Take-profit</TableCell>
                <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Statut</TableCell>
                {hasCancellable && <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }} />}
              </TableRow>
            </TableHead>
            <TableBody>
              {orders.map((order, i) => {
                let bgColor = undefined;
                // On garde la couleur de fond mais on retire la couleur de texte
                if (order.statut === 'FAILED_DAYTRADE' || order.statut === 'FAILED') {
                  bgColor = 'rgba(244, 67, 54, 0.08)';
                } else if (order.side === 'buy') {
                  bgColor = 'rgba(76, 175, 80, 0.08)';
                } else if (order.side === 'sell') {
                  bgColor = 'rgba(244, 67, 54, 0.08)';
                }
                return (
                  <TableRow
                    key={i}
                    sx={{ backgroundColor: bgColor }}
                  >
                    <TableCell>{order.side}</TableCell>
                    <TableCell>{order.symbol}</TableCell>
                    <TableCell>{order.qty}</TableCell>
                    <TableCell>{order.filledAvgPrice !== undefined && order.filledAvgPrice !== null ? Number(order.filledAvgPrice).toLocaleString('fr-FR', {minimumFractionDigits: 2, maximumFractionDigits: 2}) + ' $' : (order.limit_price !== undefined && order.limit_price !== null ? Number(order.limit_price).toLocaleString('fr-FR', {minimumFractionDigits: 2, maximumFractionDigits: 2}) + ' $' : '-')}</TableCell>
                    <TableCell>{order.stopPrice !== undefined && order.stopPrice !== null ? Number(order.stopPrice).toLocaleString('fr-FR', {minimumFractionDigits: 2, maximumFractionDigits: 2}) + ' $' : '-'}</TableCell>
                    <TableCell>{order.limitPrice !== undefined && order.limitPrice !== null ? Number(order.limitPrice).toLocaleString('fr-FR', {minimumFractionDigits: 2, maximumFractionDigits: 2}) + ' $' : '-'}</TableCell>
                    <TableCell>{order.status}</TableCell>
                    {hasCancellable && (
                      <TableCell>
                        {order.id && cancellableStatuses.includes(order.status) && (
                          <Button
                            variant="outlined"
                            color="error"
                            size="small"
                            disabled={disabled || cancellingOrderId === order.id}
                            onClick={() => onCancel(order.id)}
                          >
                            {cancellingOrderId === order.id ? <CircularProgress size={16} /> : 'Annuler'}
                          </Button>
                        )}
                      </TableCell>
                    )}
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </TableContainer>
      )}
      {cancelMessage && (
        <Box sx={{ mt: 2 }}>
          <Alert severity={cancelMessage.toLowerCase().includes('erreur') ? 'error' : 'success'}>{cancelMessage}</Alert>
        </Box>
      )}
    </Box>

      </CardContent>
    </Card>
  );
};

export default OrdersTable;
