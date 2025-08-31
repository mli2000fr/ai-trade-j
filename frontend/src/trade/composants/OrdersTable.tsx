import React from 'react';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
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
  positions
}) => {
  const hasCancellable = orders.some(order => order.id && cancellableStatuses.includes(order.status));
  return (
    <Box sx={{ mb: 3 }}>
      <Box sx={{ fontWeight: 'bold', mb: 1 }}>Ordres récents&nbsp;:</Box>
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
        <FormControl size="small" sx={{ flexDirection: 'row', alignItems: 'center' }}>
          <Checkbox
            checked={filterCancelable}
            onChange={e => onFilterCancelable(e.target.checked)}
            sx={{ p: 0, mr: 1 }}
          />
          <Box component="span">Annulables uniquement</Box>
        </FormControl>
        <Button onClick={onUpdate} disabled={loading} variant="contained" size="small">
          {loading ? <CircularProgress size={18} /> : 'Update'}
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
                <TableCell>Side</TableCell>
                <TableCell>Symbole</TableCell>
                <TableCell>Quantité</TableCell>
                <TableCell>Prix</TableCell>
                <TableCell>Statut</TableCell>
                {hasCancellable && <TableCell />}
              </TableRow>
            </TableHead>
            <TableBody>
              {orders.map((order, i) => (
                <TableRow
                  key={i}
                  sx={{
                    backgroundColor:
                      order.statut === 'FAILED_DAYTRADE' || order.statut === 'FAILED'
                        ? '#ffcccc' // rouge clair si statut FAILED_DAYTRADE ou FAILED
                        : order.side === 'buy'
                        ? '#e3f2fd'
                        : order.side === 'sell'
                        ? '#ffebee'
                        : undefined
                  }}
                >
                  <TableCell>{order.side}</TableCell>
                  <TableCell>{order.symbol}</TableCell>
                  <TableCell>{order.qty}</TableCell>
                  <TableCell>{order.filledAvgPrice !== undefined && order.filledAvgPrice !== null ? Number(order.filledAvgPrice).toLocaleString('fr-FR', {minimumFractionDigits: 2, maximumFractionDigits: 2}) + ' $' : (order.limit_price !== undefined && order.limit_price !== null ? Number(order.limit_price).toLocaleString('fr-FR', {minimumFractionDigits: 2, maximumFractionDigits: 2}) + ' $' : '-')}</TableCell>
                  <TableCell>{order.status}</TableCell>
                  {hasCancellable && (
                    <TableCell>
                      {order.id && cancellableStatuses.includes(order.status) && (
                        <Button
                          variant="outlined"
                          color="error"
                          size="small"
                          disabled={cancellingOrderId === order.id}
                          onClick={() => onCancel(order.id)}
                        >
                          {cancellingOrderId === order.id ? <CircularProgress size={16} /> : 'Annuler'}
                        </Button>
                      )}
                    </TableCell>
                  )}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Box>
  );
};

export default OrdersTable;
