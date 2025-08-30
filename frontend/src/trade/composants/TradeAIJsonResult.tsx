import React from 'react';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';

interface TradeAIJsonResultProps {
  aiJsonResult: any;
}

const TradeAIJsonResult: React.FC<TradeAIJsonResultProps> = ({ aiJsonResult }) => {
  if (!aiJsonResult) return null;
  if (Array.isArray(aiJsonResult)) {
    const hasPriceLimit = aiJsonResult.some((item: any) => item.price_limit !== undefined && item.price_limit !== null && item.price_limit !== '' || item.priceLimit !== undefined && item.priceLimit !== null && item.priceLimit !== '');
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
              </TableRow>
            </TableHead>
            <TableBody>
              {aiJsonResult.map((item: any, idx: number) => (
                <TableRow key={idx}>
                  <TableCell>{item.symbol}</TableCell>
                  <TableCell>{item.action}</TableCell>
                  <TableCell>{item.quantity ?? item.qty ?? ''}</TableCell>
                  {hasPriceLimit && (
                    <TableCell>{item.price_limit ?? item.priceLimit ? (item.price_limit ?? item.priceLimit) + ' $' : '-'}</TableCell>
                  )}
                  <TableCell>{item.stop_loss ?? item.stopLoss ? (item.stop_loss ?? item.stopLoss) + ' $' : '-'}</TableCell>
                  <TableCell>{item.take_profit ?? item.takeProfit ? (item.take_profit ?? item.takeProfit) + ' $' : '-'}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>
    );
  }
  return (
    <Paper sx={{ p: 2, mb: 2 }}>
      <Typography variant="subtitle1" sx={{ mb: 1, fontWeight: 'bold' }}>Résultat AI :</Typography>
      <TableContainer>
        <Table size="small">
          <TableBody>
            {Object.entries(aiJsonResult).map(([key, value]) => (
              <TableRow key={key}>
                <TableCell sx={{ fontWeight: 'bold' }}>{key}</TableCell>
                <TableCell>{String(value)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Paper>
  );
};

export default TradeAIJsonResult;
