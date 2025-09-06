import React, { useEffect, useState } from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Grid from '@mui/material/Grid';
import CircularProgress from '@mui/material/CircularProgress';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';
import Button from '@mui/material/Button';
import Alert from '@mui/material/Alert';

interface PortfolioBlockProps {
  portfolio: any;
  lastUpdate: Date | null;
  loading: boolean;
  compteId?: string | null;
  resetSellErrorKey?: string | number;
}

const PortfolioBlock: React.FC<PortfolioBlockProps> = ({ portfolio, lastUpdate, loading, compteId, resetSellErrorKey }) => {
  const initialDeposit = portfolio && portfolio.initialDeposit !== undefined ? Number(portfolio.initialDeposit) : undefined;
  const equity = portfolio && portfolio.account?.equity !== undefined ? Number(portfolio.account.equity) : undefined;
  const plTotal = initialDeposit !== undefined && initialDeposit !== 0 && !isNaN(initialDeposit) && equity !== undefined && !isNaN(equity) ? equity - initialDeposit : undefined;
  const plPercent = initialDeposit !== undefined && initialDeposit !== 0 && !isNaN(initialDeposit) && equity !== undefined && !isNaN(equity) ? ((equity - initialDeposit) / initialDeposit) * 100 : undefined;

  // Ajout d'un cache local pour les indices
  const [indices, setIndices] = useState<{ [symbol: string]: string }>({});
  const [sellError, setSellError] = useState<string | null>(null);

  // Réinitialisation de sellError quand resetSellErrorKey change
  useEffect(() => {
    setSellError(null);
  }, [resetSellErrorKey]);

  useEffect(() => {
    if (!portfolio || !portfolio.positions) return;
    const symbolsToFetch = portfolio.positions
      .map((pos: any) => pos.symbol)
      .filter((symbol: string) => symbol && !(symbol in indices));
    if (symbolsToFetch.length === 0) return;
    symbolsToFetch.forEach((symbol: string) => {
      setIndices(prev => ({ ...prev, [symbol]: 'pending' }));
      fetch(`/api/stra/strategies/get_indice?symbol=${encodeURIComponent(symbol)}`)
        .then(res => res.json())
        .then(data => {
          setIndices(prev => ({ ...prev, [symbol]: data ?? '-' }));
        })
        .catch(() => {
          setIndices(prev => ({ ...prev, [symbol]: '-' }));
        });
    });
  }, [portfolio]);

  const handleSell = async (pos: any) => {
    setSellError(null);
    try {
      const response = await fetch(`/api/trade/trade`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'sell',
          cancelOpposite: true,
          forceDayTrade: false,
          id: compteId,
          quantity: pos.qty,
          stopLoss: null,
          symbol: pos.symbol,
          takeProfit: null
        })
      });
      if (!response.ok) {
        const errorMsg = await response.text();
        setSellError(errorMsg);
      } else {
        setSellError(null);
      }
    } catch (e: any) {
      setSellError(e.message || 'Erreur lors de la vente.');
    }
  };

  return (
    <Card sx={{ mb: 3, backgroundColor: '#f5f5f5' }}>
      <CardContent>
        <Typography variant="h5" gutterBottom>Mon portefeuille</Typography>
        {lastUpdate && (
          <Typography variant="caption" color="text.secondary" sx={{ mb: 2, display: 'block' }}>
            Actualisé à {lastUpdate.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
          </Typography>
        )}
        {loading && <CircularProgress sx={{ my: 2 }} />}
        {!loading && portfolio && (
          <>
            {portfolio.account && (
              <Grid container spacing={2} sx={{ mb: 2 }}>
                <Grid>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="subtitle2" color="text.secondary">Valeur totale</Typography>
                      <Typography variant="h6">{Number(portfolio.account.equity).toLocaleString('fr-FR', {minimumFractionDigits: 2, maximumFractionDigits: 2})} $</Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="subtitle2" color="text.secondary">Buying Power</Typography>
                      <Typography variant="h6">{Number(portfolio.account.buying_power).toLocaleString('fr-FR', {minimumFractionDigits: 2, maximumFractionDigits: 2})} $</Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="subtitle2" color="text.secondary">Cash</Typography>
                      <Typography variant="h6">{Number(portfolio.account.cash).toLocaleString('fr-FR', {minimumFractionDigits: 2, maximumFractionDigits: 2})} $</Typography>
                    </CardContent>
                  </Card>
                </Grid>
                {portfolio.account.portfolio_value && (
                  <Grid>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="subtitle2" color="text.secondary">Portfolio Value</Typography>
                        <Typography variant="h6">{Number(portfolio.account.portfolio_value).toLocaleString('fr-FR', {minimumFractionDigits: 2, maximumFractionDigits: 2})} $</Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                )}
                {portfolio.account.status && (
                  <Grid>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="subtitle2" color="text.secondary">Status</Typography>
                        <Typography variant="h6">{portfolio.account.status}</Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                )}
                <Grid>
                  <Card variant="outlined">
                    <CardContent sx={plTotal !== undefined ? { backgroundColor: plTotal > 0 ? 'success.light' : plTotal < 0 ? 'error.light' : undefined } : {}}>
                      <Typography variant="subtitle2" color="text.secondary">P/L total</Typography>
                      <Typography variant="h6">
                        {plTotal !== undefined
                          ? plTotal.toLocaleString('fr-FR', {minimumFractionDigits: 2, maximumFractionDigits: 2}) + ' $'
                          : '-'}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid>
                  <Card variant="outlined">
                    <CardContent sx={plPercent !== undefined ? { backgroundColor: plPercent > 0 ? 'success.light' : plPercent < 0 ? 'error.light' : undefined } : {}}>
                      <Typography variant="subtitle2" color="text.secondary">P/L (%)</Typography>
                      <Typography variant="h6">
                        {plPercent !== undefined
                          ? plPercent.toLocaleString('fr-FR', {minimumFractionDigits: 2, maximumFractionDigits: 2}) + ' %'
                          : '-'}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            )}
            <Typography variant="subtitle1" sx={{ mt: 2, mb: 1 }}><b>Positions&nbsp;:</b></Typography>
            {portfolio.positions.length === 0 ? (
              <Typography>Aucune position en cours.</Typography>
            ) : (
              <TableContainer component={Paper} sx={{ mb: 2 }}>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Symbole</TableCell>
                      <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Indice</TableCell>
                      <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Prix d'achat</TableCell>
                      <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Prix actuel</TableCell>
                      <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Quantité</TableCell>
                      <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Total</TableCell>
                      <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>P & L pc</TableCell>
                      <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>P & L</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {portfolio.positions.map((pos: any, i: number) => {
                      const isNegative = pos.unrealized_pl < 0;
                      const cellStyle = isNegative ? { color: 'error.main' } : {};
                      return (
                        <TableRow
                          key={i}
                          sx={
                            pos.unrealized_pl > 0
                              ? { backgroundColor: 'rgba(76, 175, 80, 0.08)' }
                              : pos.unrealized_pl < 0
                              ? { backgroundColor: 'rgba(244, 67, 54, 0.08)' }
                              : {}
                          }
                        >
                          <TableCell sx={cellStyle}>{pos.symbol}</TableCell>
                          <TableCell sx={
                            indices[pos.symbol] === 'SELL'
                              ? { color: 'error.main', fontWeight: 'bold' }
                              : cellStyle
                          }>
                            {indices[pos.symbol] === 'pending' ? <CircularProgress size={16} /> : (
                              indices[pos.symbol] === 'SELL'
                                ? <Button variant="contained" color="error" size="small" onClick={() => handleSell(pos)} disabled={pos.qty <= 0}>SELL</Button>
                                : (indices[pos.symbol] ?? '-')
                            )}
                          </TableCell>
                          <TableCell sx={cellStyle}>{pos.avg_entry_price !== undefined && pos.avg_entry_price !== null ? Number(pos.avg_entry_price).toFixed(2) + ' $' : '-'}</TableCell>
                          <TableCell sx={cellStyle}>{pos.current_price !== undefined && pos.current_price !== null ? Number(pos.current_price).toFixed(2) + ' $' : '-'}</TableCell>
                          <TableCell sx={cellStyle}>{pos.qty}</TableCell>
                          <TableCell sx={cellStyle}>{pos.current_price !== undefined && pos.current_price !== null ? (Number(pos.qty) * Number(pos.current_price)).toFixed(2) + ' $' : '-'}</TableCell>
                          <TableCell sx={cellStyle}>{pos.unrealized_plpc !== undefined && pos.unrealized_plpc !== null ? (Number(pos.unrealized_plpc) * 100).toFixed(3) + ' %' : '-'}</TableCell>
                          <TableCell sx={cellStyle}>{pos.unrealized_pl !== undefined && pos.unrealized_pl !== null ? Number(pos.unrealized_pl).toFixed(2) + ' $' : '-'}</TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </TableContainer>
            )}
            {sellError && <Alert severity="error" sx={{ mb: 2 }}>{sellError}</Alert>}
          </>
        )}
      </CardContent>
    </Card>
  );
};

export default PortfolioBlock;
