import React, { useEffect, useState } from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';
import CircularProgress from '@mui/material/CircularProgress';
import Alert from '@mui/material/Alert';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';

interface BestInOutStrategy {
  symbol: string;
  entryName: string;
  entryParams?: any;
  exitName: string;
  exitParams?: any;
  result: {
    rendement: number;
    tradeCount: number;
    winRate: number;
    maxDrawdown: number;
    avgPnL: number;
    profitFactor: number;
    avgTradeBars: number;
    maxTradeGain: number;
    maxTradeLoss: number;
  };
  initialCapital: number;
  riskPerTrade: number;
  stopLossPct: number;
  takeProfitPct: number;
}

const BestPerformanceSymbolBlock: React.FC = () => {
  const [data, setData] = useState<BestInOutStrategy[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const [selected, setSelected] = useState<BestInOutStrategy | null>(null);
  const [checkedRows, setCheckedRows] = useState<{[key: number]: boolean}>({});
  const [limit, setLimit] = useState<number>(20);

  useEffect(() => {
    setLoading(true);
    setError(null);
    fetch(`/api/stra/strategies/best_performance_actions?limit=${limit}`)
      .then(res => {
        if (!res.ok) throw new Error('Erreur API');
        return res.json();
      })
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [limit]);

  const handleCopy = () => {
    const selectedSymbols = data
      .map((row, idx) => checkedRows[idx] ? row.symbol : null)
      .filter(Boolean)
      .join(',');
    if (selectedSymbols) {
      navigator.clipboard.writeText(selectedSymbols);
    }
  };

  return (
    <>
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Best Performance-Symbol
          </Typography>
          {/* Liste déroulante au-dessus du tableau */}
          <div style={{ marginBottom: 16 }}>
            <label htmlFor="limit-select" style={{ marginRight: 8 }}>Afficher :</label>
            <select
              id="limit-select"
              value={limit}
              onChange={e => setLimit(Number(e.target.value))}
              style={{ padding: '4px 8px' }}
            >
              <option value={20}>20</option>
              <option value={30}>30</option>
              <option value={50}>50</option>
              <option value={100}>100</option>
            </select>
            {/* Bouton copier */}
            <Button
              variant="contained"
              color="primary"
              size="small"
              startIcon={<ContentCopyIcon />}
              style={{ marginLeft: 16 }}
              onClick={handleCopy}
              disabled={Object.values(checkedRows).every(v => !v)}
            >
              Copier
            </Button>
          </div>
          {loading ? (
            <CircularProgress />
          ) : error ? (
            <Alert severity="error">{error}</Alert>
          ) : (
            <TableContainer component={Paper}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell></TableCell> {/* Case à cocher */}
                    <TableCell>Symbole</TableCell>
                    <TableCell>Entrée</TableCell>
                    <TableCell>Sortie</TableCell>
                    <TableCell>Rendement</TableCell>
                    <TableCell>Trades</TableCell>
                    <TableCell>WinRate</TableCell>
                    <TableCell>Drawdown</TableCell>
                    <TableCell>Profit Factor</TableCell>
                    <TableCell>Détails</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {data.map((row, idx) => (
                    <TableRow key={idx}>
                      <TableCell>
                        <input
                          type="checkbox"
                          checked={!!checkedRows[idx]}
                          onChange={e => setCheckedRows({...checkedRows, [idx]: e.target.checked})}
                        />
                      </TableCell>
                      <TableCell>{row.symbol}</TableCell>
                      <TableCell>{row.entryName}</TableCell>
                      <TableCell>{row.exitName}</TableCell>
                      <TableCell>{(row.result.rendement * 100).toFixed(2)}%</TableCell>
                      <TableCell>{row.result.tradeCount}</TableCell>
                      <TableCell>{(row.result.winRate * 100).toFixed(1)}%</TableCell>
                      <TableCell>{(row.result.maxDrawdown * 100).toFixed(2)}%</TableCell>
                      <TableCell>{row.result.profitFactor.toFixed(2)}</TableCell>
                      <TableCell>
                        <Button size="small" variant="outlined" onClick={() => { setSelected(row); setOpen(true); }}>
                          Détails
                        </Button>
                      </TableCell>
                    </TableRow>
                    ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </CardContent>
      </Card>
      <Dialog open={open} onClose={() => setOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Détails de la performance</DialogTitle>
        <DialogContent>
          {selected && (
            <div>
              <Typography variant="subtitle1"><b>Symbole :</b> {selected.symbol}</Typography>
              <Typography variant="subtitle1"><b>Stratégie Entrée :</b> {selected.entryName}</Typography>
              <Typography variant="body2">Paramètres entrée : <pre>{JSON.stringify(selected.entryParams, null, 2)}</pre></Typography>
              <Typography variant="subtitle1"><b>Stratégie Sortie :</b> {selected.exitName}</Typography>
              <Typography variant="body2">Paramètres sortie : <pre>{JSON.stringify(selected.exitParams, null, 2)}</pre></Typography>
                <ul>
                <li>Drawdown : {(selected.result.maxDrawdown * 100).toFixed(2)}%</li>
                <li>Profit Factor : {selected.result.profitFactor.toFixed(2)}</li>
                <li>Profit moyen : {selected.result.avgPnL.toFixed(2)}</li>
                <li>Durée moyenne trade : {selected.result.avgTradeBars.toFixed(2)}</li>
                <li>Max gain trade : {selected.result.maxTradeGain.toFixed(2)}</li>
                <li>Max perte trade : {selected.result.maxTradeLoss.toFixed(2)}</li>
              </ul>
              <Typography variant="subtitle1"><b>Gestion du risque :</b></Typography>
              <ul>
                <li>Capital initial : {selected.initialCapital}</li>
                <li>Risk/trade : {selected.riskPerTrade}</li>
                <li>Stop loss (%) : {selected.stopLossPct}</li>
                <li>Take profit (%) : {selected.takeProfitPct}</li>
              </ul>
            </div>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpen(false)}>Fermer</Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default BestPerformanceSymbolBlock;
