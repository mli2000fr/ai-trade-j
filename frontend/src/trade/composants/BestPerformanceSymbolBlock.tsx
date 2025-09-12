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
    rendementCheck: number;
    tradeCount: number;
    winRate: number;
    maxDrawdown: number;
    avgPnL: number;
    profitFactor: number;
    avgTradeBars: number;
    maxTradeGain: number;
    maxTradeLoss: number;
    scoreSwingTrade?: number;
    fltredOut?: boolean;
  };
  check: {
    rendement: number;
    tradeCount: number;
    winRate: number;
    maxDrawdown: number;
    avgPnL: number;
    profitFactor: number;
    avgTradeBars: number;
    maxTradeGain: number;
    maxTradeLoss: number;
    scoreSwingTrade?: number;
    fltredOut?: boolean;
  };
  paramsOptim: {
    initialCapital: number;
    riskPerTrade: number;
    stopLossPct: number;
    takeProfitPct: number;
    nbSimples: number;
  };
}

const BestPerformanceSymbolBlock: React.FC = () => {
  const [data, setData] = useState<BestInOutStrategy[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const [selected, setSelected] = useState<BestInOutStrategy | null>(null);
  const [checkedRows, setCheckedRows] = useState<{[key: number]: boolean}>({});
  const [limit, setLimit] = useState<number>(20);
  const [indices, setIndices] = useState<{ [symbol: string]: string }>({});
  const [sort, setSort] = useState<string>('rendement');
  const [showOnlyNonFiltered, setShowOnlyNonFiltered] = useState(false);

  useEffect(() => {
    setLoading(true);
    setError(null);
    let url = `/api/stra/strategies/best_performance_actions?limit=${limit}&sort=${sort}`;
    if (showOnlyNonFiltered) {
      url += `&filtered=true`;
    }
    fetch(url)
      .then(res => {
        if (!res.ok) throw new Error('Erreur API');
        return res.json();
      })
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [limit, sort, showOnlyNonFiltered]);

  useEffect(() => {
    if (!data || data.length === 0) return;
    const symbolsToFetch = data
      .map((row: BestInOutStrategy) => row.symbol)
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
  }, [data]);

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
              style={{ padding: '4px 8px', marginRight: 16 }}
            >
              <option value={20}>20</option>
              <option value={30}>30</option>
              <option value={50}>50</option>
              <option value={100}>100</option>
            </select>
            <label htmlFor="sort-select" style={{ marginRight: 8 }}>Trié par :</label>
            <select
              id="sort-select"
              value={sort}
              onChange={e => setSort(e.target.value)}
              style={{ padding: '4px 8px', marginRight: 16 }}
            >
              <option value="rendement">Rendement</option>
              <option value="score_swing_trade">Score Swing Trade</option>
            </select>
            <label style={{ marginLeft: 16, marginRight: 16 }}>
              <input
                type="checkbox"
                checked={showOnlyNonFiltered}
                onChange={e => setShowOnlyNonFiltered(e.target.checked)}
                style={{ marginRight: 8 }}
              />
              Stratégies non filtrées
            </label>
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
            <TableContainer component={Paper} sx={{ mb: 2 }}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}></TableCell> {/* Case à cocher */}
                    <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Symbole</TableCell>
                    <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Score Swing Trade</TableCell>
                    <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Filtrée</TableCell>
                    <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Indice</TableCell>
                    <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Stratégie IN</TableCell>
                    <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Stratégie OUT</TableCell>
                    <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Rendement</TableCell>
                    <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Rendement check</TableCell>
                    <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Trades</TableCell>
                    <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Durée moyenne trade</TableCell>
                    <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>WinRate</TableCell>
                    <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Drawdown</TableCell>
                    <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Profit Factor</TableCell>
                    <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Détails</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {data.map((row, i) => {
                    let bgColor = undefined;
                    if (indices[row.symbol] === 'BUY' && !row.result.fltredOut) bgColor = 'rgba(76, 175, 80, 0.5)';
                    if (indices[row.symbol] === 'SELL') bgColor = 'rgba(244, 67, 54, 0.05)';
                    return (
                      <TableRow key={i} sx={bgColor ? { backgroundColor: bgColor } : {}}>
                        <TableCell>
                          <input
                            type="checkbox"
                            checked={!!checkedRows[i]}
                            onChange={e => setCheckedRows({...checkedRows, [i]: e.target.checked})}
                          />
                        </TableCell>
                        <TableCell>{row.symbol}</TableCell>
                        <TableCell>{row.result.scoreSwingTrade !== undefined ? (row.result.scoreSwingTrade).toFixed(2) : '-'}</TableCell>
                        <TableCell>
                          {row.result.fltredOut ? (
                            <span style={{ color: 'red', fontWeight: 'bold' }}>Oui</span>
                          ) : (
                            <span>Non</span>
                          )}
                        </TableCell>
                        <TableCell>{indices[row.symbol] === 'pending' ? <CircularProgress size={16} /> : (indices[row.symbol] ?? '-')}</TableCell>
                        <TableCell>{row.entryName}</TableCell>
                        <TableCell>{row.exitName}</TableCell>
                        <TableCell>{(row.result.rendement * 100).toFixed(2)} %</TableCell>
                        <TableCell>{(row.result.rendementCheck * 100).toFixed(2)} %</TableCell>
                        <TableCell>{row.result.tradeCount}</TableCell>
                        <TableCell>{row.result.avgTradeBars !== undefined ? row.result.avgTradeBars.toFixed(2) : '-'}</TableCell>
                        <TableCell>{(row.result.winRate * 100).toFixed(2)} %</TableCell>
                        <TableCell>{(row.result.maxDrawdown * 100).toFixed(2)} %</TableCell>
                        <TableCell>{row.result.profitFactor.toFixed(2)}</TableCell>
                        <TableCell>
                          <Button size="small" variant="outlined" onClick={() => { setSelected(row); setOpen(true); }}>
                            Détails
                          </Button>
                        </TableCell>
                      </TableRow>
                    );
                  })}
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
              <Typography variant="subtitle1"><b>Score Swing Trade :</b> {selected.result.scoreSwingTrade !== undefined ? (selected.result.scoreSwingTrade).toFixed(2) : '-'}</Typography>
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
                <li>Capital initial : {selected.paramsOptim.initialCapital}</li>
                <li>Risk/trade : {selected.paramsOptim.riskPerTrade}</li>
                <li>Stop loss (%) : {selected.paramsOptim.stopLossPct}</li>
                <li>Take profit (%) : {selected.paramsOptim.takeProfitPct}</li>
                <li>Nb Simples : {selected.paramsOptim.nbSimples}</li>
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
