import React, { useEffect, useState, useRef } from 'react';
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
import Accordion from '@mui/material/Accordion';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ReactApexChart from 'react-apexcharts';
import type { ApexOptions } from 'apexcharts';

interface BestInOutStrategy {
  symbol: string;
  entryName: string;
  entryParams?: any;
  exitName: string;
  exitParams?: any;
  rendementSum?: any;
  rendementDiff?: any;
  rendementScore?: any;
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
  const [sort, setSort] = useState<string>('rendement_score');
  const [showOnlyNonFiltered, setShowOnlyNonFiltered] = useState(false);
  const [bougies, setBougies] = useState<any[]>([]);
  const [bougiesLoading, setBougiesLoading] = useState(false);
  const [bougiesError, setBougiesError] = useState<string | null>(null);
  const chartContainerRef = useRef<HTMLDivElement>(null);

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

  // Fonction utilitaire pour afficher un objet sous forme de tableau
  const renderObjectTable = (obj: any) => (
    <Table size="small" sx={{ mb: 2, backgroundColor: '#f9f9f9' }}>
      <TableBody>
        {Object.entries(obj).map(([key, value]) => (
          <TableRow key={key}>
            <TableCell sx={{ fontWeight: 'bold', width: '40%' }}>{key}</TableCell>
            <TableCell>
              {typeof value === 'number'
                ? (Math.abs(value) > 1
                    ? value.toFixed(2)
                    : (value * 100).toFixed(2) + (key.toLowerCase().includes('pct') || key.toLowerCase().includes('rate') || key.toLowerCase().includes('drawdown') || key.toLowerCase().includes('rendement') ? ' %' : ''))
                : typeof value === 'boolean'
                  ? value ? 'Oui' : 'Non'
                  : String(value)}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );

  useEffect(() => {
    if (open && selected?.symbol) {
      setBougiesLoading(true);
      setBougiesError(null);
      fetch(`/api/stra/getBougiesBySymbol?symbol=${encodeURIComponent(selected.symbol)}&historique=250`)
        .then(res => {
          if (!res.ok) throw new Error('Erreur API bougies');
          return res.json();
        })
        .then(data => {
          setBougies(data);
        })
        .catch(e => setBougiesError(e.message))
        .finally(() => setBougiesLoading(false));
    } else {
      setBougies([]);
      setBougiesError(null);
      setBougiesLoading(false);
    }
  }, [open, selected]);

  // ErrorBoundary pour capturer les erreurs de rendu
  class ErrorBoundary extends React.Component<{children: React.ReactNode}, {hasError: boolean, error: any}> {
    constructor(props: any) {
      super(props);
      this.state = { hasError: false, error: null };
    }
    static getDerivedStateFromError(error: any) {
      return { hasError: true, error };
    }
    componentDidCatch(error: any, errorInfo: any) {
      // log possible
    }
    render() {
      if (this.state.hasError) {
        return <div style={{color: 'red'}}>Erreur d'affichage du graphique : {String(this.state.error)}</div>;
      }
      return this.props.children;
    }
  }

  // Ajout du composant BougiesChart utilisant ApexCharts
  const BougiesChart: React.FC<{ bougies: any[] }> = ({ bougies }) => {
    if (!bougies || bougies.length === 0) {
      return <Typography variant="body2">Aucune donnée de bougie disponible.</Typography>;
    }
    // Mapping adapté au format {date, open, high, low, close, ...}
    const mappedBougies = bougies.map(b => {
      const open = b.open;
      const high = b.high;
      const low = b.low;
      const closeValue = b.close;
      // Conversion de la date "YYYY-MM-DD" en timestamp
      const time = b.date ? new Date(b.date).getTime() : undefined;
      return { open, high, low, closeValue, time };
    });
    const isFormatOK = mappedBougies.every(b => b.open !== undefined && b.high !== undefined && b.low !== undefined && b.closeValue !== undefined && b.time !== undefined);
    if (!isFormatOK) {
      return <Alert severity="error">Format des données de bougie non reconnu. Exemple attendu : {'{date, open, high, low, close}'}</Alert>;
    }
    const series = [{
      data: mappedBougies.map(b => ({
        x: b.time,
        y: [Number(b.open), Number(b.high), Number(b.low), Number(b.closeValue)]
      }))
    }];
    const options: ApexOptions = {
      chart: {
        type: 'candlestick',
        height: 300,
        toolbar: { show: false }
      },
      xaxis: {
        type: 'datetime',
        labels: { datetimeUTC: false }
      },
      yaxis: {
        tooltip: { enabled: true }
      },
      title: {
        text: 'Graphique chandelier',
        align: 'left'
      }
    };
    return <ReactApexChart options={options} series={series} type="candlestick" height={300} />;
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
              <option value="rendement_score">Score Rendement</option>
              <option value="rendement">Rendement</option>
              <option value="rendement_sum">Rendement Sum</option>
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
                    <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Filtrée</TableCell>
                    <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Indice</TableCell>
                    <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Stratégie IN</TableCell>
                    <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Stratégie OUT</TableCell>
                    <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Rendement</TableCell>
                    <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Rendement check</TableCell>
                    <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Rendement Sum</TableCell>
                    <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Rendement Diff</TableCell>
                    <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Score Rendement</TableCell>
                    <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Score Swing Trade</TableCell>
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
                        <TableCell><input type="checkbox" checked={!!checkedRows[i]} onChange={e => setCheckedRows({...checkedRows, [i]: e.target.checked})} /></TableCell>
                        <TableCell>{row.symbol}</TableCell>
                        <TableCell>{row.result.fltredOut ? <span style={{ color: 'red', fontWeight: 'bold' }}>Oui</span> : <span>Non</span>}</TableCell>
                        <TableCell>{indices[row.symbol] === 'pending' ? <CircularProgress size={16} /> : (indices[row.symbol] ?? '-')}</TableCell>
                        <TableCell>{row.entryName}</TableCell>
                        <TableCell>{row.exitName}</TableCell>
                        <TableCell>{(row.result.rendement * 100).toFixed(2)} %</TableCell>
                        <TableCell>{(row.check.rendement * 100).toFixed(2)} %</TableCell>
                        <TableCell>{(row.rendementSum * 100).toFixed(2)} %</TableCell>
                        <TableCell>{(row.rendementDiff * 100).toFixed(2)} %</TableCell>
                        <TableCell>{(row.rendementScore * 100).toFixed(2)}</TableCell>
                        <TableCell>{row.result.scoreSwingTrade !== undefined ? (row.result.scoreSwingTrade).toFixed(2) : '-'}</TableCell>
                        <TableCell>{row.result.tradeCount}</TableCell>
                        <TableCell>{row.result.avgTradeBars !== undefined ? row.result.avgTradeBars.toFixed(2) : '-'}</TableCell>
                        <TableCell>{(row.result.winRate * 100).toFixed(2)} %</TableCell>
                        <TableCell>{(row.result.maxDrawdown * 100).toFixed(2)} %</TableCell>
                        <TableCell>{row.result.profitFactor.toFixed(2)}</TableCell>
                        <TableCell><Button size="small" variant="outlined" onClick={() => { setSelected(row); setOpen(true); }}>Détails</Button></TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </CardContent>
      </Card>
      <Dialog open={open} onClose={() => setOpen(false)} maxWidth="lg" fullWidth PaperProps={{ sx: { maxWidth: '60vw' } }}>
        <DialogTitle>Détails de la performance</DialogTitle>
        <DialogContent>
          {selected && (
            <div>
              <Typography variant="h6" sx={{ mb: 2, color: '#1976d2' }}>Symbole : {selected.symbol}</Typography>
              <Accordion defaultExpanded>
                              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                                <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>Graphique des 250 dernières bougies</Typography>
                              </AccordionSummary>
                              <AccordionDetails>
                                {bougiesLoading ? (
                                  <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 300 }}>
                                    <CircularProgress />
                                  </div>
                                ) : bougiesError ? (
                                  <Alert severity="error">{bougiesError}</Alert>
                                ) : bougies && bougies.length > 0 ? (
                                  <BougiesChart bougies={bougies} />
                                ) : (
                                  <Typography variant="body2">Aucune donnée de bougie disponible.</Typography>
                                )}
                              </AccordionDetails>
                            </Accordion>
              <Accordion defaultExpanded>
                              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                                <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>Résultats & Vérification</Typography>
                              </AccordionSummary>
                              <AccordionDetails>
                                <Table size="small" sx={{ mt: 2, backgroundColor: '#f1f8e9' }}>
                                  <TableBody>
                                    <TableRow>
                                      <TableCell sx={{ fontWeight: 'bold' }}>Rendement Sum</TableCell>
                                      <TableCell>{(selected.rendementSum * 100).toFixed(2)} %</TableCell>
                                    </TableRow>
                                    <TableRow>
                                      <TableCell sx={{ fontWeight: 'bold' }}>Rendement Diff</TableCell>
                                      <TableCell>{(selected.rendementDiff * 100).toFixed(2)} %</TableCell>
                                    </TableRow>
                                    <TableRow>
                                      <TableCell sx={{ fontWeight: 'bold' }}>Rendement Score</TableCell>
                                      <TableCell>{(selected.rendementScore * 100).toFixed(2)}</TableCell>
                                    </TableRow>
                                  </TableBody>
                                </Table>
                                <br/>
                                <Table size="small" sx={{ mb: 2, backgroundColor: '#f9f9f9' }}>
                                  <TableHead>
                                    <TableRow>
                                      <TableCell sx={{ fontWeight: 'bold', width: '40%' }}>Métrique</TableCell>
                                      <TableCell sx={{ fontWeight: 'bold' }}>Résultat</TableCell>
                                      <TableCell sx={{ fontWeight: 'bold' }}>Vérification</TableCell>
                                    </TableRow>
                                  </TableHead>
                                  <TableBody>
                                    {Array.from(new Set([...Object.keys(selected.result || {}), ...Object.keys(selected.check || {})])).map((key) => {
                                      const resultObj = selected.result as Record<string, any>;
                                      const checkObj = selected.check as Record<string, any>;
                                      return (
                                        <TableRow key={key}>
                                          <TableCell sx={{ fontWeight: 'bold' }}>{key}</TableCell>
                                          <TableCell>
                                            {typeof resultObj?.[key] === 'number'
                                              ? (Math.abs(resultObj[key]) > 1
                                                  ? resultObj[key].toFixed(2)
                                                  : (resultObj[key] * 100).toFixed(2) + (key.toLowerCase().includes('pct') || key.toLowerCase().includes('rate') || key.toLowerCase().includes('drawdown') || key.toLowerCase().includes('rendement') ? ' %' : ''))
                                              : typeof resultObj?.[key] === 'boolean'
                                                ? resultObj[key] ? 'Oui' : 'Non'
                                                : resultObj?.[key] ?? '-'}
                                          </TableCell>
                                          <TableCell>
                                            {typeof checkObj?.[key] === 'number'
                                              ? (Math.abs(checkObj[key]) > 1
                                                  ? checkObj[key].toFixed(2)
                                                  : (checkObj[key] * 100).toFixed(2) + (key.toLowerCase().includes('pct') || key.toLowerCase().includes('rate') || key.toLowerCase().includes('drawdown') || key.toLowerCase().includes('rendement') ? ' %' : ''))
                                              : typeof checkObj?.[key] === 'boolean'
                                                ? checkObj[key] ? 'Oui' : 'Non'
                                                : checkObj?.[key] ?? '-'}
                                          </TableCell>
                                        </TableRow>
                                      );
                                    })}
                                  </TableBody>
                                </Table>
                              </AccordionDetails>
                            </Accordion>
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>Stratégie d'entrée</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Typography variant="body2" sx={{ mb: 1 }}>Nom : <b>{selected.entryName}</b></Typography>
                  {selected.entryParams && renderObjectTable(selected.entryParams)}
                </AccordionDetails>
              </Accordion>
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>Stratégie de sortie</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Typography variant="body2" sx={{ mb: 1 }}>Nom : <b>{selected.exitName}</b></Typography>
                  {selected.exitParams && renderObjectTable(selected.exitParams)}
                </AccordionDetails>
              </Accordion>

              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>Paramètres d'optimisation</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  {selected.paramsOptim && renderObjectTable(selected.paramsOptim)}
                </AccordionDetails>
              </Accordion>

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
