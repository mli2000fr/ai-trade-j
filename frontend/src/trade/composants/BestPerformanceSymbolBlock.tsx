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
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import Accordion from '@mui/material/Accordion';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ReactApexChart from 'react-apexcharts';
import type { ApexOptions } from 'apexcharts';
import BestPerformanceDialog from './BestPerformanceDialog';

interface PreditLstm {
    lastClose: number;
    predictedClose: number;
    lastDate?: string | null;
    signal?: SignalInfo;
    position?: string | null;
}

interface SignalInfo {
    symbol: string;
    type: string;
    date?: string | null;
    dateStr?: string;
}

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

interface BestCombinationResult {
  symbol: string | null;
  inStrategyNames: string[];
  outStrategyNames: string[];
  inParams: Record<string, any>;
  outParams: Record<string, any>;
  result: {
    rendement: number;
    maxDrawdown: number;
    tradeCount: number;
    winRate: number;
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
    maxDrawdown: number;
    tradeCount: number;
    winRate: number;
    avgPnL: number;
    profitFactor: number;
    avgTradeBars: number;
    maxTradeGain: number;
    maxTradeLoss: number;
    scoreSwingTrade?: number;
    fltredOut?: boolean;
  };
  rendementSum: number;
  rendementDiff: number;
  rendementScore: number;
  contextOptim: {
    initialCapital: number;
    riskPerTrade: number;
    stopLossPct: number;
    takeProfitPct: number;
    nbSimples: number;
  };
}

interface MixResultat {
  single: BestInOutStrategy;
  mix: BestCombinationResult;
}

const BestPerformanceSymbolBlock: React.FC = () => {
  const [data, setData] = useState<MixResultat[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const [selected, setSelected] = useState<MixResultat | null>(null);
  const [checkedRows, setCheckedRows] = useState<{[key: number]: boolean}>({});
  const [limit, setLimit] = useState<number>(20);
  const [indices, setIndices] = useState<{ [symbol: string]: SignalInfo | string }>({});
  const [indicesMix, setIndicesMix] = useState<{ [symbol: string]: SignalInfo | string }>({});
  const [sort, setSort] = useState<string>('single:rendement_score');
  const [showOnlyNonFiltered, setShowOnlyNonFiltered] = useState(true);
  const [bougies, setBougies] = useState<any[]>([]);
  const [bougiesLoading, setBougiesLoading] = useState(false);
  const [bougiesError, setBougiesError] = useState<string | null>(null);
  const [lstmResults, setLstmResults] = useState<{ [symbol: string]: PreditLstm | string}>({});
  const chartContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    const type = sort.split(':')[0];
    const tri = sort.split(':')[1];

    let url = `/api/result/global?limit=${limit}&type=${type}&sort=${tri}`;
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
      .map((row: MixResultat) => row.single.symbol)
      .filter((symbol: string) => symbol && !(symbol in indices));
    if (symbolsToFetch.length === 0) return;
    symbolsToFetch.forEach((symbol: string) => {
      setIndices(prev => ({ ...prev, [symbol]: 'pending' }));
      fetch(`/api/stra/strategies/get_indice?symbol=${encodeURIComponent(symbol)}`)
        .then(res => res.json())
        .then((data: SignalInfo) => {
          setIndices(prev => ({ ...prev, [symbol]: data ?? '-' }));
        })
        .catch(() => {
          setIndices(prev => ({ ...prev, [symbol]: '-' }));
        });
      setIndicesMix(prev => ({ ...prev, [symbol]: 'pending' }));
      fetch(`/api/best-combination/get_indice?symbol=${encodeURIComponent(symbol)}`)
        .then(res => res.json())
        .then((data: SignalInfo) => {
          setIndicesMix(prev => ({ ...prev, [symbol]: data ?? '-' }));
        })
        .catch(() => {
          setIndicesMix(prev => ({ ...prev, [symbol]: '-' }));
        });
      // Ajout fetch LSDM
      setLstmResults(prev => ({ ...prev, [symbol]: 'pending' }));
      fetch(`/api/lstm/predict?symbol=${encodeURIComponent(symbol)}`)
        .then(res => res.json())
        .then((data: any) => {
          setLstmResults(prev => ({ ...prev, [symbol]: data ?? '-' }));
        })
        .catch(() => {
          setLstmResults(prev => ({ ...prev, [symbol]: '-' }));
        });
    });
  }, [data]);

  const handleCopy = () => {
    const selectedSymbols = data
      .map((row, idx) => checkedRows[idx] ? row.single.symbol : null)
      .filter(Boolean)
      .join(',');
    if (selectedSymbols) {
      navigator.clipboard.writeText(selectedSymbols);
    }
  };


  useEffect(() => {
    if (open && selected?.single.symbol) {
      setBougiesLoading(true);
      setBougiesError(null);
      fetch(`/api/stra/getBougiesBySymbol?symbol=${encodeURIComponent(selected.single.symbol)}&historique=250`)
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


  useEffect(() => {
    setCheckedRows({});
  }, [sort, showOnlyNonFiltered]);

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
              <option value="single:rendement_score">Single - Score Rendement</option>
              <option value="single:rendement">Single - Rendement</option>
              <option value="single:rendement_sum">Single - Rendement Sum</option>
              <option value="single:score_swing_trade">Single - Score Swing Trade</option>
              <option value="mix:rendement_score">Mix - Score Rendement</option>
              <option value="mix:rendement">Mix - Rendement</option>
              <option value="mix:rendement_sum">Mix - Rendement Sum</option>
              <option value="mix:score_swing_trade">Mix - Score Swing Trade</option>
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
            <TableContainer component={Paper} sx={{ mb: 2, maxHeight: '70vh', overflow: 'auto' }}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ position: 'sticky', top: 0, zIndex: 2, backgroundColor: '#e0e0e0' }}></TableCell>
                    <TableCell sx={{ position: 'sticky', top: 0, zIndex: 2, backgroundColor: '#e0e0e0' }}></TableCell>
                    <TableCell colSpan={4} align="center" sx={{ position: 'sticky', top: 0, zIndex: 2, fontWeight: 'bold', backgroundColor: '#cff6c9', fontSize: '1rem' }}>LSDM</TableCell>
                    <TableCell colSpan={6} align="center" sx={{ position: 'sticky', top: 0, zIndex: 2, fontWeight: 'bold', backgroundColor: '#c8e6c9', fontSize: '1rem' }}>Single</TableCell>
                    <TableCell colSpan={6} align="center" sx={{ position: 'sticky', top: 0, zIndex: 2, fontWeight: 'bold', backgroundColor: '#bbdefb', fontSize: '1rem' }}>Mix</TableCell>
                    <TableCell sx={{ position: 'sticky', top: 0, zIndex: 2, backgroundColor: '#e0e0e0' }}></TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#e0e0e0' }}></TableCell> {/* Case à cocher */}
                    <TableCell sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Symbole</TableCell>
                    <TableCell sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#cff6c9' }}>Last Price</TableCell>
                    <TableCell sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#cff6c9' }}>Prédit Price</TableCell>
                    <TableCell sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#cff6c9', minWidth: 100, width: 100, maxWidth: 300  }}>Prédit Indice</TableCell>
                    <TableCell sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#cff6c9' }}>Position</TableCell>
                    <TableCell sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#c8e6c9', minWidth: 100, width: 100, maxWidth: 300 }}>Indice</TableCell>
                    <TableCell sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#c8e6c9' }}>Rendement</TableCell>
                    <TableCell sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#c8e6c9' }}>Rendement check</TableCell>
                    <TableCell sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#c8e6c9' }}>Score Rendement</TableCell>
                    <TableCell sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#c8e6c9' }}>Score Swing Trade</TableCell>
                    <TableCell sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#c8e6c9' }}>Durée moyenne trade</TableCell>
                    <TableCell sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#bbdefb', minWidth: 100, width: 100, maxWidth: 300 }}>Indice</TableCell>
                    <TableCell sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#bbdefb' }}>Rendement</TableCell>
                    <TableCell sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#bbdefb' }}>Rendement check</TableCell>
                    <TableCell sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#bbdefb' }}>Score Rendement</TableCell>
                    <TableCell sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#bbdefb' }}>Score Swing Trade</TableCell>
                    <TableCell sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#bbdefb' }}>Durée moyenne trade</TableCell>
                    <TableCell sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Détails</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {data.map((row, i) => {
                    let bgColor = undefined;
                    const indice = indices[row.single.symbol] as SignalInfo;
                    const lstmResult = lstmResults[row.single.symbol] as PreditLstm;
                    const indiceMixRaw = row.mix.symbol === null ? null : indicesMix[row.mix.symbol];
                    let indiceMix: SignalInfo | undefined = undefined;
                    if (indiceMixRaw && typeof indiceMixRaw === 'object' && 'type' in indiceMixRaw) {
                      indiceMix = indiceMixRaw as SignalInfo;
                    }

                    // Vérifie que indice est un objet et non une chaîne
                    if (indice && indice.type === 'BUY' && !row.single.result.fltredOut) bgColor = 'rgba(76, 175, 80, 0.5)';
                    if (indice && indice.type === 'SELL') bgColor = 'rgba(244, 67, 54, 0.05)';
                    return (
                      <TableRow
                        key={i}
                        sx={{
                          ...(bgColor ? { backgroundColor: bgColor } : {}),
                          '&:hover': {
                            backgroundColor: 'rgba(33, 150, 243, 0.12)',
                            transition: 'background-color 0.2s',
                          },
                        }}
                      >
                        <TableCell><input type="checkbox" checked={!!checkedRows[i]} onChange={e => setCheckedRows({...checkedRows, [i]: e.target.checked})} /></TableCell>
                        <TableCell>{row.single.symbol}</TableCell>
                        <TableCell>{
                            lstmResults[row.single.symbol] === 'pending'
                              ? (<CircularProgress size={16} />)
                              : (lstmResult && lstmResult.lastClose
                                  ? (lstmResult.lastClose)
                                  : '-')
                          }</TableCell>
                          <TableCell>{
                             lstmResults[row.single.symbol] === 'pending'
                               ? (<CircularProgress size={16} />)
                               : (lstmResult && lstmResult.predictedClose
                                  ? (lstmResult.predictedClose)
                                  : '-')
                           }</TableCell>
                        <TableCell>{
                          lstmResults[row.single.symbol] === 'pending'
                            ? (<CircularProgress size={16} />)
                            : (lstmResult && lstmResult.signal
                                ? (lstmResult.signal + ' (' + lstmResult.lastDate + ')')
                                : '-')
                        }</TableCell>
                        <TableCell>{
                           lstmResults[row.single.symbol] === 'pending'
                             ? (<CircularProgress size={16} />)
                             : (lstmResult && lstmResult?.position
                                ? (lstmResult?.position)
                                : '-')
                         }</TableCell>
                        <TableCell>{
                          indices[row.single.symbol] === 'pending'
                            ? (<CircularProgress size={16} />)
                            : (indice && indice.type
                                ? (row.single.result.fltredOut
                                    ? <span style={{ color: 'red', fontWeight: 'bold' }}>{indice.type + ' (' + indice.dateStr + ')'}</span>
                                    : indice.type + ' (' + indice.dateStr + ')')
                                : '-')
                        }</TableCell>
                        <TableCell>{(row.single.result.rendement * 100).toFixed(2)} %</TableCell>
                        <TableCell>{(row.single.check.rendement * 100).toFixed(2)} %</TableCell>
                        <TableCell>{(row.single.rendementScore * 100).toFixed(2)}</TableCell>
                        <TableCell>{row.single.result.scoreSwingTrade !== undefined ? (row.single.result.scoreSwingTrade).toFixed(2) : '-'}</TableCell>
                        <TableCell>{row.single.result.avgTradeBars !== undefined ? row.single.result.avgTradeBars.toFixed(2) : '-'}</TableCell>
                        <TableCell>{
                          indiceMixRaw === 'pending'
                            ? (<CircularProgress size={16} />)
                            : indiceMix
                              ? (row.mix.result.fltredOut
                                  ? <span style={{ color: 'red', fontWeight: 'bold' }}>{indiceMix.type + ' (' + indiceMix.dateStr + ')'}</span>
                                  : indiceMix.type + ' (' + indiceMix.dateStr + ')')
                              : '-'
                        }</TableCell>
                        <TableCell>{(row.mix.result.rendement * 100).toFixed(2)} %</TableCell>
                        <TableCell>{(row.mix.check.rendement * 100).toFixed(2)} %</TableCell>
                        <TableCell>{(row.mix.rendementScore * 100).toFixed(2)}</TableCell>
                        <TableCell>{row.mix.result.scoreSwingTrade !== undefined ? (row.mix.result.scoreSwingTrade).toFixed(2) : '-'}</TableCell>
                        <TableCell>{row.mix.result.avgTradeBars !== undefined ? row.mix.result.avgTradeBars.toFixed(2) : '-'}</TableCell>
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
      {/* Remplacement du Dialog par le composant réutilisable */}
      <BestPerformanceDialog
        open={open}
        selected={selected}
        bougies={bougies}
        bougiesLoading={bougiesLoading}
        bougiesError={bougiesError}
        onClose={() => setOpen(false)}
      />
    </>
  );
};

export default BestPerformanceSymbolBlock;
