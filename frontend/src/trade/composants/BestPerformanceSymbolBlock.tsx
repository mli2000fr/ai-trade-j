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
import {
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Checkbox,
  FormControlLabel,
  TextField,
  InputAdornment,
  Stack,
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import Tooltip from '@mui/material/Tooltip';

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
  finalResult: {
    rendement: number;
    tradeCount: number;
    winRate: number;
    maxDrawdown: number;
    avgPnL: number;
    sharpeRatio: number;
    profitFactor: number;
    avgTradeBars: number;
    maxTradeGain: number;
    maxTradeLoss: number;
    scoreSwingTrade?: number;
  };
  testResult: {
    rendement: number;
    tradeCount: number;
    winRate: number;
    maxDrawdown: number;
    avgPnL: number;
    sharpeRatio: number;
    profitFactor: number;
    avgTradeBars: number;
    maxTradeGain: number;
    maxTradeLoss: number;
    scoreSwingTrade?: number;
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
  finalResult: {
    rendement: number;
    maxDrawdown: number;
    avgPnL: number;
    sharpeRatio: number;
    tradeCount: number;
    winRate: number;
    profitFactor: number;
    avgTradeBars: number;
    maxTradeGain: number;
    maxTradeLoss: number;
    scoreSwingTrade?: number;
  };
  testResult: {
    rendement: number;
    maxDrawdown: number;
    avgPnL: number;
    sharpeRatio: number;
    tradeCount: number;
    winRate: number;
    profitFactor: number;
    avgTradeBars: number;
    maxTradeGain: number;
    maxTradeLoss: number;
    scoreSwingTrade?: number;
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
  name: string,
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
  const [sort, setSort] = useState<string>('single:score_swing_trade');
  const [topProfil, setTopProfil] = useState(true);
  const [bougies, setBougies] = useState<any[]>([]);
  const [bougiesLoading, setBougiesLoading] = useState(false);
  const [bougiesError, setBougiesError] = useState<string | null>(null);
  const [lstmResults, setLstmResults] = useState<{ [symbol: string]: PreditLstm | string}>({});
  const [searchValue, setSearchValue] = useState('');
  const [searchMode, setSearchMode] = useState(false);
  const [symbolsPerso, setSymbolsPerso] = useState<{id: string, name: string, symbols: string}[]>([]);
  const [selectedSymbolPerso, setSelectedSymbolPerso] = useState<string>('');
  const [symbolPersoData, setSymbolPersoData] = useState<{symbols: string[]} | null>(null);
  const [isToday, setIsToday] = useState(false);
  const [buySingleOnly, setBuySingleOnly] = useState(false);
  const [buyMixOnly, setBuyMixOnly] = useState(false);
  const [buyLstmOnly, setBuyLstmOnly] = useState(false);
  const chartContainerRef = useRef<HTMLDivElement>(null);

  const fetchData = (searchModeParam = false, searchValueParam = '') => {
    setLoading(true);
    setError(null);
    let url = `/api/result/global?`;
    if (searchModeParam && searchValueParam.trim()) {
      url += `&search=${encodeURIComponent(searchValueParam.trim())}`;
    } else {
      const type = sort.split(':')[0];
      const tri = sort.split(':')[1];
      url += `limit=${limit}&type=${type}&sort=${tri}`;
      if (topProfil) {
        url += `&topProfil=true`;
      }
    }
    fetch(url)
      .then(res => {
        if (!res.ok) throw new Error('Erreur API');
        return res.json();
      })
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    fetchData(false, '');
    // eslint-disable-next-line
  }, [limit, sort, topProfil]);

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
      // Ajout fetch LSTM
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
      //setBougiesLoading(true);
      setBougiesError(null);
      let url;
      if (isToday) {
        url = `/api/stra/getBougiesBySymbol?symbol=${encodeURIComponent(selected.single.symbol)}&isToday=true`;
      } else {
        url = `/api/stra/getBougiesBySymbol?symbol=${encodeURIComponent(selected.single.symbol)}&historique=250`;
      }
      fetch(url)
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
  }, [open, selected, isToday]);

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
  }, [sort, topProfil]);

  useEffect(() => {
    fetch('/api/result/symbol_pero')
      .then(res => res.json())
      .then(data => setSymbolsPerso(Array.isArray(data) ? data : []))
      .catch(() => setSymbolsPerso([]));
  }, []);

  return (
    <>
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Best Performance-Symbol
          </Typography>
          {/* Section moderne des contrôles */}
          <div style={{ marginBottom: 16 }}>
            <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap">
              <FormControl size="small" sx={{ minWidth: 100 }}>
                <InputLabel id="limit-select-label">Afficher</InputLabel>
                <Select
                  labelId="limit-select-label"
                  id="limit-select"
                  value={limit}
                  label="Afficher"
                  onChange={e => setLimit(Number(e.target.value))}
                  disabled={searchMode}
                >
                  <MenuItem value={20}>20</MenuItem>
                  <MenuItem value={30}>30</MenuItem>
                  <MenuItem value={50}>50</MenuItem>
                  <MenuItem value={100}>100</MenuItem>
                  <MenuItem value={10000}>All</MenuItem>
                </Select>
              </FormControl>
              <FormControl size="small" sx={{ minWidth: 180 }}>
                <InputLabel id="sort-select-label">Trié par</InputLabel>
                <Select
                  labelId="sort-select-label"
                  id="sort-select"
                  value={sort}
                  label="Trié par"
                  onChange={e => setSort(e.target.value)}
                  disabled={searchMode}
                >
                  <MenuItem value="single:rendement">Single - Rendement</MenuItem>
                  <MenuItem value="single:score_swing_trade">Single - Score Swing Trade</MenuItem>
                  <MenuItem value="mix:rendement">Mix - Rendement</MenuItem>
                  <MenuItem value="mix:score_swing_trade">Mix - Score Swing Trade</MenuItem>
                </Select>
              </FormControl>

              <FormControlLabel
                control={
                  <Checkbox
                    checked={topProfil}
                    onChange={e => setTopProfil(e.target.checked)}
                    disabled={searchMode}
                    size="small"
                  />
                }
                label="Top profil"
                sx={{ ml: 2 }}
              />
              <FormControlLabel
                control={
                  <Checkbox
                    checked={buySingleOnly}
                    onChange={e => setBuySingleOnly(e.target.checked)}
                    disabled={searchMode}
                    size="small"
                  />
                }
                label="Buy/Single"
                sx={{ ml: 2 }}
              />
              <FormControlLabel
                control={
                  <Checkbox
                    checked={buyMixOnly}
                    onChange={e => setBuyMixOnly(e.target.checked)}
                    disabled={searchMode}
                    size="small"
                  />
                }
                label="Buy/Mix"
                sx={{ ml: 2 }}
              />
              <FormControlLabel
                control={
                  <Checkbox
                    checked={buyLstmOnly}
                    onChange={e => setBuyLstmOnly(e.target.checked)}
                    disabled={searchMode}
                    size="small"
                  />
                }
                label="Buy/Lstm"
                sx={{ ml: 2 }}
              />
              <Button
                variant="contained"
                color="primary"
                size="small"
                startIcon={<ContentCopyIcon />}
                onClick={handleCopy}
                disabled={Object.values(checkedRows).every(v => !v)}
                sx={{ borderRadius: 2, boxShadow: 1 }}
              >
                Copier
              </Button>
              <FormControl size="small" sx={{ minWidth: 180 }}>
                              <InputLabel id="symbol-perso-select-label">Symbols personnalisés</InputLabel>
                              <Select
                                labelId="symbol-perso-select-label"
                                id="symbol-perso-select"
                                value={selectedSymbolPerso}
                                label="Symbols personnalisés"
                                onChange={e => {
                                    setSearchValue('');
                                  setSelectedSymbolPerso(e.target.value);
                                  setSearchMode(true);
                                  fetchData(true, e.target.value);
                                }}
                              >
                                {symbolsPerso.map(sym => (
                                  <MenuItem key={sym.id} value={sym.symbols}>{sym.name}</MenuItem>
                                ))}
                              </Select>
                            </FormControl>
              <TextField
                size="small"
                variant="outlined"
                placeholder="Rechercher..."
                value={searchValue}
                onChange={e => setSearchValue(e.target.value)}
                disabled={searchMode && loading}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <SearchIcon />
                    </InputAdornment>
                  ),
                  sx: { borderRadius: 2 }
                }}
                sx={{ minWidth: 160 }}
              />
              <Button
                variant="contained"
                color="secondary"
                size="small"
                onClick={() => {
                  if (searchValue.trim()) {
                    setSearchMode(true);
                    setSelectedSymbolPerso('');
                    fetchData(true, searchValue);
                  }
                }}
                disabled={!searchValue.trim()}
                sx={{ borderRadius: 2, boxShadow: 1 }}
              >
                Chercher
              </Button>
              <Button
                variant="outlined"
                color="inherit"
                size="small"
                onClick={() => {
                  setSearchValue('');
                  setSearchMode(false);
                  setSelectedSymbolPerso('');
                  fetchData(false, '');
                }}
                sx={{ borderRadius: 2, boxShadow: 1 }}
                disabled={!searchMode && !searchValue && !selectedSymbolPerso}
              >
                Réinitier
              </Button>
            </Stack>
          </div>
          {/* Bloc recherche sur une nouvelle ligne */}
          <div style={{ marginBottom: 16, display: 'flex', alignItems: 'center', flexWrap: 'wrap' }}>
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
                    <TableCell colSpan={4} align="center" sx={{ position: 'sticky', top: 0, zIndex: 2, fontWeight: 'bold', backgroundColor: '#cff6c9', fontSize: '1rem' }}>LSTM</TableCell>
                    <TableCell colSpan={9} align="center" sx={{ position: 'sticky', top: 0, zIndex: 2, fontWeight: 'bold', backgroundColor: '#c8e6c9', fontSize: '1rem' }}>Single</TableCell>
                    <TableCell colSpan={9} align="center" sx={{ position: 'sticky', top: 0, zIndex: 2, fontWeight: 'bold', backgroundColor: '#bbdefb', fontSize: '1rem' }}>Mix</TableCell>
                    <TableCell sx={{ position: 'sticky', top: 0, zIndex: 2, backgroundColor: '#e0e0e0' }}></TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell align="center" sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#e0e0e0' }}></TableCell> {/* Case à cocher */}
                    <TableCell align="center" sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Symbole</TableCell>
                    <TableCell align="center" sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#cff6c9' }}>Last Price</TableCell>
                    <TableCell align="center" sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#cff6c9' }}>Prédit Price</TableCell>
                    <TableCell align="center" sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#cff6c9', minWidth: 120, width: 130, maxWidth: 300  }}>Prédit Indice</TableCell>
                    <TableCell align="center" sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#cff6c9', minWidth: 180, width: 180, maxWidth: 300  }}>Position</TableCell>
                    <TableCell align="center" sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#c8e6c9', minWidth: 100, width: 100, maxWidth: 300 }}>Indice</TableCell>
                    <TableCell align="center" sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#c8e6c9', minWidth: 120, width: 130, maxWidth: 300 }}>Rendement</TableCell>
                    <TableCell align="center" sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#c8e6c9', minWidth: 100, width: 130, maxWidth: 300 }}>
                    <Tooltip title="Profit moyen par trade (Doit être positif)" arrow placement="top" enterDelay={200} leaveDelay={100}
                      slotProps={{ tooltip: { sx: { fontSize: '1.1rem', padding: '6px 12px' } } }}>
                      <span>Avg PnL</span>
                    </Tooltip>
                    </TableCell>
                    <TableCell align="center" sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#c8e6c9', minWidth: 100, width: 130, maxWidth: 300 }}>
                    <Tooltip title="Doit être > 1 pour être intéressant" arrow placement="top" enterDelay={200} leaveDelay={100}
                       slotProps={{ tooltip: { sx: { fontSize: '1.1rem', padding: '6px 12px' } } }}>
                       <span>Profit Factor</span>
                     </Tooltip>
                    </TableCell>
                    <TableCell align="center" sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#c8e6c9' }}>Max Drawdown</TableCell>
                    <TableCell align="center" sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#c8e6c9', minWidth: 100, width: 130, maxWidth: 300 }}>
                    <Tooltip title="Doit être positif et idéalement > 1" arrow placement="top" enterDelay={200} leaveDelay={100}
                        slotProps={{ tooltip: { sx: { fontSize: '1.1rem', padding: '6px 12px' } } }}>
                        <span>Sharpe Ratio</span>
                      </Tooltip>
                    </TableCell>
                    <TableCell align="center" sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#c8e6c9' }}>Win Rate</TableCell>
                    <TableCell align="center" sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#c8e6c9', minWidth: 100, width: 130, maxWidth: 300 }}>Score Swing Trade</TableCell>
                    <TableCell align="center" sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#c8e6c9' }}>Durée moyenne trade</TableCell>
                    <TableCell align="center" sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#bbdefb', minWidth: 100, width: 100, maxWidth: 300 }}>Indice</TableCell>
                    <TableCell align="center" sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#bbdefb', minWidth: 120, width: 130, maxWidth: 300  }}>Rendement</TableCell>
                    <TableCell align="center" sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#bbdefb', minWidth: 100, width: 130, maxWidth: 300 }}>
                    <Tooltip title="Profit moyen par trade (Doit être positif)" arrow placement="top" enterDelay={200} leaveDelay={100}
                      slotProps={{ tooltip: { sx: { fontSize: '1.1rem', padding: '6px 12px' } } }}>
                      <span>Avg PnL</span>
                    </Tooltip>
                    </TableCell>
                    <TableCell align="center" sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#bbdefb', minWidth: 100, width: 130, maxWidth: 300 }}>
                    <Tooltip title="Doit être > 1 pour être intéressant" arrow placement="top" enterDelay={200} leaveDelay={100}
                       slotProps={{ tooltip: { sx: { fontSize: '1.1rem', padding: '6px 12px' } } }}>
                       <span>Profit Factor</span>
                     </Tooltip>
                    </TableCell>
                    <TableCell align="center" sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#bbdefb' }}>Max Drawdown</TableCell>
                    <TableCell align="center" sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#bbdefb', minWidth: 100, width: 130, maxWidth: 300 }}>
                    <Tooltip title="Doit être positif et idéalement > 1" arrow placement="top" enterDelay={200} leaveDelay={100}
                        slotProps={{ tooltip: { sx: { fontSize: '1.1rem', padding: '6px 12px' } } }}>
                        <span>Sharpe Ratio</span>
                      </Tooltip>
                    </TableCell>
                    <TableCell align="center" sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#bbdefb' }}>Win Rate</TableCell>
                    <TableCell align="center" sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#bbdefb', minWidth: 100, width: 130, maxWidth: 300  }}>Score Swing Trade</TableCell>
                    <TableCell align="center" sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#bbdefb' }}>Durée moyenne trade</TableCell>
                    <TableCell align="center" sx={{ position: 'sticky', top: 36, zIndex: 2, fontWeight: 'bold', backgroundColor: '#e0e0e0' }}>Détails</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {data.filter(row => {
                    let singleOk = true;
                    let mixOk = true;
                    let lstmOk = true;
                    if (buySingleOnly) {
                      const indice = indices[row.single.symbol];
                      singleOk = (typeof indice === 'object' && indice.type && indice.type.startsWith('BUY')) ? true : false;
                    }
                    if (buyMixOnly) {
                      const indiceMix = indicesMix[row.mix.symbol ?? ''];
                      mixOk = (typeof indiceMix === 'object' && indiceMix.type && indiceMix.type.startsWith('BUY')) ? true : false;
                    }
                    if (buyLstmOnly) {
                      const lstm = lstmResults[row.single.symbol];
                      lstmOk = (typeof lstm === 'object' && lstm.signal && typeof lstm.signal === 'object' && lstm.signal.type && lstm.signal.type.startsWith('BUY')) ? true : false;
                    }
                    return singleOk && mixOk && lstmOk;
                  }).map((row, i) => {
                    let bgColor = undefined;
                    const indice = indices[row.single.symbol] as SignalInfo;
                    const lstmResult = lstmResults[row.single.symbol] as PreditLstm;
                    const indiceMixRaw = row.mix.symbol === null ? null : indicesMix[row.mix.symbol];
                    let indiceMix: SignalInfo | undefined = undefined;
                    if (indiceMixRaw && typeof indiceMixRaw === 'object' && 'type' in indiceMixRaw) {
                      indiceMix = indiceMixRaw as SignalInfo;
                    }

                    // Vérifie que indice est un objet et non une chaîne
                    if (indice && indice.type === 'BUY') bgColor = 'rgba(76, 175, 80, 0.5)';
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
                        <TableCell align="center">
                          <Tooltip title={row?.name || row.single.symbol} arrow placement="top" enterDelay={200} leaveDelay={100}
                            slotProps={{ tooltip: { sx: { fontSize: '1.1rem', padding: '6px 12px' } } }}>
                            <span>{row.single.symbol}</span>
                          </Tooltip>
                        </TableCell>
                        <TableCell align="center">{
                            lstmResults[row.single.symbol] === 'pending'
                              ? (<CircularProgress size={16} />)
                              : (lstmResult && lstmResult.lastClose
                                  ? (lstmResult.lastClose)
                                  : '-')
                          }</TableCell>
                          <TableCell align="center">{
                             lstmResults[row.single.symbol] === 'pending'
                               ? (<CircularProgress size={16} />)
                               : (lstmResult && lstmResult.predictedClose
                                  ? (lstmResult.predictedClose)
                                  : '-')
                           }</TableCell>
                        <TableCell align="center">{
                          lstmResults[row.single.symbol] === 'pending'
                            ? (<CircularProgress size={16} />)
                            : (lstmResult && lstmResult.signal
                                ? (lstmResult.signal + ' (' + lstmResult.lastDate + ')')
                                : '-')
                        }</TableCell>
                        <TableCell align="center">{
                           lstmResults[row.single.symbol] === 'pending'
                             ? (<CircularProgress size={16} />)
                             : (lstmResult && lstmResult?.position
                                ? (lstmResult?.position)
                                : '-')
                         }</TableCell>
                        <TableCell align="center">{
                          indices[row.single.symbol] === 'pending'
                            ? (<CircularProgress size={16} />)
                            : (indice && indice.type
                                ? (indice.type + ' (' + indice.dateStr + ')')
                                : '-')
                        }</TableCell>
                        <TableCell align="center"><b>{(row.single.finalResult.rendement * 100).toFixed(2)} %</b> ({(row.single.testResult.rendement * 100).toFixed(2)} %)</TableCell>
                        <TableCell align="center">{row.single.finalResult.avgPnL.toFixed(2)}</TableCell>
                        <TableCell align="center">{row.single.finalResult.profitFactor.toFixed(2)}</TableCell>
                        <TableCell align="center">{(row.single.finalResult.maxDrawdown * 100).toFixed(2)} %</TableCell>
                        <TableCell align="center">{row.single.finalResult.sharpeRatio.toFixed(2)}</TableCell>
                        <TableCell align="center">{(row.single.finalResult.winRate * 100).toFixed(2)} %</TableCell>
                        <TableCell align="center">{row.single.finalResult.scoreSwingTrade !== undefined ? (row.single.finalResult.scoreSwingTrade).toFixed(2) : '-'}</TableCell>
                        <TableCell align="center">{row.single.finalResult.avgTradeBars !== undefined ? row.single.finalResult.avgTradeBars.toFixed(2) : '-'}</TableCell>
                        <TableCell align="center">{
                          indiceMixRaw === 'pending'
                            ? (<CircularProgress size={16} />)
                            : indiceMix
                              ? (indiceMix.type + ' (' + indiceMix.dateStr + ')')
                              : '-'
                        }</TableCell>
                        <TableCell align="center"><b>{(row.mix.finalResult.rendement * 100).toFixed(2)} %</b>| {(row.mix.testResult.rendement * 100).toFixed(2)} %</TableCell>
                        <TableCell align="center">{row.mix.finalResult.avgPnL.toFixed(2)}</TableCell>
                        <TableCell align="center">{row.mix.finalResult.profitFactor.toFixed(2)}</TableCell>
                        <TableCell align="center">{(row.mix.finalResult.maxDrawdown * 100).toFixed(2)} %</TableCell>
                        <TableCell align="center">{row.mix.finalResult.sharpeRatio.toFixed(2)}</TableCell>
                        <TableCell align="center">{(row.mix.finalResult.winRate * 100).toFixed(2)} %</TableCell>
                        <TableCell align="center">{row.mix.finalResult.scoreSwingTrade !== undefined ? (row.mix.finalResult.scoreSwingTrade).toFixed(2) : '-'}</TableCell>
                        <TableCell align="center">{row.mix.finalResult.avgTradeBars !== undefined ? row.mix.finalResult.avgTradeBars.toFixed(2) : '-'}</TableCell>
                        <TableCell align="center"><Button size="small" variant="outlined" onClick={() => { setSelected(row); setOpen(true); }}>Détails</Button></TableCell>
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
        isToday={isToday}
        setIsToday={setIsToday}
      />
    </>
  );
};

export default BestPerformanceSymbolBlock;
