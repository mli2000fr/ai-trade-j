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
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import Accordion from '@mui/material/Accordion';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import BestPerformanceDialog from './BestPerformanceDialog';

interface PortfolioBlockProps {
  portfolio: any;
  lastUpdate: Date | null;
  loading: boolean;
  compteId?: string | null;
  resetSellErrorKey?: string | number;
  onRefreshPortfolio?: () => void;
}
interface SignalInfo {
    symbol: string;
    type: string;
    date?: string | null;
    dateStr?: string;
}

const PortfolioBlock: React.FC<PortfolioBlockProps> = ({ portfolio, lastUpdate, loading, compteId, resetSellErrorKey, onRefreshPortfolio }) => {
  const initialDeposit = portfolio && portfolio.initialDeposit !== undefined ? Number(portfolio.initialDeposit) : undefined;
  const equity = portfolio && portfolio.account?.equity !== undefined ? Number(portfolio.account.equity) : undefined;
  const plTotal = initialDeposit !== undefined && initialDeposit !== 0 && !isNaN(initialDeposit) && equity !== undefined && !isNaN(equity) ? equity - initialDeposit : undefined;
  const plPercent = initialDeposit !== undefined && initialDeposit !== 0 && !isNaN(initialDeposit) && equity !== undefined && !isNaN(equity) ? ((equity - initialDeposit) / initialDeposit) * 100 : undefined;

  // Ajout d'un cache local pour les indices
  const [indices, setIndices] = useState<{ [symbol: string]: SignalInfo | string }>({});
  const [sellError, setSellError] = useState<string | null>(null);
  const [sellSuccess, setSellSuccess] = useState<string | null>(null);
  const [openDialog, setOpenDialog] = useState(false);
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);
  const [dialogData, setDialogData] = useState<any>(null);
  const [bougies, setBougies] = useState<any[]>([]);
  const [bougiesLoading, setBougiesLoading] = useState(false);
  const [bougiesError, setBougiesError] = useState<string | null>(null);
  const [indiceSingle, setIndiceSingle] = useState<any>(null);
  const [indiceMix, setIndiceMix] = useState<any>(null);
  const [predict, setPredict] = useState<any>(null);
  const [selected, setSelected] = useState<any>(null);
  const [isToday, setIsToday] = useState(false);

  // Réinitialisation de sellError quand resetSellErrorKey change
  useEffect(() => {
    setSellError(null);
  }, [resetSellErrorKey]);

  // Masquer la notification de succès après 5 secondes
  useEffect(() => {
    if (sellSuccess) {
      const timer = setTimeout(() => setSellSuccess(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [sellSuccess]);
  useEffect(() => {
    if (sellError) {
      const timer = setTimeout(() => setSellError(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [sellError]);

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
        .then((data: SignalInfo) => {
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
        setSellSuccess("Vente effectuée avec succès !");
        // Rafraîchir le tableau via le callback passé en props
        if (onRefreshPortfolio) {
             const timer = setTimeout(() =>  onRefreshPortfolio(), 5000);
                  return () => clearTimeout(timer);
        }
      }
    } catch (e: any) {
      setSellError(e.message || 'Erreur lors de la vente.');
    }
  };

  // Fonction pour charger les données du Dialog
  const handleOpenDialog = async (symbol: string) => {
    setSelectedSymbol(symbol);
    setOpenDialog(true);
    setBougiesLoading(true);
    setBougiesError(null);
    try {
      // Bougies
      let url;
      if (isToday) {
         url = `/api/stra/getBougiesBySymbol?symbol=${encodeURIComponent(symbol)}&isToday=true`;
       } else {
         url = `/api/stra/getBougiesBySymbol?symbol=${encodeURIComponent(symbol)}&historique=250`;
       }
      const bougiesRes = await fetch(url);
      const bougiesData = await bougiesRes.json();
      // infos
      const infosAction = await fetch(`/api/result/infosSymbol?symbol=${symbol}&historique=250`);
      const infosActionData = await infosAction.json();
      // Indice single
      const indiceSingleRes = await fetch(`/api/stra/strategies/get_indice?symbol=${symbol}`);
      const indiceSingleData = await indiceSingleRes.json();
      // Indice mix
      const indiceMixRes = await fetch(`/api/best-combination/get_indice?symbol=${symbol}`);
      const indiceMixData = await indiceMixRes.json();
      // Prédiction
      const predictRes = await fetch(`/api/lstm/predict?symbol=${symbol}`);
      const predictData = await predictRes.json();
      setSelected({
        single: infosActionData.single,
        mix: infosActionData.mix,
        bougies: bougiesData,
        indiceSingle: indiceSingleData,
        indiceMix: indiceMixData,
        predict: predictData
      });
    } catch (e) {
      setSelected(null);
      setBougiesError('Erreur lors du chargement des données');
    }
    setBougiesLoading(false);
  };

  // Fonction utilitaire pour afficher un objet sous forme de tableau
  const renderObjectTable = (obj: Record<string, any>) => {
    if (!obj) return null;
    return (
      <Table size="small" sx={{ mb: 2 }}>
        <TableBody>
          {Object.entries(obj).map(([key, value]) => (
            <TableRow key={key}>
              <TableCell sx={{ fontWeight: 'bold', width: '40%' }}>{key}</TableCell>
              <TableCell>{typeof value === 'number' ? value.toFixed(4) : String(value)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    );
  };

  // Nouvelle fonction pour gérer le clic sur Today et appeler le backend
  const handleSetIsToday = async (value: boolean) => {
    setIsToday(value);
    if (selectedSymbol) {
      setBougiesError(null);
      try {
          let url;
          if (value) {
           url = `/api/stra/getBougiesBySymbol?symbol=${encodeURIComponent(selected.indiceSingle.symbol)}&isToday=true`;
         } else {
           url = `/api/stra/getBougiesBySymbol?symbol=${encodeURIComponent(selected.indiceSingle.symbol)}&historique=250`;
         }
        // Bougies : on ajoute le paramètre today si nécessaire
        const bougiesRes = await fetch(url);
        const bougiesData = await bougiesRes.json();
        setSelected((prev: any) => ({
          ...prev,
          bougies: bougiesData
        }));
      } catch (e) {
        setSelected(null);
        setBougiesError('Erreur lors du chargement des données');
      }
      setBougiesLoading(false);
    }
  };

  return (
    <Card sx={{ mb: 3, backgroundColor: '#f5f5f5' }}>
      <CardContent>
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
                      <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#e0e0e0' }}></TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {portfolio.positions.map((pos: any, i: number) => {
                      const isNegative = pos.unrealized_pl < 0;
                      const cellStyle = isNegative ? { color: 'error.main' } : {};
                      const indice = indices[pos.symbol] as SignalInfo | string;
                      const isSellSignal = typeof indice === 'object' && indice !== null && indice.type === 'SELL';
                      const isPendingSignal = indice === 'pending';
                      const signalCellStyle = isSellSignal ? { color: 'error.main', fontWeight: 'bold' } : cellStyle;
                      let signalCellContent;
                      if (isPendingSignal) {
                        signalCellContent = <CircularProgress size={16} />;
                      } else if (isSellSignal) {
                        signalCellContent = <Button variant="contained" color="error" size="small" onClick={() => handleSell(pos)} disabled={pos.qty <= 0}>SELL</Button>;
                      } else if (typeof indice === 'object' && indice !== null && indice.type) {
                        signalCellContent = indice.type + ' (' + indice.dateStr + ')';
                      } else if (typeof indice === 'string') {
                        signalCellContent = indice;
                      } else {
                        signalCellContent = '-';
                      }
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
                          <TableCell sx={signalCellStyle}>{signalCellContent}</TableCell>
                          <TableCell sx={cellStyle}>{pos.avg_entry_price !== undefined && pos.avg_entry_price !== null ? Number(pos.avg_entry_price).toFixed(2) + ' $' : '-'}</TableCell>
                          <TableCell sx={cellStyle}>{pos.current_price !== undefined && pos.current_price !== null ? Number(pos.current_price).toFixed(2) + ' $' : '-'}</TableCell>
                          <TableCell sx={cellStyle}>{pos.qty}</TableCell>
                          <TableCell sx={cellStyle}>{pos.current_price !== undefined && pos.current_price !== null ? (Number(pos.qty) * Number(pos.current_price)).toFixed(2) + ' $' : '-'}</TableCell>
                          <TableCell sx={cellStyle}>{pos.unrealized_plpc !== undefined && pos.unrealized_plpc !== null ? (Number(pos.unrealized_plpc) * 100).toFixed(3) + ' %' : '-'}</TableCell>
                          <TableCell sx={cellStyle}>{pos.unrealized_pl !== undefined && pos.unrealized_pl !== null ? Number(pos.unrealized_pl).toFixed(2) + ' $' : '-'}</TableCell>
                          <TableCell sx={cellStyle}>
                            <Button size="small" variant="outlined" onClick={() => handleOpenDialog(pos.symbol)}>Détails</Button>
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </TableContainer>
            )}
            {sellSuccess && <Alert severity="success" sx={{ mb: 2 }}>{sellSuccess}</Alert>}
            {sellError && <Alert severity="error" sx={{ mb: 2 }}>{sellError}</Alert>}
          </>
        )}
      </CardContent>
      {/* Ajout du Dialog ici, à la racine du composant  open={openDialog} onClose={() => setOpenDialog(false)}*/}
      <BestPerformanceDialog
              open={openDialog}
              selected={selected}
              bougies={selected?.bougies}
              bougiesLoading={bougiesLoading}
              bougiesError={bougiesError}
              onClose={() => setOpenDialog(false)}
              isToday={isToday}
              setIsToday={handleSetIsToday}
            />
    </Card>
  );
};

export default PortfolioBlock;
