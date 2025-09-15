import React from 'react';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';
import Accordion from '@mui/material/Accordion';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import CircularProgress from '@mui/material/CircularProgress';
import Alert from '@mui/material/Alert';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import BougiesChart from './BougiesChart';

interface BestPerformanceDialogProps {
  open: boolean;
  selected: any;
  bougies: any[];
  bougiesLoading: boolean;
  bougiesError: string | null;
  onClose: () => void;
}

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

const BestPerformanceDialog: React.FC<BestPerformanceDialogProps> = ({
  open,
  selected,
  bougies,
  bougiesLoading,
  bougiesError,
  onClose,
}) => (
  <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth PaperProps={{ sx: { maxWidth: '60vw' } }}>
    <DialogTitle>Détails de la performance</DialogTitle>
    <DialogContent>
      {selected && (
        <div>
          <Typography variant="h6" sx={{ mb: 2, color: '#1976d2' }}>Symbole : {selected.single.symbol}</Typography>
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
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ backgroundColor: '#e0e0e0' }}></TableCell>
                    <TableCell align="center" sx={{ fontWeight: 'bold', backgroundColor: '#c8e6c9', fontSize: '1rem' }}>Single</TableCell>
                    <TableCell align="center" sx={{ fontWeight: 'bold', backgroundColor: '#bbdefb', fontSize: '1rem' }}>Mix</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  <TableRow>
                    <TableCell sx={{ fontWeight: 'bold' }}>Rendement Sum</TableCell>
                    <TableCell align="center" >{(selected.single.rendementSum * 100).toFixed(2)} %</TableCell>
                    <TableCell align="center" >{(selected.mix.rendementSum * 100).toFixed(2)} %</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell sx={{ fontWeight: 'bold' }}>Rendement Diff</TableCell>
                    <TableCell align="center" >{(selected.single.rendementDiff * 100).toFixed(2)} %</TableCell>
                    <TableCell align="center" >{(selected.mix.rendementDiff * 100).toFixed(2)} %</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell sx={{ fontWeight: 'bold' }}>Rendement Score</TableCell>
                    <TableCell align="center" >{(selected.single.rendementScore * 100).toFixed(2)}</TableCell>
                    <TableCell align="center" >{(selected.mix.rendementScore * 100).toFixed(2)}</TableCell>
                  </TableRow>
                </TableBody>
              </Table>
              <br/>
              <Table size="small" sx={{ mb: 2, backgroundColor: '#f9f9f9' }}>
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ backgroundColor: '#e0e0e0' }}></TableCell>
                    <TableCell colSpan={2} align="center" sx={{ fontWeight: 'bold', backgroundColor: '#c8e6c9', fontSize: '1rem' }}>Single</TableCell>
                    <TableCell colSpan={2} align="center" sx={{ fontWeight: 'bold', backgroundColor: '#bbdefb', fontSize: '1rem' }}>Mix</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell sx={{ fontWeight: 'bold', width: '40%' }}>Métrique</TableCell>
                    <TableCell align="center"  sx={{ fontWeight: 'bold' }}>Résultat</TableCell>
                    <TableCell align="center"  sx={{ fontWeight: 'bold' }}>Vérification</TableCell>
                    <TableCell align="center"  sx={{ fontWeight: 'bold' }}>Résultat</TableCell>
                    <TableCell align="center"  sx={{ fontWeight: 'bold' }}>Vérification</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {Array.from(new Set([...Object.keys(selected.single.result || {}), ...Object.keys(selected.single.check || {})])).map((key) => {
                    const resultObjSingle = selected.single.result as Record<string, any>;
                    const checkObjSingle = selected.single.check as Record<string, any>;
                    const resultObjMix = selected.mix.result as Record<string, any>;
                    const checkObjMix = selected.mix.check as Record<string, any>;

                    return (
                      <TableRow key={key}>
                        <TableCell sx={{ fontWeight: 'bold' }}>{key}</TableCell>
                        <TableCell align="center" >
                          {typeof resultObjSingle?.[key] === 'number'
                            ? (Math.abs(resultObjSingle[key]) > 1
                                ? resultObjSingle[key].toFixed(2)
                                : (resultObjSingle[key] * 100).toFixed(2) + (key.toLowerCase().includes('pct') || key.toLowerCase().includes('rate') || key.toLowerCase().includes('drawdown') || key.toLowerCase().includes('rendement') ? ' %' : ''))
                            : typeof resultObjSingle?.[key] === 'boolean'
                              ? resultObjSingle[key] ? 'Oui' : 'Non'
                              : resultObjSingle?.[key] ?? '-'}
                        </TableCell>
                        <TableCell align="center" >
                          {typeof checkObjSingle?.[key] === 'number'
                            ? (Math.abs(checkObjSingle[key]) > 1
                                ? checkObjSingle[key].toFixed(2)
                                : (checkObjSingle[key] * 100).toFixed(2) + (key.toLowerCase().includes('pct') || key.toLowerCase().includes('rate') || key.toLowerCase().includes('drawdown') || key.toLowerCase().includes('rendement') ? ' %' : ''))
                            : typeof checkObjSingle?.[key] === 'boolean'
                              ? checkObjSingle[key] ? 'Oui' : 'Non'
                              : checkObjSingle?.[key] ?? '-'}
                        </TableCell>
                        <TableCell align="center" >
                          {typeof resultObjMix?.[key] === 'number'
                            ? (Math.abs(resultObjMix[key]) > 1
                                ? resultObjMix[key].toFixed(2)
                                : (resultObjMix[key] * 100).toFixed(2) + (key.toLowerCase().includes('pct') || key.toLowerCase().includes('rate') || key.toLowerCase().includes('drawdown') || key.toLowerCase().includes('rendement') ? ' %' : ''))
                            : typeof resultObjMix?.[key] === 'boolean'
                              ? resultObjMix[key] ? 'Oui' : 'Non'
                              : resultObjMix?.[key] ?? '-'}
                        </TableCell>
                        <TableCell align="center" >
                          {typeof checkObjMix?.[key] === 'number'
                            ? (Math.abs(checkObjMix[key]) > 1
                                ? checkObjMix[key].toFixed(2)
                                : (checkObjMix[key] * 100).toFixed(2) + (key.toLowerCase().includes('pct') || key.toLowerCase().includes('rate') || key.toLowerCase().includes('drawdown') || key.toLowerCase().includes('rendement') ? ' %' : ''))
                            : typeof checkObjMix?.[key] === 'boolean'
                              ? checkObjMix[key] ? 'Oui' : 'Non'
                              : checkObjMix?.[key] ?? '-'}
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
              <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>Stratégie d'entrée (Single)</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Typography variant="body2" sx={{ mb: 1 }}>Nom : <b>{selected.single.entryName}</b></Typography>
              {selected.single.entryParams && renderObjectTable(selected.single.entryParams)}
            </AccordionDetails>
          </Accordion>
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>Stratégie de sortie (Single)</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Typography variant="body2" sx={{ mb: 1 }}>Nom : <b>{selected.single.exitName}</b></Typography>
              {selected.single.exitParams && renderObjectTable(selected.single.exitParams)}
            </AccordionDetails>
          </Accordion>
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>Paramètres d'optimisation (Single)</Typography>
            </AccordionSummary>
            <AccordionDetails>
              {selected.single.paramsOptim && renderObjectTable(selected.single.paramsOptim)}
            </AccordionDetails>
          </Accordion>
        </div>
      )}
    </DialogContent>
    <DialogActions>
      <Button onClick={onClose}>Fermer</Button>
    </DialogActions>
  </Dialog>
);

export default BestPerformanceDialog;

