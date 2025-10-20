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
import Checkbox from '@mui/material/Checkbox';
import FormControlLabel from '@mui/material/FormControlLabel';

interface BestPerformanceDialogProps {
  open: boolean;
  selected: any;
  bougies: any[];
  bougiesLoading: boolean;
  bougiesError: string | null;
  onClose: () => void;
  isToday: boolean;
  setIsToday: (v: boolean) => void;
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
  isToday,
  setIsToday
}) => (
  <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth PaperProps={{ sx: { maxWidth: '60vw' } }}>
    <DialogActions>
      <Button onClick={onClose}>Fermer</Button>
    </DialogActions>
    {bougiesLoading ? (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 300 }}>
        <CircularProgress size={60} />
      </div>
    ) : (
      <DialogContent>
        {selected && (
          <div>
            <Typography variant="h6" sx={{ mb: 2, color: '#1976d2' }}>Symbole : {selected.single.symbol}</Typography>
            <Accordion defaultExpanded>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>Graphique des bougies</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <FormControlLabel
                   control={<Checkbox checked={isToday} onChange={e => setIsToday(e.target.checked)} />}
                   label="Today"
                   sx={{ ml: 2 }}
                 />
                {bougiesLoading ? (
                  <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 300 }}>
                    <CircularProgress />
                  </div>
                ) : bougiesError ? (
                  <Alert severity="error">{bougiesError}</Alert>
                ) : (Array.isArray(bougies) && bougies.length > 0 ? (
                  <BougiesChart bougies={bougies} />
                ) : (
                  <Typography variant="body2">Aucune donnée de bougie disponible.</Typography>
                ))}
              </AccordionDetails>
            </Accordion>
            { selected?.indiceSingle && selected?.indiceMix && selected?.predict && (
                <Accordion defaultExpanded>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>Données Indices & Prédiction</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                    <Table size="small" sx={{ mt: 2, backgroundColor: '#f1f8e9' }}>
                       <TableHead>
                         <TableRow>
                           <TableCell sx={{ backgroundColor: '#e0e0e0' }}></TableCell>
                           <TableCell align="center" sx={{ fontWeight: 'bold', backgroundColor: '#cff6c9', fontSize: '1rem' }}>LSDM</TableCell>
                           <TableCell align="center" sx={{ fontWeight: 'bold', backgroundColor: '#c8e6c9', fontSize: '1rem' }}>Single</TableCell>
                           <TableCell align="center" sx={{ fontWeight: 'bold', backgroundColor: '#bbdefb', fontSize: '1rem' }}>Mix</TableCell>
                         </TableRow>
                       </TableHead>
                       <TableBody>
                         <TableRow>
                           <TableCell sx={{ fontWeight: 'bold' }}>Single</TableCell>
                           <TableCell align="center" >{selected.predict?.signal} ({selected.predict?.lastClose} / {selected.predict?.predictedClose.toFixed(2)} / {selected.predict?.position})</TableCell>
                           <TableCell align="center" >{selected.indiceSingle?.type}</TableCell>
                           <TableCell align="center" >{selected.indiceMix?.type}</TableCell>
                         </TableRow>
                         <TableRow>
                           <TableCell sx={{ fontWeight: 'bold' }}>Date</TableCell>
                           <TableCell align="center" >{selected.predict?.lastDate}</TableCell>
                           <TableCell align="center" >{selected.indiceSingle?.dateStr}</TableCell>
                           <TableCell align="center" >{selected.indiceMix?.dateStr}</TableCell>
                         </TableRow>
                       </TableBody>
                     </Table>
                    </AccordionDetails>
                  </Accordion>
                )}

            <Accordion defaultExpanded={!selected?.indiceSingle && !selected?.indiceMix && !selected?.predict}>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>Résultats & Vérification</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Table size="small" sx={{ mb: 2, backgroundColor: '#f9f9f9' }}>
                  <TableHead>
                    <TableRow>
                      <TableCell sx={{ backgroundColor: '#e0e0e0' }}></TableCell>
                      <TableCell align="center" sx={{ fontWeight: 'bold', backgroundColor: '#c8e6c9', fontSize: '1rem' }}>Single</TableCell>
                      <TableCell align="center" sx={{ fontWeight: 'bold', backgroundColor: '#bbdefb', fontSize: '1rem' }}>Mix</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {Array.from(new Set([...Object.keys(selected.single.finalResult || {})])).map((key) => {
                      const resultObjSingle = selected.single.finalResult as Record<string, any>;
                      const resultObjMix = selected.mix.finalResult as Record<string, any>;

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
                            {typeof resultObjMix?.[key] === 'number'
                              ? (Math.abs(resultObjMix[key]) > 1
                                  ? resultObjMix[key].toFixed(2)
                                  : (resultObjMix[key] * 100).toFixed(2) + (key.toLowerCase().includes('pct') || key.toLowerCase().includes('rate') || key.toLowerCase().includes('drawdown') || key.toLowerCase().includes('rendement') ? ' %' : ''))
                          : typeof resultObjMix?.[key] === 'boolean'
                            ? resultObjMix[key] ? 'Oui' : 'Non'
                            : resultObjMix?.[key] ?? '-'}
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </AccordionDetails>
            </Accordion>
          </div>
        )}
      </DialogContent>
    )}
  </Dialog>
);

export default BestPerformanceDialog;
