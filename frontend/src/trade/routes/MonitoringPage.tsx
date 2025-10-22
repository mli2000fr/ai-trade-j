import React, { useState } from 'react';
import { Box, Tabs, Tab } from '@mui/material';
import CroisedStrategiesMonitorPage from './CroisedStrategiesMonitorPage';
import MixStrategiesMonitorPage from './MixStrategiesMonitorPage';
import UpdateDailyValueMonitorPage from './UpdateDailyValueMonitorPage';
import SymbolBuyMonitorPage from './SymbolBuyMonitorPage';
import TuningMonitorPage from './TuningMonitorPage';

const MonitoringPage: React.FC = () => {
  const [tab, setTab] = useState(0);

  return (
    <Box sx={{ p: 4 }}>
      <Tabs value={tab} onChange={(_, v) => setTab(v)} centered>
        <Tab label="Single Strategies" />
        <Tab label="Mix Strategies" />
        <Tab label="Tuning LSTM" />
        <Tab label="Value Daily" />
        <Tab label="Symbols buy" />
      </Tabs>
      <Box sx={{ mt: 4 }}>
        {tab === 0 && <CroisedStrategiesMonitorPage />}
        {tab === 1 && <MixStrategiesMonitorPage />}
        {tab === 2 && <TuningMonitorPage />}
        {tab === 3 && <UpdateDailyValueMonitorPage />}
        {tab === 4 && <SymbolBuyMonitorPage />}
      </Box>
    </Box>
  );
};

export default MonitoringPage;
export {};
