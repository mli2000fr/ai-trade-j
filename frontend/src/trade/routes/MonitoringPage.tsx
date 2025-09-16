import React, { useState } from 'react';
import { Box, Tabs, Tab } from '@mui/material';
import MixStrategiesMonitorPage from './MixStrategiesMonitorPage';
import TuningMonitorPage from './TuningMonitorPage';

const MonitoringPage: React.FC = () => {
  const [tab, setTab] = useState(0);

  return (
    <Box sx={{ p: 4 }}>
      <Tabs value={tab} onChange={(_, v) => setTab(v)} centered>
        <Tab label="Mix Strategies" />
        <Tab label="Tuning LSTM" />
      </Tabs>
      <Box sx={{ mt: 4 }}>
        {tab === 0 && <MixStrategiesMonitorPage />}
        {tab === 1 && <TuningMonitorPage />}
      </Box>
    </Box>
  );
};

export default MonitoringPage;
export {};
