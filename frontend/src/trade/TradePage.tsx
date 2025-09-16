import React from 'react';
import './TradePage.css';
import Box from '@mui/material/Box';
import Sidebar from './composants/Sidebar';
import { Routes, Route, Navigate } from 'react-router-dom';
import DashboardPage from './routes/DashboardPage';
import StrategiesPage from './routes/StrategiesPage';
import SettingsPage from './routes/SettingsPage';
import TuningMonitorPage from './routes/TuningMonitorPage';
import MixStrategiesMonitorPage from './routes/MixStrategiesMonitorPage';
import { SelectedCompteProvider } from './SelectedCompteContext';

const TradePage: React.FC = () => {
  return (
    <SelectedCompteProvider>
      <Box sx={{ display: 'flex', height: '100vh' }}>
        <Sidebar />
        <Box sx={{ flex: 1, p: 3, overflow: 'auto' }}>
          <Routes>
            <Route path="dashboard" element={<DashboardPage />} />
            <Route path="strategies" element={<StrategiesPage />} />
            <Route path="settings" element={<SettingsPage />} />
            <Route path="tuning-monitor" element={<TuningMonitorPage />} />
            <Route path="mix-strategies-monitor" element={<MixStrategiesMonitorPage />} />
            <Route path="*" element={<Navigate to="dashboard" replace />} />
          </Routes>
        </Box>
      </Box>
    </SelectedCompteProvider>
  );
};

export default TradePage;
