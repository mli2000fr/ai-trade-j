import React from 'react';
import { List, ListItem, ListItemButton, ListItemText, Box, Typography } from '@mui/material';
import { Link, useLocation } from 'react-router-dom';
import DashboardIcon from '@mui/icons-material/Dashboard';
import AutoGraphIcon from '@mui/icons-material/AutoGraph';
import SettingsIcon from '@mui/icons-material/Settings';
import MonitorHeartIcon from '@mui/icons-material/MonitorHeart';

const menuItems = [
  { label: 'Dashboard', path: '/trade/dashboard', icon: <DashboardIcon /> },
  { label: 'Stratégies', path: '/trade/strategies', icon: <AutoGraphIcon /> },
  { label: 'Monitoring Tuning', path: '/trade/tuning-monitor', icon: <MonitorHeartIcon /> },
  { label: 'Paramètres', path: '/trade/settings', icon: <SettingsIcon /> },
];

const Sidebar: React.FC = () => {
  const location = useLocation();
  return (
    <Box sx={{
      width: 240,
      bgcolor: 'linear-gradient(180deg, #f8fafc 0%, #e3eafc 100%)', // fond clair dégradé
      color: '#223',
      height: '100vh',
      boxShadow: 3,
      borderTopRightRadius: 16,
      borderBottomRightRadius: 16,
      display: 'flex',
      flexDirection: 'column',
      p: 0
    }}>
      <Box sx={{ p: 3, pb: 2, display: 'flex', alignItems: 'center', justifyContent: 'center', borderBottom: '1px solid #dde3ee' }}>
        <Typography variant="h6" sx={{ fontWeight: 700, letterSpacing: 1, color: '#223' }}>
          AI TRADE
        </Typography>
      </Box>
      <List sx={{ flex: 1, p: 0 }}>
        {menuItems.map(item => (
          <ListItem key={item.path} disablePadding>
            <ListItemButton
              component={Link}
              to={item.path}
              selected={location.pathname === item.path}
              sx={{
                color: '#223',
                borderRadius: 2,
                mx: 1,
                my: 0.5,
                background: 'none',
                '& .MuiListItemIcon-root, & svg': {
                  color: '#3b5998',
                },
                '& .MuiListItemText-root': {
                  color: '#223',
                },
                '&.Mui-selected': {
                  bgcolor: 'rgba(59,89,152,0.10)',
                  color: '#3b5998',
                  fontWeight: 700,
                },
                '&:hover': {
                  bgcolor: 'rgba(59,89,152,0.07)',
                  color: '#3b5998',
                },
                transition: 'background 0.2s',
                fontWeight: 500,
                fontSize: '1.08rem',
                minHeight: 48
              }}
            >
              <Box sx={{ minWidth: 32, display: 'flex', alignItems: 'center', justifyContent: 'center', mr: 1, color: '#3b5998' }}>
                {item.icon}
              </Box>
              <ListItemText primary={item.label} sx={{ color: '#223' }} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
    </Box>
  );
};

export default Sidebar;
