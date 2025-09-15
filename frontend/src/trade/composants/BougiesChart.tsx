import React from 'react';
import Typography from '@mui/material/Typography';
import Alert from '@mui/material/Alert';
import ReactApexChart from 'react-apexcharts';
import type { ApexOptions } from 'apexcharts';

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

export default BougiesChart;

