import React from 'react';
import StrategyManagerBlock from '../composants/StrategyManagerBlock';
import BestPerformanceSymbolBlock from '../composants/BestPerformanceSymbolBlock';

const StrategiesPage: React.FC = () => {
  return (
    <div>
      <StrategyManagerBlock />
      <BestPerformanceSymbolBlock />
    </div>
  );
};

export default StrategiesPage;
