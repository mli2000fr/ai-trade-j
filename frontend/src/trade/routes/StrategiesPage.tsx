import React from 'react';
import StrategyManagerBlock from '../composants/StrategyManagerBlock';
import TestSignalForm from '../composants/TestSignalForm';

const StrategiesPage: React.FC = () => {
  return (
    <div>
      <StrategyManagerBlock />
      <TestSignalForm />
    </div>
  );
};

export default StrategiesPage;
