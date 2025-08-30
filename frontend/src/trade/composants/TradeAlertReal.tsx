import React from 'react';

interface TradeAlertRealProps {}

const TradeAlertReal: React.FC<TradeAlertRealProps> = () => (
  <div className="trade-alert-real">
    Attention : Vous êtes connecté à un <b>compte RÉEL</b> ! Toutes les opérations sont effectives sur le marché réel.
  </div>
);

export default TradeAlertReal;

