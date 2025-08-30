import React from 'react';

interface PortfolioBlockProps {
  portfolio: any;
  lastUpdate: Date | null;
  loading: boolean;
}

const PortfolioBlock: React.FC<PortfolioBlockProps> = ({ portfolio, lastUpdate, loading }) => (
  <div className="portfolio-block">
    <h3>Mon portefeuille</h3>
    {lastUpdate && (
      <div className="portfolio-last-update">
        Actualisé à {lastUpdate.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
      </div>
    )}
    {loading && <div>Chargement...</div>}
    {!loading && portfolio && (
      <>
        {portfolio.account && (
          <div className="portfolio-account portfolio-account-cards">
            <div className="account-card">
              <div className="account-label">Valeur totale</div>
              <div className="account-value">{Number(portfolio.account.equity).toLocaleString('fr-FR', {minimumFractionDigits: 2, maximumFractionDigits: 2})} $</div>
            </div>
            <div className="account-card">
              <div className="account-label">Buying Power</div>
              <div className="account-value">{Number(portfolio.account.buying_power).toLocaleString('fr-FR', {minimumFractionDigits: 2, maximumFractionDigits: 2})} $</div>
            </div>
            <div className="account-card">
              <div className="account-label">Cash</div>
              <div className="account-value">{Number(portfolio.account.cash).toLocaleString('fr-FR', {minimumFractionDigits: 2, maximumFractionDigits: 2})} $</div>
            </div>
            {portfolio.account.portfolio_value && (
              <div className="account-card">
                <div className="account-label">Portfolio Value</div>
                <div className="account-value">{Number(portfolio.account.portfolio_value).toLocaleString('fr-FR', {minimumFractionDigits: 2, maximumFractionDigits: 2})} $</div>
              </div>
            )}
            {portfolio.account.status && (
              <div className="account-card">
                <div className="account-label">Status</div>
                <div className="account-value status-value">{portfolio.account.status}</div>
              </div>
            )}
          </div>
        )}
        <div>
          <b>Positions&nbsp;:</b>
          {portfolio.positions.length === 0 ? (
            <div>Aucune position en cours.</div>
          ) : (
            <table className="orders-table portfolio-positions-table">
              <thead>
                <tr>
                  <th>Symbole</th>
                  <th>Prix d'achat</th>
                  <th>Prix actuel</th>
                  <th>Quantité</th>
                  <th>Total</th>
                  <th>P & L pc</th>
                  <th>P & L</th>
                </tr>
              </thead>
              <tbody>
                {portfolio.positions.map((pos: any, i: number) => (
                  <tr key={i}>
                    <td>{pos.symbol}</td>
                    <td>{pos.avg_entry_price !== undefined && pos.avg_entry_price !== null ? Number(pos.avg_entry_price).toFixed(2) + ' $' : '-'}</td>
                    <td>{pos.current_price !== undefined && pos.current_price !== null ? Number(pos.current_price).toFixed(2) + ' $' : '-'}</td>
                    <td>{pos.qty}</td>
                    <td>{pos.current_price !== undefined && pos.current_price !== null ? (Number(pos.qty) * Number(pos.current_price)).toFixed(2) + ' $' : '-'}</td>
                    <td>{pos.unrealized_plpc !== undefined && pos.unrealized_plpc !== null ? (Number(pos.unrealized_plpc) * 100).toFixed(3) + ' %' : '-'}</td>
                    <td>{pos.unrealized_pl !== undefined && pos.unrealized_pl !== null ? Number(pos.unrealized_pl).toFixed(2) + ' $' : '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </>
    )}
  </div>
);

export default PortfolioBlock;

