import React from 'react';

interface OrdersTableProps {
  orders: any[];
  loading: boolean;
  filterSymbol: string;
  filterCancelable: boolean;
  onFilterSymbol: (symbol: string) => void;
  onFilterCancelable: (checked: boolean) => void;
  onUpdate: () => void;
  onCancel: (orderId: string) => void;
  cancellingOrderId: string | null;
  cancellableStatuses: string[];
  positions: any[];
}

const OrdersTable: React.FC<OrdersTableProps> = ({
  orders,
  loading,
  filterSymbol,
  filterCancelable,
  onFilterSymbol,
  onFilterCancelable,
  onUpdate,
  onCancel,
  cancellingOrderId,
  cancellableStatuses,
  positions
}) => {
  const hasCancellable = orders.some(order => order.id && cancellableStatuses.includes(order.status));
  return (
    <div className="trade-orders-recent">
      <b>Ordres récents&nbsp;:</b>
      <div className="trade-orders-filter-bar">
        <label className="trade-orders-filter-label">
          Filtrer par symbole&nbsp;
          <select value={filterSymbol} onChange={e => onFilterSymbol(e.target.value)}>
            <option value="">Tous</option>
            {positions
              .map(pos => pos.symbol)
              .filter((symbol, idx, arr) => arr.indexOf(symbol) === idx)
              .map(symbol => (
                <option key={symbol} value={symbol}>{symbol}</option>
              ))}
          </select>
        </label>
        <label className="trade-orders-filter-checkbox">
          <input type="checkbox" checked={filterCancelable} onChange={e => onFilterCancelable(e.target.checked)} />
          Annulables uniquement
        </label>
        <button onClick={onUpdate} disabled={loading} className="trade-orders-update-btn">
          {loading ? 'Mise à jour...' : 'Update'}
        </button>
      </div>
      {orders.length === 0 && !loading ? (
        <div>Aucun ordre à afficher.</div>
      ) : loading ? (
        <div>Chargement des ordres...</div>
      ) : (
        <table className="orders-table">
          <thead>
            <tr>
              <th>Action</th>
              <th>Symbole</th>
              <th>Quantité</th>
              <th>Prix</th>
              <th>Statut</th>
              {hasCancellable && <th></th>}
            </tr>
          </thead>
          <tbody>
            {orders.map((order, i) => (
              <tr key={i} className={order.side === 'buy' ? 'buy-row' : order.side === 'sell' ? 'sell-row' : ''}>
                <td className="status">{order.side}</td>
                <td>{order.symbol}</td>
                <td>{order.qty}</td>
                <td>{order.filledAvgPrice !== undefined && order.filledAvgPrice !== null ? Number(order.filledAvgPrice).toLocaleString('fr-FR', {minimumFractionDigits: 2, maximumFractionDigits: 2}) + ' $' : (order.limit_price !== undefined && order.limit_price !== null ? Number(order.limit_price).toLocaleString('fr-FR', {minimumFractionDigits: 2, maximumFractionDigits: 2}) + ' $' : '-')}</td>
                <td className="status">{order.status}</td>
                {hasCancellable && (
                  <td>
                    {order.id && cancellableStatuses.includes(order.status) && (
                      <button
                        className="trade-cancel-btn"
                        disabled={cancellingOrderId === order.id}
                        onClick={() => onCancel(order.id)}
                      >
                        {cancellingOrderId === order.id ? 'Annulation...' : 'Annuler'}
                      </button>
                    )}
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};

export default OrdersTable;

