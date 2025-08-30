import React from 'react';

interface Compte {
  id: number;
  nom: string;
  alias?: string;
  real?: boolean;
}

interface TradeCompteSelectProps {
  comptes: Compte[];
  selectedCompteId: number | null;
  loading: boolean;
  error: string | null;
  onSelect: (id: number) => void;
}

const TradeCompteSelect: React.FC<TradeCompteSelectProps> = ({ comptes, selectedCompteId, loading, error, onSelect }) => (
  <div className="trade-comptes-select">
    <label className="trade-comptes-label">Compte&nbsp;:</label>
    {loading ? (
      <span>Chargement des comptes...</span>
    ) : error ? (
      <span className="trade-comptes-error">{error}</span>
    ) : (
      <select
        value={selectedCompteId ?? ''}
        onChange={e => onSelect(Number(e.target.value))}
        className="trade-comptes-dropdown"
      >
        {comptes.map(compte => (
          <option key={compte.id} value={compte.id}>
            {compte.nom}
            {compte.alias ? ` (${compte.alias})` : ''}
            {compte.real === true ? ' [REAL]' : compte.real === false ? ' [PAPER]' : ''}
          </option>
        ))}
      </select>
    )}
  </div>
);

export default TradeCompteSelect;

