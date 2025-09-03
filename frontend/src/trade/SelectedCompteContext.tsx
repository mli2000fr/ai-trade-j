import React, { createContext, useContext, useEffect, useState } from 'react';

export interface Compte {
  id: number;
  nom: string;
  alias?: string;
  real?: boolean;
}

interface SelectedCompteContextType {
  comptes: Compte[];
  selectedCompteId: number | null;
  setSelectedCompteId: (id: number) => void;
  comptesLoading: boolean;
  comptesError: string | null;
  refreshComptes: () => void;
}

const SelectedCompteContext = createContext<SelectedCompteContextType | undefined>(undefined);

export const SelectedCompteProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [comptes, setComptes] = useState<Compte[]>([]);
  const [selectedCompteId, setSelectedCompteId] = useState<number | null>(null);
  const [comptesLoading, setComptesLoading] = useState(true);
  const [comptesError, setComptesError] = useState<string | null>(null);

  const fetchComptes = async () => {
    setComptesLoading(true);
    setComptesError(null);
    try {
      const res = await fetch('/api/trade/comptes');
      if (!res.ok) throw new Error('Erreur lors du chargement des comptes');
      const data = await res.json();
      setComptes(Array.isArray(data) ? data : []);
      if (Array.isArray(data) && data.length > 0) setSelectedCompteId(data[0].id);
    } catch (e) {
      setComptesError('Impossible de charger les comptes');
      setComptes([]);
    } finally {
      setComptesLoading(false);
    }
  };

  useEffect(() => {
    fetchComptes();
  }, []);

  return (
    <SelectedCompteContext.Provider value={{ comptes, selectedCompteId, setSelectedCompteId, comptesLoading, comptesError, refreshComptes: fetchComptes }}>
      {children}
    </SelectedCompteContext.Provider>
  );
};

export const useSelectedCompte = () => {
  const ctx = useContext(SelectedCompteContext);
  if (!ctx) throw new Error('useSelectedCompte doit être utilisé dans un SelectedCompteProvider');
  return ctx;
};

