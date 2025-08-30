package com.app.backend.trade.repository;

import com.app.backend.trade.model.CompteEntity;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface CompteRepository extends JpaRepository<CompteEntity, Integer> {
}

