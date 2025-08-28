package com.app.backend.trade.repository;

import com.app.backend.trade.model.GptEntity;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface GptRepository extends JpaRepository<GptEntity, Long> {
}

