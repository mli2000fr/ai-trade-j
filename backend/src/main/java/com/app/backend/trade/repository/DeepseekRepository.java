package com.app.backend.trade.repository;

import com.app.backend.trade.model.AgentEntity;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface DeepseekRepository extends JpaRepository<AgentEntity, Long> {
}

