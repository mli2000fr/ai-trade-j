package com.app.backend.trade.repository;

import com.app.backend.trade.model.OrderEntity;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

/**
 * Repository Spring Data JPA pour la gestion des entités OrderEntity (ordres de trading).
 */
@Repository
public interface OrderRepository extends JpaRepository<OrderEntity, Integer> {
}
