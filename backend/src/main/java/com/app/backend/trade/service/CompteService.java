package com.app.backend.trade.service;

import com.app.backend.trade.model.CompteDto;
import com.app.backend.trade.model.CompteEntity;
import com.app.backend.trade.repository.CompteRepository;
import org.springframework.stereotype.Service;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.cache.annotation.CacheEvict;

import java.util.List;
import java.util.NoSuchElementException;
import java.util.stream.Collectors;

@Service
public class CompteService {
    private final CompteRepository compteRepository;

    public CompteService(CompteRepository compteRepository) {
        this.compteRepository = compteRepository;
    }

    /**
     * Récupère tous les comptes (entités complètes).
     * Résultat mis en cache pour la session.
     */
    @Cacheable(value = "comptes", unless = "#result == null || #result.isEmpty()")
    public List<CompteEntity> getAllComptes() {
        return compteRepository.findAll();
    }

    /**
     * Invalide le cache des comptes (à appeler après modification des comptes).
     */
    @CacheEvict(value = "comptes", allEntries = true)
    public void evictComptesCache() {}

    /**
     * Récupère tous les comptes sous forme de DTO (pour l'affichage côté client).
     * Résultat mis en cache pour la session.
     */
    @Cacheable(value = "comptesDto", unless = "#result == null || #result.isEmpty()")
    public List<CompteDto> getAllComptesDto() {
        return compteRepository.findAll().stream()
                .map(compte -> new CompteDto(compte.getId(), compte.getNom(), compte.getAlias(), compte.getReal()))
                .collect(Collectors.toList());
    }

    /**
     * Invalide le cache des comptes DTO (à appeler après modification des comptes).
     */
    @CacheEvict(value = "comptesDto", allEntries = true)
    public void evictComptesDtoCache() {}

    /**
     * Récupère les credentials d'un compte par son ID.
     * Résultat mis en cache pour la session (clé = id).
     *
     * @param id identifiant du compte
     * @return CompteEntity correspondant
     * @throws NoSuchElementException si le compte n'existe pas
     */
    @Cacheable(value = "compteById", key = "#id", unless = "#result == null")
    public CompteEntity getCompteCredentialsById(String id) {
        return compteRepository.findById(Integer.parseInt(id))
                .orElseThrow(() -> new NoSuchElementException("Aucun compte trouvé pour l'ID : " + id));
    }

    /**
     * Invalide le cache des comptes par ID (à appeler après modification des comptes).
     */
    @CacheEvict(value = "compteById", allEntries = true)
    public void evictCompteByIdCache() {}
}
