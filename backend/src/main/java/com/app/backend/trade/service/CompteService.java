package com.app.backend.trade.service;

import com.app.backend.trade.model.CompteDto;
import com.app.backend.trade.model.CompteEntity;
import com.app.backend.trade.repository.CompteRepository;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

@Service
public class CompteService {
    private final CompteRepository compteRepository;

    public CompteService(CompteRepository compteRepository) {
        this.compteRepository = compteRepository;
    }

    public List<CompteEntity> getAllComptes() {
        return compteRepository.findAll();
    }

    public List<CompteDto> getAllComptesDto() {
        return compteRepository.findAll().stream()
                .map(compte -> new CompteDto(compte.getId(), compte.getNom(), compte.getAlias(), compte.getReal()))
                .collect(Collectors.toList());
    }

    public CompteEntity getCompteCredentialsById(String id) {
        return compteRepository.findById(Integer.parseInt(id)).orElse(null);
    }
}
