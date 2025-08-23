package com.example.backend;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class SearchController {
    @Autowired
    private GoogleCustomSearchService googleCustomSearchService;

    @GetMapping("/api/search")
    public String search(@RequestParam String query) {
        return googleCustomSearchService.search(query);
    }
}

