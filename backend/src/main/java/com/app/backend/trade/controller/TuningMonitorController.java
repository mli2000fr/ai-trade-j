package com.app.backend.trade.controller;

import com.app.backend.trade.lstm.LstmTuningService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.http.ResponseEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
import org.springframework.jdbc.core.JdbcTemplate;
import java.util.List;
import java.util.Map;
import java.io.IOException;

@RestController
@RequestMapping("/api/tuning")
public class TuningMonitorController {
    @Autowired
    private JdbcTemplate jdbcTemplate;
    @Autowired
    private LstmTuningService lstmTuningService;

    // Endpoint JSON pour le monitoring
    @GetMapping("/metrics")
    public List<Map<String, Object>> getTuningMetrics(@RequestParam(required = false) String symbol) {
        String sql = symbol == null ?
            "SELECT symbol, mse, rmse, direction, tested_date FROM lstm_tuning_metrics ORDER BY tested_date DESC LIMIT 100" :
            "SELECT symbol, mse, rmse, direction, tested_date FROM lstm_tuning_metrics WHERE symbol = ? ORDER BY tested_date DESC LIMIT 100";
        return symbol == null ?
            jdbcTemplate.queryForList(sql) :
            jdbcTemplate.queryForList(sql, symbol);
    }

    // Endpoint CSV pour l'export
    @GetMapping(value = "/metrics/csv", produces = MediaType.TEXT_PLAIN_VALUE)
    public ResponseEntity<ByteArrayResource> exportTuningMetricsCsv(@RequestParam(required = false) String symbol) throws IOException {
        String outputPath = "tuning_metrics_export.csv";
        String filePath = lstmTuningService.hyperparamsRepository.exportTuningMetricsToCsv(symbol, outputPath);
        if (filePath != null) {
            java.nio.file.Path path = java.nio.file.Paths.get(filePath);
            byte[] data = java.nio.file.Files.readAllBytes(path);
            ByteArrayResource resource = new ByteArrayResource(data);
            HttpHeaders headers = new HttpHeaders();
            headers.add(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=metrics.csv");
            headers.add(HttpHeaders.CONTENT_TYPE, "text/csv");
            return ResponseEntity.ok()
                    .headers(headers)
                    .contentLength(data.length)
                    .body(resource);
        } else {
            return ResponseEntity.notFound().build();
        }
    }

    // Endpoint JSON pour la progression en temps réel
    @GetMapping("/progress")
    public List<Map<String, Object>> getTuningProgress(@RequestParam(required = false) String symbol) {
        List<Map<String, Object>> result = new java.util.ArrayList<>();
        java.util.Map<String, LstmTuningService.TuningProgress> map = lstmTuningService.getTuningProgressMap();
        for (LstmTuningService.TuningProgress progress : map.values()) {
            if (symbol == null || symbol.equals(progress.symbol)) {
                java.util.Map<String, Object> row = new java.util.HashMap<>();
                row.put("symbol", progress.symbol);
                row.put("totalConfigs", progress.totalConfigs);
                row.put("testedConfigs", progress.testedConfigs.get());
                row.put("status", progress.status);
                row.put("startTime", progress.startTime);
                row.put("endTime", progress.endTime);
                row.put("lastUpdate", progress.lastUpdate);
                result.add(row);
            }
        }
        return result;
    }
}
