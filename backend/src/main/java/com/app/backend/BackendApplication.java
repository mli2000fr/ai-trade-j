package com.app.backend;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cache.annotation.EnableCaching;
import jakarta.annotation.PostConstruct;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.buffer.DataType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@SpringBootApplication
@EnableCaching
public class BackendApplication {
    private static final Logger log = LoggerFactory.getLogger(BackendApplication.class);

    public static void main(String[] args) {
        SpringApplication.run(BackendApplication.class, args);
    }

    @PostConstruct
    void logNd4jBackend() {
        try {
            String backend = Nd4j.getBackend().getClass().getSimpleName();
            DataType fp = Nd4j.defaultFloatingPointType();
            log.info("[BOOT] ND4J backend={} defaultFPType={}", backend, fp);
            if(fp != DataType.FLOAT){
                log.warn("[BOOT][WARN] defaultFloatingPointType != FLOAT ({}). Vérifier setDefaultDataTypes.", fp);
            }
        } catch (NoClassDefFoundError | ExceptionInInitializerError t) {
            log.error("[BOOT][ND4J] Impossible de récupérer backend/dtype: {}", t.getMessage());
        } catch (Throwable t) {
            log.error("[BOOT][ND4J] Impossible de récupérer backend/dtype: {}", t.getMessage());
        }
    }
}
