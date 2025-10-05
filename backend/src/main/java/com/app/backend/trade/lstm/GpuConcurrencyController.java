package com.app.backend.trade.lstm;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.concurrent.Semaphore;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * GpuConcurrencyController
 *
 * Rôle:
 *  - Régule le nombre d'entraînements DL4J concurrents accédant au GPU pour éviter la saturation VRAM.
 *  - Utilise une sémaphore bornée (min..max) et ajuste dynamiquement la capacité en fonction de la VRAM utilisée.
 *  - Lecture périodique via nvidia-smi (si disponible). Si indisponible -> aucun ajustement dynamique.
 *
 * Paramètres dynamiques (configurables via configure()):
 *  - minConcurrency / maxConcurrency (défaut 1..3)
 *  - scaleUpThresholdPct (si usage VRAM < threshold => tentative d'augmenter concurrence)
 *  - scaleDownThresholdPct (si usage VRAM > threshold => tentative de réduire concurrence)
 *
 * Thread-safety:
 *  - La sémaphore gère la limitation.
 *  - Les compteurs actifs utilisent AtomicInteger.
 */
public class GpuConcurrencyController {
    private static final Logger logger = LoggerFactory.getLogger(GpuConcurrencyController.class);

    private final Semaphore permits = new Semaphore(3, true); // capacité effective actuelle
    private volatile int targetMax = 3; // plafond courant
    private volatile int minConcurrency = 1;
    private volatile int maxConcurrency = 3;
    private volatile double scaleUpThresholdPct = 70.0;   // ancien <70% => scale up
    private volatile double scaleDownThresholdPct = 90.0; // ancien >90% => scale down
    private volatile boolean running = false;
    private ScheduledExecutorService scheduler;
    private ScheduledFuture<?> monitorFuture;
    private final AtomicInteger activeTrainings = new AtomicInteger(0);
    private volatile long lastVramLogTs = 0L;
    private volatile double lastVramUsagePct = -1.0; // dernière mesure VRAM (pour batch scaling dynamique)

    /**
     * Configure les bornes et seuils d'adaptation.
     * Peut être appelée avant ou après start(); thread-safe.
     */
    public synchronized void configure(int min, int max, double scaleUpPct, double scaleDownPct) {
        if (min < 1) min = 1;
        if (max < min) max = min;
        this.minConcurrency = min;
        this.maxConcurrency = max;
        this.scaleUpThresholdPct = Math.max(1.0, Math.min(99.0, scaleUpPct));
        this.scaleDownThresholdPct = Math.max(this.scaleUpThresholdPct + 0.5, Math.min(100.0, scaleDownPct));
        if (targetMax > maxConcurrency) {
            // réduire immédiatement les permits excédents
            int diff = targetMax - maxConcurrency;
            for (int i = 0; i < diff; i++) {
                if (permits.tryAcquire()) { /* retire un permit disponible */ }
            }
            targetMax = maxConcurrency;
        } else if (targetMax < minConcurrency) {
            int diff = minConcurrency - targetMax;
            permits.release(diff);
            targetMax = minConcurrency;
        }
        logger.info("[GPU][CONF] Concurrence GPU min={} max={} upIf<{}% downIf>{}% (targetCurrent={})", minConcurrency, maxConcurrency,
                String.format("%.1f", scaleUpThresholdPct), String.format("%.1f", scaleDownThresholdPct), targetMax);
    }

    /** Démarre le monitoring adaptatif. Sans effet si déjà lancé ou si CUDA indisponible. */
    public void start(boolean cudaBackend) {
        if (!cudaBackend) return;
        if (running) return;
        running = true;
        scheduler = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "gpu-vram-monitor");
            t.setDaemon(true);
            return t;
        });
        monitorFuture = scheduler.scheduleAtFixedRate(this::monitor, 5, 15, TimeUnit.SECONDS); // intervalle réduit 15s
        logger.info("[GPU][ADAPT] Monitoring VRAM démarré (intervalle 15s, plage {}..{})", minConcurrency, maxConcurrency);
    }

    /** Acquisition bloquante d'un permis. */
    public void acquirePermit() throws InterruptedException { permits.acquire(); }
    /** Restitution d'un permis. */
    public void releasePermit() { permits.release(); }
    /** Nombre d'entraînements actifs (utilisation indicative). */
    public int getActiveTrainings() { return activeTrainings.get(); }
    public void markTrainingStarted(){ activeTrainings.incrementAndGet(); }
    public void markTrainingFinished(){ activeTrainings.decrementAndGet(); }
    public double getLastVramUsagePct(){ return lastVramUsagePct; }

    /** Arrêt gracieux du monitoring (optionnel). */
    public void shutdown() {
        try {
            running = false;
            if (monitorFuture != null) monitorFuture.cancel(true);
            if (scheduler != null) scheduler.shutdownNow();
        } catch (Exception ignored) {}
    }

    private void monitor() {
        try {
            double usage = queryVramUsagePct();
            if (usage < 0) return; // échec lecture, ignorer
            lastVramUsagePct = usage;
            long now = System.currentTimeMillis();
            if (now - lastVramLogTs > 60000) { // log VRAM toutes les 60s max
                lastVramLogTs = now;
                logger.info("[GPU][VRAM] usage={}%, concurrency={} (permitsAvail={})", String.format("%.1f", usage), targetMax, permits.availablePermits());
            }
            if (usage > scaleDownThresholdPct && targetMax > minConcurrency) {
                adjust(targetMax - 1, usage, "DOWN");
            } else if (usage < scaleUpThresholdPct && targetMax < maxConcurrency) {
                adjust(targetMax + 1, usage, "UP");
            }
        } catch (Exception ignore) { }
    }

    private void adjust(int newMax, double usage, String dir) {
        if (newMax == targetMax) return;
        int old = targetMax;
        if (newMax > old) {
            permits.release(newMax - old); // augmenter capacité
        } else {
            int toRemove = old - newMax;
            int removed = 0;
            for (int i = 0; i < toRemove; i++) {
                if (permits.tryAcquire()) removed++; else break; // récupère permits libres pour réduire dispo
            }
        }
        targetMax = newMax;
        logger.info("[GPU][ADAPT][{}] VRAM={}%, concurrence {} -> {}", dir, String.format("%.1f", usage), old, newMax);
    }

    /** @return pourcentage VRAM utilisée ou -1 si indisponible/erreur. */
    private double queryVramUsagePct() {
        Process p = null;
        try {
            p = new ProcessBuilder("nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits").redirectErrorStream(true).start();
            try (BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
                String line = br.readLine();
                if (line == null || line.isEmpty()) return -1;
                String[] parts = line.split(",");
                if (parts.length < 2) return -1;
                double used = Double.parseDouble(parts[0].trim());
                double total = Double.parseDouble(parts[1].trim());
                if (total <= 0) return -1;
                return (used / total) * 100.0;
            }
        } catch (Exception e) {
            return -1;
        } finally {
            if (p != null) p.destroy();
        }
    }
}
