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
 *  - Utilise une sémaphore bornée (1..3) et ajuste dynamiquement la capacité en fonction de la VRAM utilisée.
 *  - Lecture périodique via nvidia-smi (si disponible). Si indisponible -> aucun ajustement dynamique.
 *
 * Politique d'adaptation:
 *  - VRAM > 90% : décrémenter la capacité (jusqu'à 1)
 *  - VRAM < 70% : incrémenter la capacité (jusqu'à 3)
 *
 * Thread-safety:
 *  - La sémaphore gère la limitation.
 *  - Les compteurs actifs utilisent AtomicInteger.
 */
public class GpuConcurrencyController {
    private static final Logger logger = LoggerFactory.getLogger(GpuConcurrencyController.class);

    private final Semaphore permits = new Semaphore(3, true); // max initial 3
    private volatile int targetMax = 3; // 1..3
    private volatile boolean running = false;
    private ScheduledExecutorService scheduler;
    private ScheduledFuture<?> monitorFuture;
    private final AtomicInteger activeTrainings = new AtomicInteger(0);

    /**
     * Démarre le monitoring adaptatif. Sans effet si déjà lancé ou si CUDA indisponible.
     * @param cudaBackend true si backend CUDA actif
     */
    public void start(boolean cudaBackend) {
        if (!cudaBackend) return;
        if (running) return;
        running = true;
        scheduler = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "gpu-vram-monitor");
            t.setDaemon(true);
            return t;
        });
        monitorFuture = scheduler.scheduleAtFixedRate(this::monitor, 5, 20, TimeUnit.SECONDS);
        logger.info("[GPU][ADAPT] Monitoring VRAM démarré (intervalle 20s, plage concurrence 1..3)");
    }

    /**
     * Acquisition bloquante d'un permis.
     * @throws InterruptedException interruption thread
     */
    public void acquirePermit() throws InterruptedException { permits.acquire(); }

    /** Restitution d'un permis. */
    public void releasePermit() { permits.release(); }

    /** Nombre d'entraînements actifs (utilisation indicative). */
    public int getActiveTrainings() { return activeTrainings.get(); }

    /** Marque le début logique d'un entraînement (statistiques internes). */
    public void markTrainingStarted(){ activeTrainings.incrementAndGet(); }

    /** Marque la fin logique d'un entraînement (statistiques internes). */
    public void markTrainingFinished(){ activeTrainings.decrementAndGet(); }

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
            if (usage > 90.0 && targetMax > 1) {
                adjust(targetMax - 1, usage);
            } else if (usage < 70.0 && targetMax < 3) {
                adjust(targetMax + 1, usage);
            }
        } catch (Exception ignore) {
        }
    }

    private void adjust(int newMax, double usage) {
        if (newMax == targetMax) return;
        int old = targetMax;
        if (newMax > old) {
            permits.release(newMax - old); // augmenter capacité
        } else {
            int toRemove = old - newMax;
            int removed = 0;
            for (int i = 0; i < toRemove; i++) {
                if (permits.tryAcquire()) removed++; else break;
            }
            // Si removed < toRemove, les permits restants seront relâchés naturellement plus tard
        }
        targetMax = newMax;
        logger.info("[GPU][ADAPT] VRAM usage={}%, concurrence {} -> {}", String.format("%.1f", usage), old, newMax);
    }

    /**
     * @return pourcentage VRAM utilisée ou -1 si indisponible/erreur.
     */
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

