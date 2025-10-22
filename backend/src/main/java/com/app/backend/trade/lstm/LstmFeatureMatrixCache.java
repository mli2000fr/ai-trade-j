package com.app.backend.trade.lstm;

import java.io.*;
import java.security.MessageDigest;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;

public class LstmFeatureMatrixCache {
    private static final String CACHE_DIR = "cache";
    // Compteurs globaux hits/misses pour tests
    private static final AtomicLong hits = new AtomicLong();
    private static final AtomicLong misses = new AtomicLong();

    static {
        File dir = new File(CACHE_DIR);
        if (!dir.exists()) dir.mkdirs();
    }

    public static String computeKey(String symbol, String interval, int barCount, long lastBarEndTime, String featureSetVersion, List<String> features) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            String base = symbol + "|" + interval + "|" + barCount + "|" + lastBarEndTime + "|" + featureSetVersion + "|" + String.join(",", features);
            byte[] hash = md.digest(base.getBytes("UTF-8"));
            StringBuilder sb = new StringBuilder();
            for (byte b : hash) sb.append(String.format("%02x", b));
            return sb.toString();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static double[][] load(String key) {
        File f = new File(CACHE_DIR, key + ".bin");
        if (!f.exists()) {
            misses.incrementAndGet();
            return null;
        }
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(f)))) {
            int n = in.readInt();
            int m = in.readInt();
            double[][] matrix = new double[n][m];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < m; j++)
                    matrix[i][j] = in.readDouble();
            hits.incrementAndGet();
            return matrix;
        } catch (Exception e) {
            // corruption => considérer comme miss (on supprime le fichier pour éviter répétition)
            misses.incrementAndGet();
            try { f.delete(); } catch (Exception ignore) {}
            return null;
        }
    }

    public static void save(String key, double[][] matrix) {
        File f = new File(CACHE_DIR, key + ".bin");
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(f)))) {
            int n = matrix.length;
            int m = n > 0 ? matrix[0].length : 0;
            out.writeInt(n);
            out.writeInt(m);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < m; j++)
                    out.writeDouble(matrix[i][j]);
        } catch (Exception e) {
            // ignore
        }
    }

    // --- Instrumentation API ---
    public static void resetStats() { hits.set(0); misses.set(0); }
    public static CacheStats getStats() { return new CacheStats(hits.get(), misses.get()); }

    public record CacheStats(long hits, long misses) {
        public long total() { return hits + misses; }
        public double hitRatio() { long t = total(); return t==0? 0.0 : (double) hits / t; }
        public String toString(){ return "CacheStats{hits="+hits+", misses="+misses+", hitRatio="+String.format(Locale.US, "%.2f", hitRatio()*100)+"%}"; }
    }
}
