package com.app.backend.trade.lstm;

import java.io.*;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.*;

public class LstmFeatureMatrixCache {
    private static final String CACHE_DIR = "cache";

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
        if (!f.exists()) return null;
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(f)))) {
            int n = in.readInt();
            int m = in.readInt();
            double[][] matrix = new double[n][m];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < m; j++)
                    matrix[i][j] = in.readDouble();
            return matrix;
        } catch (Exception e) {
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
}

