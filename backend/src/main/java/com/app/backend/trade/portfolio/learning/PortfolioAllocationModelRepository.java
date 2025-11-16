package com.app.backend.trade.portfolio.learning;

import com.google.gson.Gson;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;

import java.sql.ResultSet;
import java.sql.SQLException;

@Repository
public class PortfolioAllocationModelRepository {

    private final JdbcTemplate jdbcTemplate;
    private final Gson gson = new Gson();

    public PortfolioAllocationModelRepository(JdbcTemplate jdbcTemplate) {
        this.jdbcTemplate = jdbcTemplate;
    }

    public void saveModel(String tri, String algoVersion, String notes, PortfolioAllocationModel model) {
        String sql = "INSERT INTO trade_ai.portfolio_allocation_models (tri, algo_version, input_size, feature_means, feature_stds, model_zip, learning_rate, l2, hidden1, hidden2, epochs, notes, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?, now(), now())";
        byte[] bytes;
        try {
            bytes = modelToBytes(model);
        } catch (Exception e) {
            throw new RuntimeException("Erreur sérialisation modèle allocation: " + e.getMessage(), e);
        }
        String meansJson = model.getFeatureMeans() != null ? gson.toJson(model.getFeatureMeans()) : null;
        String stdsJson = model.getFeatureStds() != null ? gson.toJson(model.getFeatureStds()) : null;
        jdbcTemplate.update(connection -> {
            var ps = connection.prepareStatement(sql);
            ps.setString(1, tri);
            ps.setString(2, algoVersion);
            ps.setInt(3, model.getInputSize());
            ps.setString(4, meansJson);
            ps.setString(5, stdsJson);
            ps.setBytes(6, bytes);
            ps.setDouble(7, model.getConfig().getLearningRate());
            ps.setDouble(8, model.getConfig().getL2());
            ps.setInt(9, model.getConfig().getHidden1());
            ps.setInt(10, model.getConfig().getHidden2());
            ps.setInt(11, model.getConfig().getEpochs());
            ps.setString(12, notes);
            return ps;
        });
    }

    public LoadedAllocationModel loadLatestModel(String tri, PortfolioLearningConfig config) {
        String sql = "SELECT id, algo_version, input_size, feature_means, feature_stds, model_zip, created_at, updated_at FROM trade_ai.portfolio_allocation_models WHERE tri = ? ORDER BY created_at DESC LIMIT 1";
        return jdbcTemplate.query(sql, ps -> ps.setString(1, tri), rs -> rs.next() ? mapRow(rs, config) : null);
    }

    private LoadedAllocationModel mapRow(ResultSet rs, PortfolioLearningConfig config) throws SQLException {
        int inputSize = rs.getInt("input_size");
        String meansJson = rs.getString("feature_means");
        String stdsJson = rs.getString("feature_stds");
        double[] means = meansJson != null ? gson.fromJson(meansJson, double[].class) : null;
        double[] stds = stdsJson != null ? gson.fromJson(stdsJson, double[].class) : null;
        byte[] bytes = rs.getBytes("model_zip");
        PortfolioAllocationModel model;
        try {
            model = PortfolioAllocationModel.loadFromBytes(bytes, config, inputSize, means, stds);
        } catch (Exception e) {
            throw new SQLException("Impossible de restaurer le modèle d'allocation: " + e.getMessage(), e);
        }
        LoadedAllocationModel out = new LoadedAllocationModel();
        out.model = model;
        out.createdAtEpochMs = rs.getTimestamp("created_at").getTime();
        out.updatedAtEpochMs = rs.getTimestamp("updated_at").getTime();
        out.algoVersion = rs.getString("algo_version");
        out.id = rs.getLong("id");
        return out;
    }

    private byte[] modelToBytes(PortfolioAllocationModel model) throws java.io.IOException {
        java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
        org.deeplearning4j.util.ModelSerializer.writeModel(model.getNetwork(), baos, true);
        return baos.toByteArray();
    }

    public static class LoadedAllocationModel {
        public long id;
        public String algoVersion;
        public long createdAtEpochMs;
        public long updatedAtEpochMs;
        public PortfolioAllocationModel model;
        public java.time.LocalDate createdAtDate() { return java.time.Instant.ofEpochMilli(createdAtEpochMs).atZone(java.time.ZoneId.systemDefault()).toLocalDate(); }
    }
}
