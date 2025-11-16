package com.app.backend.trade.portfolio.advanced;

import com.google.gson.Gson;
import org.deeplearning4j.util.ModelSerializer;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;

import java.sql.ResultSet;
import java.sql.SQLException;

@Repository
public class PortfolioMultiHeadRepository {

    private final JdbcTemplate jdbcTemplate;
    private final Gson gson = new Gson();

    public PortfolioMultiHeadRepository(JdbcTemplate jdbcTemplate) { this.jdbcTemplate = jdbcTemplate; }

    public void save(String tri, String version, String notes, PortfolioMultiHeadModel model, PortfolioMultiHeadConfig cfg) {
        String sql = "INSERT INTO trade_ai.portfolio_multihead_models (tri, algo_version, input_size, hidden1, hidden2, lr, l2, dropout, feature_means, feature_stds, model_zip, notes, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?, now(), now())";
        byte[] zip;
        try {
            java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
            ModelSerializer.writeModel(model.getGraph(), baos, true);
            zip = baos.toByteArray();
        } catch (Exception e) { throw new RuntimeException("Sérialisation multi-head échouée: " + e.getMessage(), e); }
        String meansJson = model.getFeatureMeans() != null ? gson.toJson(model.getFeatureMeans()) : null;
        String stdsJson = model.getFeatureStds() != null ? gson.toJson(model.getFeatureStds()) : null;
        jdbcTemplate.update(con -> {
            var ps = con.prepareStatement(sql);
            ps.setString(1, tri);
            ps.setString(2, version);
            ps.setInt(3, model.getInputSize());
            ps.setInt(4, cfg.getHidden1());
            ps.setInt(5, cfg.getHidden2());
            ps.setDouble(6, cfg.getLearningRate());
            ps.setDouble(7, cfg.getL2());
            ps.setDouble(8, cfg.getDropout());
            ps.setString(9, meansJson);
            ps.setString(10, stdsJson);
            ps.setBytes(11, zip);
            ps.setString(12, notes);
            return ps;
        });
    }

    public LoadedMultiHead loadLatest(String tri, PortfolioMultiHeadConfig cfg) {
        String sql = "SELECT * FROM trade_ai.portfolio_multihead_models WHERE tri = ? ORDER BY created_at DESC LIMIT 1";
        return jdbcTemplate.query(sql, ps -> ps.setString(1, tri), rs -> rs.next() ? map(rs, cfg) : null);
    }

    private LoadedMultiHead map(ResultSet rs, PortfolioMultiHeadConfig cfg) throws SQLException {
        int inputSize = rs.getInt("input_size");
        int h1 = rs.getInt("hidden1");
        int h2 = rs.getInt("hidden2");
        double lr = rs.getDouble("lr");
        double l2 = rs.getDouble("l2");
        double dropout = rs.getDouble("dropout");
        String meansJson = rs.getString("feature_means");
        String stdsJson = rs.getString("feature_stds");
        double[] means = meansJson != null ? gson.fromJson(meansJson, double[].class) : null;
        double[] stds = stdsJson != null ? gson.fromJson(stdsJson, double[].class) : null;
        byte[] zip = rs.getBytes("model_zip");
        try {
            PortfolioMultiHeadModel model = PortfolioMultiHeadModel.loadFromBytes(zip, inputSize, h1, h2, lr, l2, dropout, means, stds);
            LoadedMultiHead out = new LoadedMultiHead();
            out.model = model;
            out.version = rs.getString("algo_version");
            out.createdAt = rs.getTimestamp("created_at").getTime();
            return out;
        } catch (Exception e) { throw new SQLException("Restauration multi-head échouée: " + e.getMessage(), e); }
    }

    public static class LoadedMultiHead {
        public PortfolioMultiHeadModel model;
        public String version;
        public long createdAt;
    }
}

