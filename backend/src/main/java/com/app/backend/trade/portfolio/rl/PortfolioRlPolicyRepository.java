package com.app.backend.trade.portfolio.rl;

import com.google.gson.Gson;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;

import java.sql.ResultSet;
import java.sql.SQLException;

@Repository
public class PortfolioRlPolicyRepository {

    private final JdbcTemplate jdbcTemplate;
    private final Gson gson = new Gson();

    public PortfolioRlPolicyRepository(JdbcTemplate jdbcTemplate) { this.jdbcTemplate = jdbcTemplate; }

    public void save(String tri, String version, String notes, PortfolioRlPolicyModel model) {
        String sql = "INSERT INTO trade_ai.portfolio_rl_policy_models (tri, algo_version, input_size, feature_means, feature_stds, model_zip, lr, l2, hidden1, hidden2, dropout, notes, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?, now(), now())";
        byte[] bytes;
        try {
            java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
            org.deeplearning4j.util.ModelSerializer.writeModel(model.getNetwork(), baos, true);
            bytes = baos.toByteArray();
        } catch (Exception e) { throw new RuntimeException("Sérialisation RL policy échouée: " + e.getMessage(), e); }
        String meansJson = model.getFeatureMeans() != null ? gson.toJson(model.getFeatureMeans()) : null;
        String stdsJson = model.getFeatureStds() != null ? gson.toJson(model.getFeatureStds()) : null;
        jdbcTemplate.update(con -> {
            var ps = con.prepareStatement(sql);
            ps.setString(1, tri);
            ps.setString(2, version);
            ps.setInt(3, model.getInputSize());
            ps.setString(4, meansJson);
            ps.setString(5, stdsJson);
            ps.setBytes(6, bytes);
            ps.setDouble(7, model.getConfig().getLearningRate());
            ps.setDouble(8, model.getConfig().getL2());
            ps.setInt(9, model.getConfig().getHidden1());
            ps.setInt(10, model.getConfig().getHidden2());
            ps.setDouble(11, model.getConfig().getDropout());
            ps.setString(12, notes);
            return ps;
        });
    }

    public LoadedRlPolicy loadLatest(String tri, PortfolioRlConfig cfg) {
        String sql = "SELECT * FROM trade_ai.portfolio_rl_policy_models WHERE tri = ? ORDER BY created_at DESC LIMIT 1";
        return jdbcTemplate.query(sql, ps -> ps.setString(1, tri), rs -> rs.next() ? map(rs, cfg) : null);
    }

    private LoadedRlPolicy map(ResultSet rs, PortfolioRlConfig cfg) throws SQLException {
        int inputSize = rs.getInt("input_size");
        String meansJson = rs.getString("feature_means");
        String stdsJson = rs.getString("feature_stds");
        double[] means = meansJson != null ? gson.fromJson(meansJson, double[].class) : null;
        double[] stds = stdsJson != null ? gson.fromJson(stdsJson, double[].class) : null;
        byte[] bytes = rs.getBytes("model_zip");
        try {
            PortfolioRlPolicyModel model = PortfolioRlPolicyModel.loadFromBytes(bytes, cfg, inputSize, means, stds);
            LoadedRlPolicy out = new LoadedRlPolicy();
            out.model = model;
            out.version = rs.getString("algo_version");
            out.createdAtMs = rs.getTimestamp("created_at").getTime();
            return out;
        } catch (Exception e) { throw new SQLException("Restauration RL policy échouée: " + e.getMessage(), e); }
    }

    public static class LoadedRlPolicy {
        public PortfolioRlPolicyModel model;
        public String version;
        public long createdAtMs;
    }
}

