package com.app.backend.trade.portfolio.rl;

import com.app.backend.trade.controller.LstmHelper;
import com.app.backend.trade.model.SignalType;
import com.app.backend.trade.portfolio.learning.PortfolioAllocationTrainer;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Entraîneur RL offline simplifié (baseline):
 * Approche: on traite futur rendement observé comme proxy du Q-value pour ouverture long.
 * - features construits via PortfolioAllocationTrainer.buildInferenceFeatures à chaque date passée
 * - label = futureRet horizon (positif => encouragé, négatif => découragé)
 * - Policy réseau apprend par régression (MSE) vers futureRet
 * Limites: pas de véritable exploration/off-policy; pointeur pour intégration PPO/SAC ultérieure.
 */
@Component
public class PortfolioRlTrainer {

    private static final Logger logger = LoggerFactory.getLogger(PortfolioRlTrainer.class);
    private final JdbcTemplate jdbcTemplate;
    private final PortfolioRlConfig config;
    private final PortfolioRlPolicyRepository repository;
    private final PortfolioAllocationTrainer allocationTrainer; // réutilise buildInferenceFeatures pour cohérence
    private final LstmHelper lstmHelper;

    public PortfolioRlTrainer(JdbcTemplate jdbcTemplate,
                              PortfolioRlConfig config,
                              PortfolioRlPolicyRepository repository,
                              PortfolioAllocationTrainer allocationTrainer,
                              LstmHelper lstmHelper) {
        this.jdbcTemplate = jdbcTemplate;
        this.config = config;
        this.repository = repository;
        this.allocationTrainer = allocationTrainer;
        this.lstmHelper = lstmHelper;
    }

    /**
     * Entraîne et sauvegarde le policy model.
     * @param symbols univers d'entraînement
     * @param tri clé de tri/index (ex: daily)
     */
    public PortfolioRlPolicyModel train(List<String> symbols, String tri) {
        List<double[]> featureRows = new ArrayList<>();
        List<double[]> labelRows = new ArrayList<>();
        int horizon = 5; // baseline horizon futur rendement
        for (String sym : symbols) {
            try { buildSymbolDataset(sym, tri, horizon, featureRows, labelRows); }
            catch (Exception e) { logger.warn("RL dataset partiel ignoré {}: {}", sym, e.getMessage()); }
        }
        if (featureRows.isEmpty()) {
            logger.warn("Dataset RL vide => modèle placeholder.");
            return new PortfolioRlPolicyModel(1, config);
        }
        int inputSize = featureRows.get(0).length;
        double[][] X = featureRows.toArray(new double[0][]);
        double[][] y = labelRows.toArray(new double[0][]); // shape [n][1]

        double[] means = null, stds = null;
        if (config.isNormalize()) {
            double[][] stats = computeMeansStds(X); means = stats[0]; stds = stats[1]; applyNorm(X, means, stds);
        }
        PortfolioRlPolicyModel model = new PortfolioRlPolicyModel(inputSize, config);
        model.setFeatureMeans(means); model.setFeatureStds(stds);
        var inX = Nd4j.create(X);
        var inY = Nd4j.create(y);
        double bestLoss = Double.POSITIVE_INFINITY;
        int patienceLeft = config.getPatience();
        var bestParams = model.getNetwork().params().dup();
        for (int epoch = 0; epoch < config.getEpochs(); epoch++) {
            model.getNetwork().fit(inX, inY);
            double loss = model.getNetwork().score();
            if (loss + config.getMinDelta() < bestLoss) {
                bestLoss = loss; patienceLeft = config.getPatience(); bestParams.assign(model.getNetwork().params());
            } else { patienceLeft--; if (patienceLeft <= 0) break; }
        }
        model.getNetwork().setParams(bestParams);
        logger.info("Fin entraînement RL policy loss={} tri={}", bestLoss, tri);
        try { repository.save(tri, String.format("rlpolicy-%dx%d", config.getHidden1(), config.getHidden2()), "offline-train", model); }
        catch (Exception e) { logger.warn("Sauvegarde RL policy échouée: {}", e.getMessage()); }
        return model;
    }

    private void buildSymbolDataset(String symbol, String tri, int horizon,
                                    List<double[]> featureRows,
                                    List<double[]> labelRows) {
        String sql = "SELECT lstm_created_at, signal_lstm, price_lstm, price_clo FROM signal_lstm WHERE symbol = ? AND tri = ? ORDER BY lstm_created_at ASC";
        List<Map<String,Object>> rows = jdbcTemplate.queryForList(sql, symbol, tri);
        if (rows.size() < horizon + 10) return; // insuffisant
        List<Double> closes = new ArrayList<>();
        for (Map<String,Object> r : rows) {
            Double c = asDouble(r.get("price_clo")); closes.add(c != null ? c : Double.NaN);
        }
        for (int i = 0; i < rows.size(); i++) {
            int futureIdx = i + horizon; if (futureIdx >= rows.size()) break;
            Double start = closes.get(i); Double end = closes.get(futureIdx);
            if (start == null || end == null || start.isNaN() || end.isNaN() || start == 0) continue;
            double futureRet = (end - start) / start; // label
            Map<String,Object> r = rows.get(i);
            String sigStr = String.valueOf(r.get("signal_lstm")); SignalType sig;
            try { sig = SignalType.valueOf(sigStr); } catch (Exception e) { sig = SignalType.NONE; }
            double signalBuy = sig == SignalType.BUY ? 1.0 : 0.0;
            double signalSell = sig == SignalType.SELL ? 1.0 : 0.0;
            Double pPred = asDouble(r.get("price_lstm")); Double pC = asDouble(r.get("price_clo"));
            double deltaPred = (pPred != null && pC != null && pC != 0) ? (pPred - pC)/pC : 0.0;
            double vol20 = computeVol(closes, i, 20);
            // Récup features métriques via allocationTrainer.buildInferenceFeatures (dernière version disponible)
            double[] baseFeat = allocationTrainer.buildInferenceFeatures(symbol, tri);
            if (baseFeat == null) continue;
            // Align dimension: baseFeat déjà contient 9 éléments (cf. trainer) => on utilise directement
            double[] feat = baseFeat.clone(); // {signalBuy, signalSell, deltaPred, vol, profitFactor, winRate, maxDD, businessScore, totalTrades}
            featureRows.add(feat);
            labelRows.add(new double[]{futureRet});
        }
    }

    private double[][] computeMeansStds(double[][] X) { int r=X.length,c=X[0].length; double[] means=new double[c]; double[] stds=new double[c]; for(int j=0;j<c;j++){ double sum=0; for(int i=0;i<r;i++) sum+=X[i][j]; double mean=sum/r; means[j]=mean; double var=0; for(int i=0;i<r;i++){ double d=X[i][j]-mean; var+=d*d;} var/=Math.max(1,r-1); stds[j]=var<=0?1.0:Math.sqrt(var);} return new double[][]{means,stds}; }
    private void applyNorm(double[][] X,double[] means,double[] stds){ for(int i=0;i<X.length;i++){ for(int j=0;j<X[i].length;j++){ double std=stds[j]==0?1.0:stds[j]; X[i][j]=(X[i][j]-means[j])/std; } } }
    private Double asDouble(Object o){ if(o==null) return null; try{return Double.parseDouble(o.toString());}catch(Exception e){return null;} }
    private double computeVol(List<Double> closes,int idx,int window){ if(idx<window) return 0.0; List<Double> slice=closes.subList(idx-window, idx); List<Double> rets=new ArrayList<>(); for(int i=1;i<slice.size();i++){ double prev=slice.get(i-1); double cur=slice.get(i); if(prev!=0) rets.add((cur-prev)/prev);} if(rets.isEmpty()) return 0.0; double mean=rets.stream().mapToDouble(d->d).average().orElse(0.0); double var=0; for(double r:rets) var+=(r-mean)*(r-mean); var/=rets.size(); return Math.sqrt(var);} }

