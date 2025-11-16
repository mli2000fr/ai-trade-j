package com.app.backend.trade.portfolio.advanced;

import com.app.backend.trade.controller.LstmHelper;
import com.app.backend.trade.model.SignalType;
import com.app.backend.trade.util.TradeUtils;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Entraîne le modèle multi-tête supervision simple.
 * Pour chaque ligne historique (signal_lstm):
 * - labelSelect = futureReturn > positiveRetThreshold ? 1 : 0
 * - labelWeight  = max(0, futureReturn) (ou |futureReturn| si autorise short)
 * - labelSide    = signe(futureReturn) (-1,0,1)
 */
@Component
public class PortfolioMultiHeadTrainer {

    private static final Logger logger = LoggerFactory.getLogger(PortfolioMultiHeadTrainer.class);
    private final JdbcTemplate jdbcTemplate;
    private final PortfolioMultiHeadConfig config;
    private final LstmHelper lstmHelper;
    private final PortfolioMultiHeadRepository repository;

    public PortfolioMultiHeadTrainer(JdbcTemplate jdbcTemplate, PortfolioMultiHeadConfig config,
                                     LstmHelper lstmHelper, PortfolioMultiHeadRepository repository) {
        this.jdbcTemplate = jdbcTemplate;
        this.config = config;
        this.lstmHelper = lstmHelper;
        this.repository = repository;
    }

    public PortfolioMultiHeadModel train(List<String> symbols, String tri) {
        List<double[]> featureRows = new ArrayList<>();
        List<double[]> labelSelect = new ArrayList<>();
        List<double[]> labelWeight = new ArrayList<>();
        List<double[]> labelSide = new ArrayList<>();

        for (String sym : symbols) {
            try { buildDataset(sym, tri, featureRows, labelSelect, labelWeight, labelSide); }
            catch (Exception e) { logger.warn("Dataset multi-head partiel ignoré {}: {}", sym, e.getMessage()); }
        }
        if (featureRows.isEmpty()) {
            logger.warn("Dataset multi-head vide => modèle placeholder.");
            return new PortfolioMultiHeadModel(1, config.getHidden1(), config.getHidden2(), config.getLearningRate(), config.getL2(), config.getDropout());
        }
        int inputSize = featureRows.get(0).length;
        double[][] X = featureRows.toArray(new double[0][]);
        double[][] ySel = labelSelect.toArray(new double[0][]);
        double[][] yWei = labelWeight.toArray(new double[0][]);
        double[][] ySide = labelSide.toArray(new double[0][]);

        double[] means = null; double[] stds = null;
        if (config.isNormalize()) {
            double[][] stats = computeMeansStds(X); means = stats[0]; stds = stats[1];
            applyNorm(X, means, stds);
        }
        PortfolioMultiHeadModel model = new PortfolioMultiHeadModel(inputSize, config.getHidden1(), config.getHidden2(), config.getLearningRate(), config.getL2(), config.getDropout());
        model.setFeatureMeans(means); model.setFeatureStds(stds);
        ComputationGraph g = model.getGraph();
        var inX = Nd4j.create(X);
        var inSel = Nd4j.create(ySel);
        var inWei = Nd4j.create(yWei);
        var inSide = Nd4j.create(ySide);

        double bestLoss = Double.POSITIVE_INFINITY;
        int patienceLeft = config.getPatience();
        var bestParams = g.params().dup();
        for (int epoch = 0; epoch < config.getEpochs(); epoch++) {
            g.fit(new org.nd4j.linalg.dataset.MultiDataSet(new org.nd4j.linalg.api.ndarray.INDArray[]{inX}, new org.nd4j.linalg.api.ndarray.INDArray[]{inSel, inWei, inSide}));
            double loss = g.score();
            if (loss + config.getMinDelta() < bestLoss) {
                bestLoss = loss; patienceLeft = config.getPatience(); bestParams.assign(g.params());
            } else { patienceLeft--; if (patienceLeft <= 0) break; }
        }
        g.setParams(bestParams);
        try {
            repository.save(tri, String.format("multihead-%dx%d", config.getHidden1(), config.getHidden2()), "post-train", model, config);
        } catch (Exception e) { logger.warn("Sauvegarde modèle multi-head échouée: {}", e.getMessage()); }
        logger.info("Fin entraînement multi-head loss={} tri={}", bestLoss, tri);
        return model;
    }

    private void buildDataset(String symbol, String tri,
                              List<double[]> featureRows,
                              List<double[]> labelSelect,
                              List<double[]> labelWeight,
                              List<double[]> labelSide) {
        String sql = "SELECT lstm_created_at, signal_lstm, price_lstm, price_clo FROM signal_lstm WHERE symbol = ? AND tri = ? ORDER BY lstm_created_at ASC";
        List<Map<String,Object>> rows = jdbcTemplate.queryForList(sql, symbol, tri);
        if (rows.size() < 30) return; // minimum
        List<Double> closes = new ArrayList<>();
        for (Map<String,Object> r : rows) {
            Double c = asDouble(r.get("price_clo")); closes.add(c != null ? c : Double.NaN);
        }
        // Metrics modèle
        String sqlMetric = "SELECT profit_factor, win_rate, max_drawdown, business_score, total_trades FROM trade_ai.lstm_models WHERE symbol = ? ORDER BY updated_date DESC LIMIT 1";
        Map<String,Object> metric = null; try { metric = jdbcTemplate.queryForMap(sqlMetric, symbol); } catch (Exception ignored) {}
        double profitFactor = safeMetric(metric, "profit_factor");
        double winRate = safeMetric(metric, "win_rate");
        double maxDD = safeMetric(metric, "max_drawdown");
        double businessScore = safeMetric(metric, "business_score");
        double totalTrades = safeMetric(metric, "total_trades");

        int horizon = 5; // fixe pour baseline
        for (int i = 0; i < rows.size(); i++) {
            int futureIdx = i + horizon; if (futureIdx >= rows.size()) break;
            Double start = closes.get(i); Double end = closes.get(futureIdx);
            if (start == null || end == null || start.isNaN() || end.isNaN() || start == 0) continue;
            double futureRet = (end - start) / start;
            Map<String,Object> r = rows.get(i);
            String sigStr = String.valueOf(r.get("signal_lstm")); SignalType sig;
            try { sig = SignalType.valueOf(sigStr); } catch (Exception e) { sig = SignalType.NONE; }
            double signalBuy = sig == SignalType.BUY ? 1.0 : 0.0;
            double signalSell = sig == SignalType.SELL ? 1.0 : 0.0;
            Double pPred = asDouble(r.get("price_lstm")); Double pC = asDouble(r.get("price_clo"));
            double deltaPred = (pPred != null && pC != null && pC != 0) ? (pPred - pC)/pC : 0.0;
            double vol20 = computeVol(closes, i, 20);
            double[] feat = new double[]{signalBuy, signalSell, deltaPred, vol20, profitFactor, winRate, maxDD, businessScore, totalTrades};
            featureRows.add(feat);
            double sel = futureRet > config.getPositiveRetThreshold() ? 1.0 : 0.0;
            double weight = futureRet > 0 ? futureRet : 0.0; // long-only baseline
            double side = futureRet > 0 ? 1.0 : (futureRet < 0 ? -1.0 : 0.0);
            labelSelect.add(new double[]{sel});
            labelWeight.add(new double[]{weight});
            labelSide.add(new double[]{side});
        }
    }

    private double[][] computeMeansStds(double[][] X) {
        int r = X.length, c = X[0].length; double[] means = new double[c]; double[] stds = new double[c];
        for (int j=0;j<c;j++){ double sum=0; for (int i=0;i<r;i++) sum+=X[i][j]; double mean=sum/r; means[j]=mean; double var=0; for(int i=0;i<r;i++){ double d=X[i][j]-mean; var+=d*d;} var/=Math.max(1,r-1); stds[j]=var<=0?1.0:Math.sqrt(var);} return new double[][]{means,stds}; }
    private void applyNorm(double[][] X,double[] means,double[] stds){ for(int i=0;i<X.length;i++){ for(int j=0;j<X[i].length;j++){ double std=stds[j]==0?1.0:stds[j]; X[i][j]=(X[i][j]-means[j])/std; } } }
    private double safeMetric(Map<String,Object> metric,String key){ if(metric==null||!metric.containsKey(key)||metric.get(key)==null) return 0.0; try{return Double.parseDouble(metric.get(key).toString());}catch(Exception e){return 0.0;} }
    private Double asDouble(Object o){ if(o==null) return null; try{return Double.parseDouble(o.toString());}catch(Exception e){return null;} }
    private double computeVol(List<Double> closes,int idx,int window){ if(idx<window) return 0.0; List<Double> slice=closes.subList(idx-window, idx); List<Double> rets=new ArrayList<>(); for(int i=1;i<slice.size();i++){ double prev=slice.get(i-1); double cur=slice.get(i); if(prev!=0) rets.add((cur-prev)/prev);} if(rets.isEmpty()) return 0.0; double mean=rets.stream().mapToDouble(d->d).average().orElse(0.0); double var=0; for(double r:rets) var+=(r-mean)*(r-mean); var/=rets.size(); return Math.sqrt(var);} }

