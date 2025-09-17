package com.app.backend.trade.controller;

import com.app.backend.trade.model.*;
import com.app.backend.trade.strategy.ParamsOptim;
import com.app.backend.trade.strategy.*;
import com.app.backend.trade.util.TradeConstant;
import com.app.backend.trade.util.TradeUtils;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.ta4j.core.BarSeries;
import org.ta4j.core.Rule;

import java.time.LocalDate;
import java.util.*;

@Service
public class BestCombinationStrategyHelper {

    private static final Logger logger = LoggerFactory.getLogger(StrategieHelper.class);
    private final SwingTradeOptimParams swingParams = new SwingTradeOptimParams();
    private final StrategieHelper strategieHelper;
    private final JdbcTemplate jdbcTemplate;
    private final StrategieBackTest strategieBackTest;
    private final Gson gson = new Gson();

    private static final int NB_IN = 2;
    private static final int NB_OUT = 2;
    private static final boolean INSERT_ONLY = true;

    @Autowired
    public BestCombinationStrategyHelper(StrategieHelper strategieHelper, JdbcTemplate jdbcTemplate, StrategieBackTest strategieBackTest) {
        this.strategieHelper = strategieHelper;
        this.jdbcTemplate = jdbcTemplate;
        this.strategieBackTest = strategieBackTest;
    }


    public BestCombinationResult findBestCombinationGlobal(String symbol) {
        Map<String, String> bestInOutStrategy = this.getInOutStrategiesForSymbol(symbol);
        BestCombinationResult bestGlobal = null;
        double bestScore = Double.NEGATIVE_INFINITY;
        List<Class<? extends TradeStrategy>> strategies = Arrays.asList(
            ImprovedTrendFollowingStrategy.class,
            SmaCrossoverStrategy.class,
            RsiStrategy.class,
            BreakoutStrategy.class,
            MacdStrategy.class,
            MeanReversionStrategy.class
        );
        // Conversion nom -> classe
        Class<? extends TradeStrategy> inClass = null;
        Class<? extends TradeStrategy> outClass = null;
        if (bestInOutStrategy != null) {
            for (Class<? extends TradeStrategy> clazz : strategies) {
                if (clazz.getSimpleName().equals(TradeUtils.parseStrategyName(bestInOutStrategy.get("in")))) {
                    inClass = clazz;
                }
                if (clazz.getSimpleName().equals(TradeUtils.parseStrategyName(bestInOutStrategy.get("out")))) {
                    outClass = clazz;
                }
            }
        }
        strategieHelper.updateDBDailyValu(symbol);
        List<DailyValue> listeValus = strategieHelper.getDailyValuesFromDb(symbol, TradeConstant.NOMBRE_TOTAL_BOUGIES_OPTIM);
        BarSeries barSeries = TradeUtils.mapping(listeValus);
        for (int in = 1; in <= NB_IN; in++) {
            for (int out = 1; out <= NB_OUT; out++) {
                // Générer les combinaisons qui incluent obligatoirement la best in/out
                List<List<Class<? extends TradeStrategy>>> inCombinations = (inClass != null) ? generateCombinationsWithMandatory(strategies, in, inClass) : generateCombinations(strategies, in);
                List<List<Class<? extends TradeStrategy>>> outCombinations = (outClass != null) ? generateCombinationsWithMandatory(strategies, out, outClass) : generateCombinations(strategies, out);
                for (List<Class<? extends TradeStrategy>> inCombo : inCombinations) {
                    for (List<Class<? extends TradeStrategy>> outCombo : outCombinations) {

                        StrategyFilterConfig filterConfig = new StrategyFilterConfig();
                        BestCombinationResult result = optimseStrategyMix(Arrays.asList(barSeries), inCombo, outCombo, filterConfig, swingParams);
                        // TradeUtils.log("Global search: inCombo=" + inCombo + ", outCombo=" + outCombo + " => score=" + result.score);
                        if (result != null && result.result.rendement > bestScore) {
                            bestScore = result.result.rendement;
                            bestGlobal = result;
                        }
                    }
                }
            }
        }
        TradeUtils.log("Best global combination for symbol=" + symbol + " : " + resultObjToString(bestGlobal));
        return bestGlobal;
    }


    private List<List<Class<? extends TradeStrategy>>> generateCombinations(List<Class<? extends TradeStrategy>> strategies, int n) {
        List<List<Class<? extends TradeStrategy>>> result = new ArrayList<>();
        generateCombinationsHelper(strategies, n, 0, new ArrayList<>(), result);
        return result;
    }
    private void generateCombinationsHelper(List<Class<? extends TradeStrategy>> strategies, int n, int start, List<Class<? extends TradeStrategy>> current, List<List<Class<? extends TradeStrategy>>> result) {
        if (current.size() == n) {
            result.add(new ArrayList<>(current));
            return;
        }
        for (int i = start; i < strategies.size(); i++) {
            current.add(strategies.get(i));
            generateCombinationsHelper(strategies, n, i + 1, current, result);
            current.remove(current.size() - 1);
        }
    }

    private List<List<Class<? extends TradeStrategy>>> generateCombinationsWithMandatory(List<Class<? extends TradeStrategy>> strategies, int n, Class<? extends TradeStrategy> mandatory) {
        List<List<Class<? extends TradeStrategy>>> all = generateCombinations(strategies, n);
        List<List<Class<? extends TradeStrategy>>> filtered = new ArrayList<>();
        for (List<Class<? extends TradeStrategy>> combo : all) {
            if (combo.contains(mandatory)) {
                filtered.add(combo);
            }
        }
        return filtered;
    }


    private BestCombinationResult evaluateCombination(List<BarSeries> seriesList, List<Class<? extends TradeStrategy>> inCombo, List<Class<? extends TradeStrategy>> outCombo) {
        BestCombinationResult resultObj = new BestCombinationResult();
        if (seriesList == null || seriesList.isEmpty()) {
            resultObj.result.rendement = Double.NEGATIVE_INFINITY;
            return resultObj;
        }
        BarSeries fullSeries = seriesList.get(0);
        int totalCount = fullSeries.getBarCount();
        int optimCount = (int) Math.round(totalCount * TradeConstant.PC_OPTIM);
        int testCount = totalCount - optimCount;
        // Séparer les bougies pour optimisation et test
        BarSeries[] split = TradeUtils.splitSeriesForWalkForward(fullSeries);
        BarSeries optimSeries = split[0];
        BarSeries testSeries = split[1];
        StrategieBackTest backTest = new StrategieBackTest();
        List<TradeStrategy> inStrategies = new ArrayList<>();
        List<String> inStrategyNames = new ArrayList<>();
        for (Class<? extends TradeStrategy> clazz : inCombo) {
            inStrategyNames.add(clazz.getSimpleName());
            if (clazz.equals(ImprovedTrendFollowingStrategy.class)) {
                StrategieBackTest.ImprovedTrendFollowingParams params = backTest.optimiseImprovedTrendFollowingParameters(optimSeries, 10, 30, 5, 15, 15, 25, 0.001, 0.01, 0.002);
                inStrategies.add(new ImprovedTrendFollowingStrategy(params.trendPeriod, params.shortMaPeriod, params.longMaPeriod, params.breakoutThreshold, params.useRsiFilter, params.rsiPeriod));
                resultObj.inParams.put("ImprovedTrendFollowing", params);
            } else if (clazz.equals(SmaCrossoverStrategy.class)) {
                StrategieBackTest.SmaCrossoverParams params = backTest.optimiseSmaCrossoverParameters(optimSeries, 5, 20, 10, 50);
                inStrategies.add(new SmaCrossoverStrategy(params.shortPeriod, params.longPeriod));
                resultObj.inParams.put("SmaCrossover", params);
            } else if (clazz.equals(RsiStrategy.class)) {
                StrategieBackTest.RsiParams params = backTest.optimiseRsiParameters(optimSeries, 10, 20, 20, 40, 5, 60, 80, 5);
                inStrategies.add(new RsiStrategy(params.rsiPeriod, params.oversold, params.overbought));
                resultObj.inParams.put("Rsi", params);
            } else if (clazz.equals(BreakoutStrategy.class)) {
                StrategieBackTest.BreakoutParams params = backTest.optimiseBreakoutParameters(optimSeries, 5, 50);
                inStrategies.add(new BreakoutStrategy(params.lookbackPeriod));
                resultObj.inParams.put("Breakout", params);
            } else if (clazz.equals(MacdStrategy.class)) {
                StrategieBackTest.MacdParams params = backTest.optimiseMacdParameters(optimSeries, 8, 16, 20, 30, 6, 12);
                inStrategies.add(new MacdStrategy(params.shortPeriod, params.longPeriod, params.signalPeriod));
                resultObj.inParams.put("Macd", params);
            } else if (clazz.equals(MeanReversionStrategy.class)) {
                StrategieBackTest.MeanReversionParams params = backTest.optimiseMeanReversionParameters(optimSeries, 10, 30, 1.0, 5.0, 0.5);
                inStrategies.add(new MeanReversionStrategy(params.smaPeriod, params.threshold));
                resultObj.inParams.put("MeanReversion", params);
            }
        }
        List<TradeStrategy> outStrategies = new ArrayList<>();
        List<String> outStrategyNames = new ArrayList<>();
        for (Class<? extends TradeStrategy> clazz : outCombo) {
            outStrategyNames.add(clazz.getSimpleName());
            if (clazz.equals(ImprovedTrendFollowingStrategy.class)) {
                StrategieBackTest.ImprovedTrendFollowingParams params = backTest.optimiseImprovedTrendFollowingParameters(optimSeries, 10, 30, 5, 15, 15, 25, 0.001, 0.01, 0.002);
                outStrategies.add(new ImprovedTrendFollowingStrategy(params.trendPeriod, params.shortMaPeriod, params.longMaPeriod, params.breakoutThreshold, params.useRsiFilter, params.rsiPeriod));
                resultObj.outParams.put("ImprovedTrendFollowing", params);
            } else if (clazz.equals(SmaCrossoverStrategy.class)) {
                StrategieBackTest.SmaCrossoverParams params = backTest.optimiseSmaCrossoverParameters(optimSeries, 5, 20, 10, 50);
                outStrategies.add(new SmaCrossoverStrategy(params.shortPeriod, params.longPeriod));
                resultObj.outParams.put("SmaCrossover", params);
            } else if (clazz.equals(RsiStrategy.class)) {
                StrategieBackTest.RsiParams params = backTest.optimiseRsiParameters(optimSeries, 10, 20, 20, 40, 5, 60, 80, 5);
                outStrategies.add(new RsiStrategy(params.rsiPeriod, params.oversold, params.overbought));
                resultObj.outParams.put("Rsi", params);
            } else if (clazz.equals(BreakoutStrategy.class)) {
                StrategieBackTest.BreakoutParams params = backTest.optimiseBreakoutParameters(optimSeries, 5, 50);
                outStrategies.add(new BreakoutStrategy(params.lookbackPeriod));
                resultObj.outParams.put("Breakout", params);
            } else if (clazz.equals(MacdStrategy.class)) {
                StrategieBackTest.MacdParams params = backTest.optimiseMacdParameters(optimSeries, 8, 16, 20, 30, 6, 12);
                outStrategies.add(new MacdStrategy(params.shortPeriod, params.longPeriod, params.signalPeriod));
                resultObj.outParams.put("Macd", params);
            } else if (clazz.equals(MeanReversionStrategy.class)) {
                StrategieBackTest.MeanReversionParams params = backTest.optimiseMeanReversionParameters(optimSeries, 10, 30, 1.0, 5.0, 0.5);
                outStrategies.add(new MeanReversionStrategy(params.smaPeriod, params.threshold));
                resultObj.outParams.put("MeanReversion", params);
            }
        }
        final Rule finalEntryRule;
        final Rule finalExitRule;
        {
            Rule tempEntryRule = null;
            Rule tempExitRule = null;
            for (TradeStrategy strat : inStrategies) {
                if (tempEntryRule == null) tempEntryRule = strat.getEntryRule(testSeries);
                else tempEntryRule = tempEntryRule.or(strat.getEntryRule(testSeries));
            }
            for (TradeStrategy strat : outStrategies) {
                if (tempExitRule == null) tempExitRule = strat.getExitRule(testSeries);
                else tempExitRule = tempExitRule.or(strat.getExitRule(testSeries));
            }
            finalEntryRule = tempEntryRule;
            finalExitRule = tempExitRule;
        }
        TradeStrategy combinedStrategy = new TradeStrategy() {
            @Override
            public Rule getEntryRule(BarSeries s) { return finalEntryRule; }
            @Override
            public Rule getExitRule(BarSeries s) { return finalExitRule; }
            @Override
            public String getName() { return "CombinedStrategy"; }
        };
        RiskResult backtestResult = backTest.backtestStrategyRisk(combinedStrategy, testSeries);
        resultObj.result = backtestResult;
        resultObj.inStrategyNames = inStrategyNames;
        resultObj.outStrategyNames = outStrategyNames;
        resultObj.contextOptim = ParamsOptim.builder()
                .initialCapital(StrategieBackTest.INITIAL_CAPITAL)
                .riskPerTrade(StrategieBackTest.RISK_PER_TRADE)
                .stopLossPct(StrategieBackTest.STOP_LOSS_PCT)
                .takeProfitPct(StrategieBackTest.TAKE_PROFIL_PCT)
                .nbSimples(totalCount)
                .build();
        TradeUtils.log("BestCombinationResult : " + resultObjToString(resultObj));
        return resultObj;
    }


    private String resultObjToString(BestCombinationResult result) {
        StringBuilder sb = new StringBuilder();
        sb.append("inStrategyNames=").append(result.inStrategyNames).append(", ");
        sb.append("outStrategyNames=").append(result.outStrategyNames).append(", ");
        sb.append("score=").append(result.result.rendement).append(", ");
        sb.append("nbSimples=").append(result.contextOptim.nbSimples).append(", ");
        sb.append("inParams=").append(result.inParams).append(", ");
        sb.append("outParams=").append(result.outParams).append(", ");
        if (result.result != null) {
            sb.append("result={");
            sb.append("rendement=").append(result.result.rendement).append(", ");
            sb.append("maxDrawdown=").append(result.result.maxDrawdown).append(", ");
            sb.append("tradeCount=").append(result.result.tradeCount).append(", ");
            sb.append("winRate=").append(result.result.winRate).append(", ");
            sb.append("avgPnL=").append(result.result.avgPnL).append(", ");
            sb.append("profitFactor=").append(result.result.profitFactor).append(", ");
            sb.append("avgTradeBars=").append(result.result.avgTradeBars).append(", ");
            sb.append("maxTradeGain=").append(result.result.maxTradeGain).append(", ");
            sb.append("maxTradeLoss=").append(result.result.maxTradeLoss);
            sb.append("}");
        }
        return sb.toString();
    }


    public Map<String, String> getInOutStrategiesForSymbol(String symbol) {
        BestInOutStrategy best = strategieHelper.getBestInOutStrategy(symbol);
        Map<String, String> result = new HashMap<>();
        if (best != null) {
            result.put("in", best.getEntryName());
            result.put("out", best.getExitName());
        }
        return result;
    }


    public void saveBestCombinationResult(String symbol, BestCombinationResult result) {
        String inStrategyNamesJson = gson.toJson(result.inStrategyNames);
        String outStrategyNamesJson = gson.toJson(result.outStrategyNames);
        String inParamsJson = gson.toJson(result.inParams);
        String outParamsJson = gson.toJson(result.outParams);
        String check = new com.google.gson.Gson().toJson(result.check);
        // Vérifier si le symbol existe déjà
        String checkSql = "SELECT COUNT(*) FROM best_in_out_mix_strategy WHERE symbol = ?";
        int count = jdbcTemplate.queryForObject(checkSql, Integer.class, symbol);
        if (count > 0) {
            // UPDATE
            String updateSql = """
                    UPDATE best_in_out_mix_strategy SET 
                    in_strategy_names = ?, 
                    out_strategy_names = ?, 
                    in_params = ?, 
                    out_params = ?, 
                    rendement = ?, 
                    rendement_check = ?,
                    rendement_sum = ?,
                    rendement_diff = ?,
                    rendement_score = ?,
                    trade_count = ?, 
                    win_rate = ?, 
                    max_drawdown = ?, 
                    avg_pnl = ?, 
                    profit_factor = ?, 
                    avg_trade_bars = ?, 
                    max_trade_gain = ?, 
                    max_trade_loss = ?, 
                    score_swing_trade = ?, 
                    score_swing_trade_check = ?,
                    fltred_out = ?,
                    nb_simples = ?,
                    initial_capital = ?, 
                    risk_per_trade = ?, 
                    stop_loss_pct = ?, 
                    take_profit_pct = ?, 
                    check_result = ?
                    update_date = CURRENT_TIMESTAMP WHERE symbol = ?""";
            jdbcTemplate.update(updateSql,
                    inStrategyNamesJson,
                    outStrategyNamesJson,
                    inParamsJson,
                    outParamsJson,
                    result.result.rendement,
                    result.check.rendement,
                    result.rendementSum,
                    result.rendementDiff,
                    result.rendementScore,
                    result.result.tradeCount,
                    result.result.winRate,
                    result.result.maxDrawdown,
                    result.result.avgPnL,
                    result.result.profitFactor,
                    result.result.avgTradeBars,
                    result.result.maxTradeGain,
                    result.result.maxTradeLoss,
                    result.result.scoreSwingTrade,
                    result.check.scoreSwingTrade,
                    result.result.fltredOut,
                    result.contextOptim.nbSimples,
                    result.contextOptim.initialCapital,
                    result.contextOptim.riskPerTrade,
                    result.contextOptim.stopLossPct,
                    result.contextOptim.takeProfitPct,
                    check,
                    symbol);
        } else {
            // INSERT
            String insertSql = """
                    INSERT INTO best_in_out_mix_strategy (
                    symbol, 
                    in_strategy_names, 
                    out_strategy_names,
                    in_params, 
                    out_params, 
                    rendement,
                    rendement_check,
                    rendement_sum,
                    rendement_diff,
                    rendement_score,
                    trade_count,
                    win_rate,
                    max_drawdown,
                    avg_pnl,
                    profit_factor,
                    avg_trade_bars,
                    max_trade_gain,
                    max_trade_loss,
                    score_swing_trade, 
                    score_swing_trade_check,
                    fltred_out,
                    initial_capital, 
                    risk_per_trade, 
                    stop_loss_pct, 
                    take_profit_pct, 
                    nb_simples,
                    check_result,
                    create_date,
                    update_date
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""";
            jdbcTemplate.update(insertSql,
                    symbol,
                    inStrategyNamesJson,
                    outStrategyNamesJson,
                    inParamsJson,
                    outParamsJson,
                    result.result.rendement,
                    result.check.rendement,
                    result.rendementSum,
                    result.rendementDiff,
                    result.rendementScore,
                    result.result.tradeCount,
                    result.result.winRate,
                    result.result.maxDrawdown,
                    result.result.avgPnL,
                    result.result.profitFactor,
                    result.result.avgTradeBars,
                    result.result.maxTradeGain,
                    result.result.maxTradeLoss,
                    result.result.scoreSwingTrade,
                    result.check.scoreSwingTrade,
                    result.result.fltredOut,
                    result.contextOptim.initialCapital,
                    result.contextOptim.riskPerTrade,
                    result.contextOptim.stopLossPct,
                    result.contextOptim.takeProfitPct,
                    result.contextOptim.nbSimples,
                    check,
                    java.sql.Date.valueOf(java.time.LocalDate.now()));
        }
    }


    public BestCombinationResult getBestCombinationResult(String symbol) {
        String sql = "SELECT * FROM best_in_out_mix_strategy WHERE symbol = ? ORDER BY update_date DESC LIMIT 1";
        List<Map<String, Object>> rows = jdbcTemplate.queryForList(sql, symbol);
        if (rows.isEmpty()) return null;
        Map<String, Object> row = rows.get(0);
        BestCombinationResult result = new BestCombinationResult();
        result.symbol = symbol;
        result.inStrategyNames = gson.fromJson((String) row.get("in_strategy_names"), new TypeToken<List<String>>(){}.getType());
        result.outStrategyNames = gson.fromJson((String) row.get("out_strategy_names"), new TypeToken<List<String>>(){}.getType());
        result.inParams = gson.fromJson((String) row.get("in_params"), new TypeToken<Map<String, Object>>(){}.getType());
        result.outParams = gson.fromJson((String) row.get("out_params"), new TypeToken<Map<String, Object>>(){}.getType());
        result.result = RiskResult.builder()
                .rendement(row.get("rendement") != null ? ((Number) row.get("rendement")).doubleValue() : 0.0)
                .maxDrawdown(row.get("max_drawdown") != null ? ((Number) row.get("max_drawdown")).doubleValue() : 0.0)
                .tradeCount(row.get("trade_count") != null ? ((Number) row.get("trade_count")).intValue() : 0)
                .winRate(row.get("win_rate") != null ? ((Number) row.get("win_rate")).doubleValue() : 0.)
                .avgPnL(row.get("avg_pnl") != null ? ((Number) row.get("avg_pnl")).doubleValue() : 0.0)
                .profitFactor(row.get("profit_factor") != null ? ((Number) row.get("profit_factor")).doubleValue() : 0.0)
                .avgTradeBars( row.get("avg_trade_bars") != null ? ((Number) row.get("avg_trade_bars")).doubleValue() : 0.0)
                .maxTradeGain(row.get("max_trade_gain") != null ? ((Number) row.get("max_trade_gain")).doubleValue() : 0.0)
                .maxTradeLoss(row.get("max_trade_loss") != null ? ((Number) row.get("max_trade_loss")).doubleValue() : 0.0)
                .scoreSwingTrade(row.get("score_swing_trade") != null ? ((Number) row.get("score_swing_trade")).doubleValue() : 0.0)
                .build();
        result.contextOptim = ParamsOptim.builder()
                .initialCapital(row.get("initial_capital") != null ? ((Number) row.get("initial_capital")).doubleValue() : 0.0)
                .riskPerTrade(row.get("risk_per_trade") != null ? ((Number) row.get("risk_per_trade")).doubleValue() : 0.0)
                .stopLossPct(row.get("stop_loss_pct") != null ? ((Number) row.get("stop_loss_pct")).doubleValue() : 0.0)
                .takeProfitPct(row.get("take_profit_pct") != null ? ((Number) row.get("take_profit_pct")).doubleValue() : 0.0)
                .nbSimples(row.get("nb_simples") != null ? ((Integer) row.get("nb_simples")).intValue() : 0)
                .build();
        result.check= new com.google.gson.Gson().fromJson((String)row.get("check_result"), RiskResult.class);
        result.rendementSum = row.get("rendement_sum") != null ? ((Number) row.get("rendement_sum")).doubleValue() : 0.0;
        result.rendementDiff = row.get("rendement_diff") != null ? ((Number) row.get("rendement_diff")).doubleValue() : 0.0;
        result.rendementScore = row.get("rendement_score") != null ? ((Number) row.get("rendement_score")).doubleValue() : 0.0;
        return result;
    }


    public SignalInfo getSingalTypeFromDB(String symbol) {
        String sql = "SELECT signal_mix, mix_created_at FROM signal_mix WHERE symbol = ? ORDER BY mix_created_at DESC LIMIT 1";
        try {
            return jdbcTemplate.query(sql, ps -> ps.setString(1, symbol), rs -> {
                if (rs.next()) {
                    String signalStr = rs.getString("signal_mix");
                    java.sql.Date lastDate = rs.getDate("mix_created_at");
                    SignalType type;
                    try {
                        type = SignalType.valueOf(signalStr);
                    } catch (Exception e) {
                        logger.warn("SignalType inconnu en base: {}", signalStr);
                        type = null;
                    }

                    java.time.LocalDate lastTradingDay = TradeUtils.getLastTradingDayBefore(java.time.LocalDate.now());
                    java.time.LocalDate lastKnown = lastDate.toLocalDate();

                    // Si la dernière date connue est le dernier jour de cotation, la base est à jour
                    if (lastKnown.isEqual(lastTradingDay) || lastKnown.isAfter(lastTradingDay)) {
                        String dateSavedStr = lastKnown.format(java.time.format.DateTimeFormatter.ofPattern("dd/MM"));
                        return SignalInfo.builder().symbol(symbol).type(type).date(lastDate).dateStr(dateSavedStr).build();
                    }else{
                        return null;
                    }
                }
                return null;
            });
        } catch (Exception e) {
            logger.warn("Erreur SQL getSingalTypeFromDB pour {}: {}", symbol, e.getMessage());
            return null;
        }
    }

    public LocalDate saveSignalHistory(String symbol, SignalType signal) {
        java.time.LocalDate lastTradingDay = TradeUtils.getLastTradingDayBefore(java.time.LocalDate.now());
        String insertSql = "INSERT INTO signal_mix (symbol, signal_mix, mix_created_at) VALUES (?, ?, ?)";
        jdbcTemplate.update(insertSql,
                symbol,
                signal.name(),
                java.sql.Date.valueOf(lastTradingDay));
        return lastTradingDay;

    }

    public SignalInfo getSignal(String symbol) {

        SignalInfo mixDB = this.getSingalTypeFromDB(symbol);
        if(mixDB != null){
            String dateStr = mixDB.getDate().toLocalDate().format(java.time.format.DateTimeFormatter.ofPattern("dd/MM"));
            return SignalInfo.builder().symbol(symbol).type(mixDB.getType()).dateStr(dateStr).build();
        }

        java.time.LocalDate lastTradingDay = TradeUtils.getLastTradingDayBefore(java.time.LocalDate.now());
        BestCombinationResult bestCombinationResult = getBestCombinationResult(symbol);
        if(bestCombinationResult == null){
            return SignalInfo.builder().symbol(symbol).type(SignalType.NONE)
                    .dateStr(lastTradingDay.format(java.time.format.DateTimeFormatter.ofPattern("dd/MM"))).build();
        }
        this.strategieHelper.updateDBDailyValu(symbol);
        List<DailyValue> listeValus = strategieHelper.getDailyValuesFromDb(symbol, TradeConstant.NOMBRE_TOTAL_BOUGIES_FOR_SIGNAL);
        BarSeries barSeries = TradeUtils.mapping(listeValus);
        if (barSeries.getBarCount() == 0) {
            return SignalInfo.builder().symbol(symbol).type(SignalType.NONE)
                    .dateStr(lastTradingDay.format(java.time.format.DateTimeFormatter.ofPattern("dd-MM"))).build();
        }
        // Recréer les stratégies d'entrée
        List<TradeStrategy> inStrategies = new ArrayList<>();
        for (String name : bestCombinationResult.inStrategyNames) {
            Object params = bestCombinationResult.inParams.get(name.replace("Strategy", ""));
            if (name.equals("ImprovedTrendFollowingStrategy")) {
                StrategieBackTest.ImprovedTrendFollowingParams p = new Gson().fromJson(params.toString(), StrategieBackTest.ImprovedTrendFollowingParams.class);
                inStrategies.add(new ImprovedTrendFollowingStrategy(p.trendPeriod, p.shortMaPeriod, p.longMaPeriod, p.breakoutThreshold, p.useRsiFilter, p.rsiPeriod));
            } else if (name.equals("SmaCrossoverStrategy")) {
                StrategieBackTest.SmaCrossoverParams p = new Gson().fromJson(params.toString(), StrategieBackTest.SmaCrossoverParams.class);
                inStrategies.add(new SmaCrossoverStrategy(p.shortPeriod, p.longPeriod));
            } else if (name.equals("RsiStrategy")) {
                StrategieBackTest.RsiParams p = new Gson().fromJson(params.toString(), StrategieBackTest.RsiParams.class);
                inStrategies.add(new RsiStrategy(p.rsiPeriod, p.oversold, p.overbought));
            } else if (name.equals("BreakoutStrategy")) {
                StrategieBackTest.BreakoutParams p = new Gson().fromJson(params.toString(), StrategieBackTest.BreakoutParams.class);
                inStrategies.add(new BreakoutStrategy(p.lookbackPeriod));
            } else if (name.equals("MacdStrategy")) {
                StrategieBackTest.MacdParams p = new Gson().fromJson(params.toString(), StrategieBackTest.MacdParams.class);
                inStrategies.add(new MacdStrategy(p.shortPeriod, p.longPeriod, p.signalPeriod));
            } else if (name.equals("MeanReversionStrategy")) {
                StrategieBackTest.MeanReversionParams p = new Gson().fromJson(params.toString(), StrategieBackTest.MeanReversionParams.class);
                inStrategies.add(new MeanReversionStrategy(p.smaPeriod, p.threshold));
            }
        }
        // Recréer les stratégies de sortie
        List<TradeStrategy> outStrategies = new ArrayList<>();
        for (String name : bestCombinationResult.outStrategyNames) {
            Object params = bestCombinationResult.outParams.get(name.replace("Strategy", ""));
            if (name.equals("ImprovedTrendFollowingStrategy")) {
                StrategieBackTest.ImprovedTrendFollowingParams p = new Gson().fromJson(params.toString(), StrategieBackTest.ImprovedTrendFollowingParams.class);
                outStrategies.add(new ImprovedTrendFollowingStrategy(p.trendPeriod, p.shortMaPeriod, p.longMaPeriod, p.breakoutThreshold, p.useRsiFilter, p.rsiPeriod));
            } else if (name.equals("SmaCrossoverStrategy")) {
                StrategieBackTest.SmaCrossoverParams p = new Gson().fromJson(params.toString(), StrategieBackTest.SmaCrossoverParams.class);
                outStrategies.add(new SmaCrossoverStrategy(p.shortPeriod, p.longPeriod));
            } else if (name.equals("RsiStrategy")) {
                StrategieBackTest.RsiParams p = new Gson().fromJson(params.toString(), StrategieBackTest.RsiParams.class);
                outStrategies.add(new RsiStrategy(p.rsiPeriod, p.oversold, p.overbought));
            } else if (name.equals("BreakoutStrategy")) {
                StrategieBackTest.BreakoutParams p = new Gson().fromJson(params.toString(), StrategieBackTest.BreakoutParams.class);
                outStrategies.add(new BreakoutStrategy(p.lookbackPeriod));
            } else if (name.equals("MacdStrategy")) {
                StrategieBackTest.MacdParams p = new Gson().fromJson(params.toString(), StrategieBackTest.MacdParams.class);
                outStrategies.add(new MacdStrategy(p.shortPeriod, p.longPeriod, p.signalPeriod));
            } else if (name.equals("MeanReversionStrategy")) {
                StrategieBackTest.MeanReversionParams p = new Gson().fromJson(params.toString(), StrategieBackTest.MeanReversionParams.class);
                outStrategies.add(new MeanReversionStrategy(p.smaPeriod, p.threshold));
            }
        }
        // Combiner les règles d'entrée et de sortie
        Rule entryRule = null;
        Rule exitRule = null;
        for (TradeStrategy strat : inStrategies) {
            if (entryRule == null) entryRule = strat.getEntryRule(barSeries);
            else entryRule = entryRule.or(strat.getEntryRule(barSeries));
        }
        for (TradeStrategy strat : outStrategies) {
            if (exitRule == null) exitRule = strat.getExitRule(barSeries);
            else exitRule = exitRule.or(strat.getExitRule(barSeries));
        }
        SignalType signal;
        int lastIndex = barSeries.getEndIndex();
        if (entryRule != null && entryRule.isSatisfied(lastIndex)) {
            signal = SignalType.BUY;
        } else if (exitRule != null && exitRule.isSatisfied(lastIndex)) {
            signal = SignalType.SELL;
        } else {
            signal = SignalType.HOLD;
        }
        LocalDate dateSaved = saveSignalHistory(symbol, signal);
        String dateSavedStr = dateSaved.format(java.time.format.DateTimeFormatter.ofPattern("dd/MM"));
        return SignalInfo.builder().symbol(symbol).type(signal).dateStr(dateSavedStr).build();
    }


    public List<String> getAllAssetSymbolsEligibleFromDb() {
        String sql = "SELECT symbol FROM trade_ai.alpaca_asset WHERE status = 'active' and eligible = true ORDER BY symbol ASC;";
        return jdbcTemplate.queryForList(sql, String.class);
    }

    public List<String> getSymbolFitredFromTabSingle(String sort) {
        String orderBy = sort == null ? "rendement_score" : sort;
        String sql = "select symbol from trade_ai.best_in_out_single_strategy where fltred_out = 'false'";
        sql += " ORDER BY " + orderBy + " DESC";
        return jdbcTemplate.queryForList(sql, String.class);
    }


    // Structure de suivi pour le monitoring du calcul mix strategies
    public static class MixStrategiesProgress {
        public int totalSymbols;
        public int processedSymbols;
        public int nbInsert;
        public int error;
        public String status; // en_cours, termine, erreur
        public String lastSymbol;
        public long startTime;
        public long endTime;
        public long lastUpdate;
    }

    private volatile MixStrategiesProgress mixStrategiesProgress = null;

    public MixStrategiesProgress getMixStrategiesProgress() {
        return mixStrategiesProgress;
    }

    public void calculMixStrategies(String sort){
        List<String> listeDbSymbols = this.getSymbolFitredFromTabSingle(sort);
        mixStrategiesProgress = new MixStrategiesProgress();
        mixStrategiesProgress.totalSymbols = listeDbSymbols.size();
        mixStrategiesProgress.processedSymbols = 0;
        mixStrategiesProgress.nbInsert = 0;
        mixStrategiesProgress.error = 0;
        mixStrategiesProgress.status = "en_cours";
        mixStrategiesProgress.startTime = System.currentTimeMillis();
        mixStrategiesProgress.lastUpdate = System.currentTimeMillis();
        // ...existing code...
        int nbThreads = Math.max(2, Runtime.getRuntime().availableProcessors());
        java.util.concurrent.ExecutorService executor = java.util.concurrent.Executors.newFixedThreadPool(nbThreads);
        java.util.concurrent.atomic.AtomicInteger error = new java.util.concurrent.atomic.AtomicInteger(0);
        java.util.concurrent.atomic.AtomicInteger nbInsert = new java.util.concurrent.atomic.AtomicInteger(0);
        java.util.concurrent.atomic.AtomicInteger processed = new java.util.concurrent.atomic.AtomicInteger(0);
        List<java.util.concurrent.Future<?>> futures = new ArrayList<>();
        for(String symbol : listeDbSymbols){
            futures.add(executor.submit(() -> {
                boolean isCalcul = true;
                try{
                    if(INSERT_ONLY){
                        String checkSql = "SELECT COUNT(*) FROM best_in_out_mix_strategy WHERE symbol = ?";
                        int count = jdbcTemplate.queryForObject(checkSql, Integer.class, symbol);
                        if(count > 0){
                            isCalcul = false;
                            TradeUtils.log("calculMixStrategies: symbole "+symbol+" déjà en base, on passe");
                        }
                    }
                    if(isCalcul){
                        BestCombinationResult result = findBestCombinationGlobal(symbol);
                        double scoreST =  TradeUtils.calculerScoreSwingTrade(result.getResult());
                        result.result.setScoreSwingTrade(scoreST);
                        this.saveBestCombinationResult(symbol, result);
                        nbInsert.incrementAndGet();
                    }
                }catch(Exception e){
                    error.incrementAndGet();
                    TradeUtils.log("Erreur calcul("+symbol+") : " + e.getMessage());
                } finally {
                    int proc = processed.incrementAndGet();
                    mixStrategiesProgress.processedSymbols = proc;
                    mixStrategiesProgress.nbInsert = nbInsert.get();
                    mixStrategiesProgress.error = error.get();
                    mixStrategiesProgress.lastSymbol = symbol;
                    mixStrategiesProgress.lastUpdate = System.currentTimeMillis();
                }
            }));
        }
        // Attendre la fin de tous les threads
        for (java.util.concurrent.Future<?> f : futures) {
            try { f.get(); } catch(Exception ignored) {}
        }
        executor.shutdown();
        mixStrategiesProgress.status = "termine";
        mixStrategiesProgress.endTime = System.currentTimeMillis();
        TradeUtils.log("calculMixStrategies: total: "+listeDbSymbols.size()+", nbInsert: "+nbInsert.get()+", error: " + error.get());
    }



    public void calculScoreST(){
        String selectSql = "SELECT symbol, backtest_result FROM best_in_out_mix_strategy";
        List<Map<String, Object>> rows = jdbcTemplate.queryForList(selectSql);
        for (Map<String, Object> row : rows) {
            String symbol = (String) row.get("symbol");
            String backtestResultJson = (String) row.get("backtest_result");
            if (backtestResultJson == null || backtestResultJson.isEmpty()) continue;
            RiskResult result = null;
            try {
                result = gson.fromJson(backtestResultJson, RiskResult.class);
            } catch (Exception e) {
                continue;
            }
            if (result == null) continue;
            double scoreST = TradeUtils.calculerScoreSwingTrade(result);
            String updateSql = "UPDATE best_in_out_mix_strategy SET score_swing_trade = ? WHERE symbol = ?";
            jdbcTemplate.update(updateSql, scoreST, symbol);
        }
    }



    private BestCombinationResult optimseStrategyMix(List<BarSeries> seriesList, List<Class<? extends TradeStrategy>> inCombo, List<Class<? extends TradeStrategy>> outCombo, StrategyFilterConfig filterConfig, SwingTradeOptimParams swingParams) {
        BestCombinationResult resultObj = new BestCombinationResult();
        if (seriesList == null || seriesList.isEmpty()) {
            resultObj.result.rendement = Double.NEGATIVE_INFINITY;
            return resultObj;
        }
        BarSeries fullSeries = seriesList.get(0);
        int totalCount = fullSeries.getBarCount();
        // Paramètres walk-forward (modifiable si besoin)
        double optimWindowPct = 0.2;
        double testWindowPct = 0.1;
        double stepWindowPct = 0.1;
        int optimWindow = Math.max(1, (int) Math.round(totalCount * optimWindowPct));
        int testWindow = Math.max(1, (int) Math.round(totalCount * testWindowPct));
        int stepWindow = Math.max(1, (int) Math.round(totalCount * stepWindowPct));
        int kFolds = 5;
        int foldSize = (totalCount - (optimWindow + testWindow)) / kFolds;
        if (foldSize < 1) foldSize = 1;
        List<RiskResult> foldResults = new ArrayList<>();
        List<Double> trainPerformances = new ArrayList<>();
        List<Double> testPerformances = new ArrayList<>();
        List<ComboMixResult> listeComboMixResult = new ArrayList<>();

        for (int fold = 0; fold < kFolds; fold++) {
            int start = fold * foldSize;
            if (start + optimWindow + testWindow > totalCount) break;
            BarSeries optimSeries = fullSeries.getSubSeries(start, start + optimWindow);
            BarSeries testSeries = fullSeries.getSubSeries(start + optimWindow, start + optimWindow + testWindow);
            List<TradeStrategy> inStrategies = new ArrayList<>();
            List<String> inStrategyNames = new ArrayList<>();
            for (Class<? extends TradeStrategy> clazz : inCombo) {
                inStrategyNames.add(clazz.getSimpleName());
                if (clazz.equals(ImprovedTrendFollowingStrategy.class)) {
                    StrategieBackTest.ImprovedTrendFollowingParams params = strategieBackTest.optimiseImprovedTrendFollowingParameters(
                        optimSeries,
                        swingParams.trendMaMin, swingParams.trendMaMax,
                        swingParams.trendShortMaMin, swingParams.trendShortMaMax,
                        swingParams.trendLongMaMin, swingParams.trendLongMaMax,
                        swingParams.trendBreakoutMin, swingParams.trendBreakoutMax, swingParams.trendBreakoutStep
                    );
                    inStrategies.add(new ImprovedTrendFollowingStrategy(params.trendPeriod, params.shortMaPeriod, params.longMaPeriod, params.breakoutThreshold, params.useRsiFilter, params.rsiPeriod));
                    resultObj.inParams.put("ImprovedTrendFollowing", params);
                } else if (clazz.equals(SmaCrossoverStrategy.class)) {
                    StrategieBackTest.SmaCrossoverParams params = strategieBackTest.optimiseSmaCrossoverParameters(
                        optimSeries,
                        swingParams.smaShortMin, swingParams.smaShortMax,
                        swingParams.smaLongMin, swingParams.smaLongMax
                    );
                    inStrategies.add(new SmaCrossoverStrategy(params.shortPeriod, params.longPeriod));
                    resultObj.inParams.put("SmaCrossover", params);
                } else if (clazz.equals(RsiStrategy.class)) {
                    StrategieBackTest.RsiParams params = strategieBackTest.optimiseRsiParameters(
                        optimSeries,
                        swingParams.rsiPeriodMin, swingParams.rsiPeriodMax,
                        swingParams.rsiOversoldMin, swingParams.rsiOversoldMax,
                        swingParams.rsiStep,
                        swingParams.rsiOverboughtMin, swingParams.rsiOverboughtMax,
                        swingParams.rsiStep
                    );
                    inStrategies.add(new RsiStrategy(params.rsiPeriod, params.oversold, params.overbought));
                    resultObj.inParams.put("Rsi", params);
                } else if (clazz.equals(BreakoutStrategy.class)) {
                    StrategieBackTest.BreakoutParams params = strategieBackTest.optimiseBreakoutParameters(
                        optimSeries,
                        swingParams.breakoutLookbackMin, swingParams.breakoutLookbackMax
                    );
                    inStrategies.add(new BreakoutStrategy(params.lookbackPeriod));
                    resultObj.inParams.put("Breakout", params);
                } else if (clazz.equals(MacdStrategy.class)) {
                    StrategieBackTest.MacdParams params = strategieBackTest.optimiseMacdParameters(
                        optimSeries,
                        swingParams.macdShortMin, swingParams.macdShortMax,
                        swingParams.macdLongMin, swingParams.macdLongMax,
                        swingParams.macdSignalMin, swingParams.macdSignalMax
                    );
                    inStrategies.add(new MacdStrategy(params.shortPeriod, params.longPeriod, params.signalPeriod));
                    resultObj.inParams.put("Macd", params);
                } else if (clazz.equals(MeanReversionStrategy.class)) {
                    StrategieBackTest.MeanReversionParams params = strategieBackTest.optimiseMeanReversionParameters(
                            optimSeries,
                            swingParams.meanRevSmaMin, swingParams.meanRevSmaMax,
                            swingParams.meanRevThresholdMin, swingParams.meanRevThresholdMax,
                            swingParams.meanRevThresholdStep
                    );
                    inStrategies.add(new MeanReversionStrategy(params.smaPeriod, params.threshold));
                    resultObj.inParams.put("MeanReversion", params);
                }
            }
            List<TradeStrategy> outStrategies = new ArrayList<>();
            List<String> outStrategyNames = new ArrayList<>();
            for (Class<? extends TradeStrategy> clazz : outCombo) {
                outStrategyNames.add(clazz.getSimpleName());
                if (clazz.equals(ImprovedTrendFollowingStrategy.class)) {
                    StrategieBackTest.ImprovedTrendFollowingParams params = strategieBackTest.optimiseImprovedTrendFollowingParameters(
                            optimSeries,
                            swingParams.trendMaMin, swingParams.trendMaMax,
                            swingParams.trendShortMaMin, swingParams.trendShortMaMax,
                            swingParams.trendLongMaMin, swingParams.trendLongMaMax,
                            swingParams.trendBreakoutMin, swingParams.trendBreakoutMax, swingParams.trendBreakoutStep
                    );
                    outStrategies.add(new ImprovedTrendFollowingStrategy(params.trendPeriod, params.shortMaPeriod, params.longMaPeriod, params.breakoutThreshold, params.useRsiFilter, params.rsiPeriod));
                    resultObj.outParams.put("ImprovedTrendFollowing", params);
                } else if (clazz.equals(SmaCrossoverStrategy.class)) {
                    StrategieBackTest.SmaCrossoverParams params = strategieBackTest.optimiseSmaCrossoverParameters(
                            optimSeries,
                            swingParams.smaShortMin, swingParams.smaShortMax,
                            swingParams.smaLongMin, swingParams.smaLongMax
                    );
                    outStrategies.add(new SmaCrossoverStrategy(params.shortPeriod, params.longPeriod));
                    resultObj.outParams.put("SmaCrossover", params);
                } else if (clazz.equals(RsiStrategy.class)) {
                    StrategieBackTest.RsiParams params = strategieBackTest.optimiseRsiParameters(
                            optimSeries,
                            swingParams.rsiPeriodMin, swingParams.rsiPeriodMax,
                            swingParams.rsiOversoldMin, swingParams.rsiOversoldMax,
                            swingParams.rsiStep,
                            swingParams.rsiOverboughtMin, swingParams.rsiOverboughtMax,
                            swingParams.rsiStep
                    );
                    outStrategies.add(new RsiStrategy(params.rsiPeriod, params.oversold, params.overbought));
                    resultObj.outParams.put("Rsi", params);
                } else if (clazz.equals(BreakoutStrategy.class)) {
                    StrategieBackTest.BreakoutParams params = strategieBackTest.optimiseBreakoutParameters(
                            optimSeries,
                            swingParams.breakoutLookbackMin, swingParams.breakoutLookbackMax
                    );
                    outStrategies.add(new BreakoutStrategy(params.lookbackPeriod));
                    resultObj.outParams.put("Breakout", params);
                } else if (clazz.equals(MacdStrategy.class)) {
                    StrategieBackTest.MacdParams params = strategieBackTest.optimiseMacdParameters(
                            optimSeries,
                            swingParams.macdShortMin, swingParams.macdShortMax,
                            swingParams.macdLongMin, swingParams.macdLongMax,
                            swingParams.macdSignalMin, swingParams.macdSignalMax
                    );
                    outStrategies.add(new MacdStrategy(params.shortPeriod, params.longPeriod, params.signalPeriod));
                    resultObj.outParams.put("Macd", params);
                } else if (clazz.equals(MeanReversionStrategy.class)) {
                    StrategieBackTest.MeanReversionParams params = strategieBackTest.optimiseMeanReversionParameters(
                            optimSeries,
                            swingParams.meanRevSmaMin, swingParams.meanRevSmaMax,
                            swingParams.meanRevThresholdMin, swingParams.meanRevThresholdMax,
                            swingParams.meanRevThresholdStep
                    );
                    outStrategies.add(new MeanReversionStrategy(params.smaPeriod, params.threshold));
                    resultObj.outParams.put("MeanReversion", params);
                }
            }
            // Combiner les règles d'entrée et de sortie
            final Rule finalEntryRuleOptim;
            final Rule finalExitRuleOptim;
            {
                Rule tempEntryRule = null;
                Rule tempExitRule = null;
                for (TradeStrategy strat : inStrategies) {
                    if (tempEntryRule == null) tempEntryRule = strat.getEntryRule(optimSeries);
                    else tempEntryRule = tempEntryRule.or(strat.getEntryRule(optimSeries));
                }
                for (TradeStrategy strat : outStrategies) {
                    if (tempExitRule == null) tempExitRule = strat.getExitRule(optimSeries);
                    else tempExitRule = tempExitRule.or(strat.getExitRule(optimSeries));
                }
                finalEntryRuleOptim = tempEntryRule;
                finalExitRuleOptim = tempExitRule;
            }
            final Rule finalEntryRuleOTest;
            final Rule finalExitRuleOpTest;
            {
                Rule tempEntryRule = null;
                Rule tempExitRule = null;
                for (TradeStrategy strat : inStrategies) {
                    if (tempEntryRule == null) tempEntryRule = strat.getEntryRule(testSeries);
                    else tempEntryRule = tempEntryRule.or(strat.getEntryRule(testSeries));
                }
                for (TradeStrategy strat : outStrategies) {
                    if (tempExitRule == null) tempExitRule = strat.getExitRule(testSeries);
                    else tempExitRule = tempExitRule.or(strat.getExitRule(testSeries));
                }
                finalEntryRuleOTest = tempEntryRule;
                finalExitRuleOpTest = tempExitRule;
            }
            TradeStrategy combinedStrategyOptim = new TradeStrategy() {
                @Override
                public Rule getEntryRule(BarSeries s) { return finalEntryRuleOptim; }
                @Override
                public Rule getExitRule(BarSeries s) { return finalExitRuleOptim; }
                @Override
                public String getName() { return "CombinedStrategy"; }
            };
            TradeStrategy combinedStrategyTest = new TradeStrategy() {
                @Override
                public Rule getEntryRule(BarSeries s) { return finalEntryRuleOTest; }
                @Override
                public Rule getExitRule(BarSeries s) { return finalExitRuleOpTest; }
                @Override
                public String getName() { return "CombinedStrategy"; }
            };
            // Backtest sur la partie optimisation (train)
            RiskResult trainResult = strategieBackTest.backtestStrategy(combinedStrategyOptim, optimSeries);
            trainPerformances.add(trainResult.rendement);
            // Backtest sur la partie test
            RiskResult testResult = strategieBackTest.backtestStrategy(combinedStrategyTest, testSeries);
            testPerformances.add(testResult.rendement);
            foldResults.add(testResult);

            double overfitRatioCombo = testResult.getRendement() / (trainResult.getRendement() == 0.0 ? 1.0 : trainResult.getRendement());
            boolean isOverfitCombo = (overfitRatioCombo < 0.7 || overfitRatioCombo > 1.3);
            boolean stable = TradeUtils.isStableAndSimple(testResult, filterConfig);
            testResult.setFltredOut(!stable || isOverfitCombo);
            ComboMixResult combo =  ComboMixResult.builder()
                    .inStrategyNames(inStrategies.stream().map(TradeStrategy::getName).toList())
                    .outStrategyNames(outStrategies.stream().map(TradeStrategy::getName).toList())
                    .inParams(resultObj.inParams)
                    .outParams(resultObj.outParams)
                    .result(testResult)
                    .build();
            listeComboMixResult.add(combo);
        }
        // Agrégation des résultats des folds
        double sumRendement = 0.0, sumDrawdown = 0.0, sumWinRate = 0.0, sumProfitFactor = 0.0, sumAvgPnL = 0.0;
        double sumAvgTradeBars = 0.0, sumMaxTradeGain = 0.0, sumMaxTradeLoss = 0.0, sumScoreSwingTrade = 0.0;
        int sumTradeCount = 0;
        for (RiskResult res : foldResults) {
            sumRendement += res.rendement;
            sumDrawdown += res.maxDrawdown;
            sumWinRate += res.winRate;
            sumProfitFactor += res.profitFactor;
            sumAvgPnL += res.avgPnL;
            sumAvgTradeBars += res.avgTradeBars;
            sumMaxTradeGain += res.maxTradeGain;
            sumMaxTradeLoss += res.maxTradeLoss;
            sumScoreSwingTrade += res.scoreSwingTrade;
            sumTradeCount += res.tradeCount;

        }

        ComboMixResult bestCombo = null;
        List<ComboMixResult> nonOverfitResults = new ArrayList<>();
        for(ComboMixResult cbo :listeComboMixResult){
            // Mise à jour du meilleur combo global (fallback)
            if (bestCombo == null || cbo.getResult().getRendement() > bestCombo.getResult().getRendement()) {
                bestCombo = cbo;
            }
            if (!cbo.getResult().isFltredOut()) {
                nonOverfitResults.add(cbo);
            }
        }
        ComboMixResult bestComboNonOverfit = null;
        double bestPerfNonOverfit = Double.NEGATIVE_INFINITY;
        for (ComboMixResult cbo : nonOverfitResults) {
            if (cbo.getResult().getRendement() > bestPerfNonOverfit) {
                bestPerfNonOverfit = cbo.getResult().getRendement();
                bestComboNonOverfit = cbo;
            }
        }
        ComboMixResult finalBestCombo = bestComboNonOverfit != null ? bestComboNonOverfit : bestCombo;

        int n = foldResults.size();
        RiskResult aggResult = RiskResult.builder()
                .rendement(n > 0 ? sumRendement / n : Double.NEGATIVE_INFINITY)
                .maxDrawdown(n > 0 ? sumDrawdown / n : 0.0)
                .winRate(n > 0 ? sumWinRate / n : 0.0)
                .profitFactor(n > 0 ? sumProfitFactor / n : 0.0)
                .avgPnL(n > 0 ? sumAvgPnL / n : 0.0)
                .avgTradeBars(n > 0 ? sumAvgTradeBars / n : 0.0)
                .maxTradeGain(n > 0 ? sumMaxTradeGain / n : 0.0)
                .maxTradeLoss(n > 0 ? sumMaxTradeLoss / n : 0.0)
                .scoreSwingTrade(n > 0 ? sumScoreSwingTrade / n : 0.0)
                .tradeCount(sumTradeCount)
                .build();
        // Calcul du ratio d'overfitting
        double avgTrainPerf = trainPerformances.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double avgTestPerf = testPerformances.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double overfitRatio = avgTestPerf / (avgTrainPerf == 0.0 ? 1.0 : avgTrainPerf);
        boolean isOverfit = (overfitRatio < 0.7 || overfitRatio > 1.3);
        // --- Filtrage final sur le meilleur résultat ---
        boolean stable = aggResult != null && TradeUtils.isStableAndSimple(aggResult, filterConfig);
        if (aggResult != null) {
            aggResult.setFltredOut(!stable || isOverfit);
        }

        resultObj.inParams = finalBestCombo.inParams;
        resultObj.outParams = finalBestCombo.outParams;
        resultObj.result = finalBestCombo.getResult();
        resultObj.inStrategyNames = inCombo.stream().map(Class::getSimpleName).toList();
        resultObj.outStrategyNames = outCombo.stream().map(Class::getSimpleName).toList();
        resultObj.contextOptim = ParamsOptim.builder()
                .initialCapital(StrategieBackTest.INITIAL_CAPITAL)
                .riskPerTrade(StrategieBackTest.RISK_PER_TRADE)
                .stopLossPct(StrategieBackTest.STOP_LOSS_PCT)
                .takeProfitPct(StrategieBackTest.TAKE_PROFIL_PCT)
                .nbSimples(totalCount)
                .build();

        //fait un check final
        BarSeries checkSeries = fullSeries.getSubSeries(fullSeries.getBarCount() - testWindow, fullSeries.getBarCount());
        RiskResult checkResult = this.checkResultat(checkSeries, resultObj);
        resultObj.rendementSum  = resultObj.getResult().getRendement() + checkResult.getRendement();
        resultObj.rendementDiff = resultObj.getResult().getRendement() - checkResult.getRendement();
        resultObj.rendementScore = resultObj.rendementSum - (resultObj.rendementDiff > 0 ? resultObj.rendementDiff : -resultObj.rendementDiff);
        resultObj.check = checkResult;

        TradeUtils.log("BestCombinationResult (walk-forward) : " + resultObjToString(resultObj));
        return resultObj;
    }


    public List<BestCombinationResult> getBestPerfActions(Integer limit, String sort, Boolean filtered){
        String orderBy = (sort == null || sort.isBlank()) ? "rendement_score" : sort;

        String sql = "SELECT * FROM best_in_out_mix_strategy WHERE profit_factor <> 0 AND max_drawdown <> 0 AND win_rate < 1";
        if (filtered != null && filtered) {
            sql += " AND fltred_out = false";
        }
        sql += " ORDER BY " + orderBy + " DESC";
        if (limit != null && limit > 0) {
            sql += " LIMIT " + limit;
        }

        List<BestCombinationResult> results = jdbcTemplate.query(sql, (rs, rowNum) -> {
            BestCombinationResult result = new BestCombinationResult();
            result.symbol = rs.getString("symbol");
            result.inStrategyNames = gson.fromJson(rs.getString("in_strategy_names"), new TypeToken<List<String>>(){}.getType());
            result.outStrategyNames = gson.fromJson(rs.getString("out_strategy_names"), new TypeToken<List<String>>(){}.getType());
            result.inParams = gson.fromJson(rs.getString("in_params"), new TypeToken<Map<String, Object>>(){}.getType());
            result.outParams = gson.fromJson(rs.getString("out_params"), new TypeToken<Map<String, Object>>(){}.getType());
            result.result = RiskResult.builder()
                    .rendement(rs.getDouble("rendement"))
                    .maxDrawdown(rs.getDouble("max_drawdown"))
                    .tradeCount(rs.getInt("trade_count"))
                    .winRate(rs.getDouble("win_rate"))
                    .avgPnL(rs.getDouble("avg_pnl"))
                    .profitFactor(rs.getDouble("profit_factor"))
                    .avgTradeBars(rs.getDouble("avg_trade_bars"))
                    .maxTradeGain(rs.getDouble("max_trade_gain"))
                    .maxTradeLoss(rs.getDouble("max_trade_loss"))
                    .scoreSwingTrade(rs.getDouble("score_swing_trade"))
                    .fltredOut(rs.getBoolean("fltred_out")).build();
            result.contextOptim = ParamsOptim.builder()
                    .initialCapital(rs.getDouble("initial_capital"))
                    .riskPerTrade(rs.getDouble("risk_per_trade"))
                    .stopLossPct(rs.getDouble("stop_loss_pct"))
                    .takeProfitPct(rs.getDouble("take_profit_pct"))
                    .nbSimples(rs.getInt("nb_simples"))
                    .build();
            result.check= new com.google.gson.Gson().fromJson(rs.getString("check_result"), RiskResult.class);
            result.rendementSum = rs.getDouble("rendement_sum");
            result.rendementDiff = rs.getDouble("rendement_diff");
            result.rendementScore = rs.getDouble("rendement_score");
            return result;
        });
        return results;
    }


    public RiskResult checkResultat(BarSeries checkSeries, BestCombinationResult resultObj){
        List<TradeStrategy> inStrategiesCheck = new ArrayList<>();
        for (String name : resultObj.inStrategyNames) {
            Object params = resultObj.inParams.get(name.replace("Strategy", ""));
            if (name.equals("ImprovedTrendFollowingStrategy") && params instanceof StrategieBackTest.ImprovedTrendFollowingParams) {
                StrategieBackTest.ImprovedTrendFollowingParams p = (StrategieBackTest.ImprovedTrendFollowingParams) params;
                inStrategiesCheck.add(new ImprovedTrendFollowingStrategy(p.trendPeriod, p.shortMaPeriod, p.longMaPeriod, p.breakoutThreshold, p.useRsiFilter, p.rsiPeriod));
            } else if (name.equals("SmaCrossoverStrategy") && params instanceof StrategieBackTest.SmaCrossoverParams) {
                StrategieBackTest.SmaCrossoverParams p = (StrategieBackTest.SmaCrossoverParams) params;
                inStrategiesCheck.add(new SmaCrossoverStrategy(p.shortPeriod, p.longPeriod));
            } else if (name.equals("RsiStrategy") && params instanceof StrategieBackTest.RsiParams) {
                StrategieBackTest.RsiParams p = (StrategieBackTest.RsiParams) params;
                inStrategiesCheck.add(new RsiStrategy(p.rsiPeriod, p.oversold, p.overbought));
            } else if (name.equals("BreakoutStrategy") && params instanceof StrategieBackTest.BreakoutParams) {
                StrategieBackTest.BreakoutParams p = (StrategieBackTest.BreakoutParams) params;
                inStrategiesCheck.add(new BreakoutStrategy(p.lookbackPeriod));
            } else if (name.equals("MacdStrategy") && params instanceof StrategieBackTest.MacdParams) {
                StrategieBackTest.MacdParams p = (StrategieBackTest.MacdParams) params;
                inStrategiesCheck.add(new MacdStrategy(p.shortPeriod, p.longPeriod, p.signalPeriod));
            } else if (name.equals("MeanReversionStrategy") && params instanceof StrategieBackTest.MeanReversionParams) {
                StrategieBackTest.MeanReversionParams p = (StrategieBackTest.MeanReversionParams) params;
                inStrategiesCheck.add(new MeanReversionStrategy(p.smaPeriod, p.threshold));
            }
        }
        List<TradeStrategy> outStrategiesCheck = new ArrayList<>();
        for (String name : resultObj.outStrategyNames) {
            Object params = resultObj.outParams.get(name.replace("Strategy", ""));
            if (name.equals("ImprovedTrendFollowingStrategy") && params instanceof StrategieBackTest.ImprovedTrendFollowingParams) {
                StrategieBackTest.ImprovedTrendFollowingParams p = (StrategieBackTest.ImprovedTrendFollowingParams) params;
                outStrategiesCheck.add(new ImprovedTrendFollowingStrategy(p.trendPeriod, p.shortMaPeriod, p.longMaPeriod, p.breakoutThreshold, p.useRsiFilter, p.rsiPeriod));
            } else if (name.equals("SmaCrossoverStrategy") && params instanceof StrategieBackTest.SmaCrossoverParams) {
                StrategieBackTest.SmaCrossoverParams p = (StrategieBackTest.SmaCrossoverParams) params;
                outStrategiesCheck.add(new SmaCrossoverStrategy(p.shortPeriod, p.longPeriod));
            } else if (name.equals("RsiStrategy") && params instanceof StrategieBackTest.RsiParams) {
                StrategieBackTest.RsiParams p = (StrategieBackTest.RsiParams) params;
                outStrategiesCheck.add(new RsiStrategy(p.rsiPeriod, p.oversold, p.overbought));
            } else if (name.equals("BreakoutStrategy") && params instanceof StrategieBackTest.BreakoutParams) {
                StrategieBackTest.BreakoutParams p = (StrategieBackTest.BreakoutParams) params;
                outStrategiesCheck.add(new BreakoutStrategy(p.lookbackPeriod));
            } else if (name.equals("MacdStrategy") && params instanceof StrategieBackTest.MacdParams) {
                StrategieBackTest.MacdParams p = (StrategieBackTest.MacdParams) params;
                outStrategiesCheck.add(new MacdStrategy(p.shortPeriod, p.longPeriod, p.signalPeriod));
            } else if (name.equals("MeanReversionStrategy") && params instanceof StrategieBackTest.MeanReversionParams) {
                StrategieBackTest.MeanReversionParams p = (StrategieBackTest.MeanReversionParams) params;
                outStrategiesCheck.add(new MeanReversionStrategy(p.smaPeriod, p.threshold));
            }
        }
        // Combiner les règles d'entrée et de sortie pour le check
        Rule entryRuleCheck = null;
        Rule exitRuleCheck = null;
        for (TradeStrategy strat : inStrategiesCheck) {
            if (entryRuleCheck == null) entryRuleCheck = strat.getEntryRule(checkSeries);
            else entryRuleCheck = entryRuleCheck.or(strat.getEntryRule(checkSeries));
        }
        for (TradeStrategy strat : outStrategiesCheck) {
            if (exitRuleCheck == null) exitRuleCheck = strat.getExitRule(checkSeries);
            else exitRuleCheck = exitRuleCheck.or(strat.getExitRule(checkSeries));
        }
        final Rule finalEntryRuleCheck = entryRuleCheck;
        final Rule finalExitRuleCheck = exitRuleCheck;
        TradeStrategy combinedCheckStrategy = new TradeStrategy() {
            @Override
            public Rule getEntryRule(BarSeries s) { return finalEntryRuleCheck; }
            @Override
            public Rule getExitRule(BarSeries s) { return finalExitRuleCheck; }
            @Override
            public String getName() { return "CombinedCheckStrategy"; }
        };
        StrategieBackTest backTestCheck = new StrategieBackTest();
        return backTestCheck.backtestStrategy(combinedCheckStrategy, checkSeries);
    }

}

