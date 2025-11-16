package com.app.backend.trade.portfolio.learning;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Loss personnalisée utilitaire (MSE + turnover + drawdown) pour évaluation.
 * Note: non utilisée directement par OutputLayer pour compatibilité; voir trainer.customValLoss.
 */
public class LossPortfolioAllocation {
    private final double lambdaTurnover;
    private final double lambdaDrawdown;

    public LossPortfolioAllocation(double lambdaTurnover, double lambdaDrawdown) {
        this.lambdaTurnover = lambdaTurnover;
        this.lambdaDrawdown = lambdaDrawdown;
    }

    private org.nd4j.linalg.indexing.INDArrayIndex interval(int startInclusive, int endExclusive) {
        return org.nd4j.linalg.indexing.NDArrayIndex.interval(startInclusive, endExclusive);
    }

    public double computeScore(INDArray labels, INDArray predictions) {
        INDArray y = labels.rank()==2 && labels.size(1)==1 ? labels.reshape(labels.size(0)) : labels;
        INDArray p = predictions.rank()==2 && predictions.size(1)==1 ? predictions.reshape(predictions.size(0)) : predictions;
        int n = (int) p.length();
        INDArray diff = p.sub(y);
        double mse = diff.mul(diff).meanNumber().doubleValue();
        double turnover = 0.0;
        if (n > 1) {
            INDArray shifted = p.get(interval(1, n));
            INDArray prev = p.get(interval(0, n - 1));
            INDArray turnoverVec = Transforms.abs(shifted.sub(prev), false);
            turnover = turnoverVec.meanNumber().doubleValue();
        }
        INDArray reluNeg = Transforms.relu(y.neg(), false);
        double drawdownPenalty = reluNeg.mul(p.mul(p)).meanNumber().doubleValue();
        return mse + lambdaTurnover * turnover + lambdaDrawdown * drawdownPenalty;
    }
}
