package com.app.backend.trade.portfolio.learning;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.common.primitives.Pair;

/**
 * Implémentation ILossFunction pour MSE + pénalité de turnover + pénalité drawdown.
 * Compatible avec DL4J (utilise preOutput + activationFn).
 */
public class PortfolioCustomLoss implements ILossFunction {

    private final double lambdaTurnover;
    private final double lambdaDrawdown;

    public PortfolioCustomLoss(double lambdaTurnover, double lambdaDrawdown) {
        this.lambdaTurnover = lambdaTurnover;
        this.lambdaDrawdown = lambdaDrawdown;
    }

    private INDArrayIndex interval(int startInclusive, int endExclusive) {
        return NDArrayIndex.interval(startInclusive, endExclusive);
    }

    private INDArray activate(INDArray preOut, IActivation activationFn) {
        INDArray act = preOut.dup();
        activationFn.getActivation(act, true);
        return act;
    }

    @Override
    public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
        INDArray p = activate(preOutput, activationFn);
        INDArray y = labels;
        // Aplatissement si [n,1]
        if (p.rank()==2 && p.size(1)==1) p = p.reshape(p.size(0));
        if (y.rank()==2 && y.size(1)==1) y = y.reshape(y.size(0));
        int n = (int) p.length();
        if (n == 0) return 0.0;

        // MSE
        INDArray diff = p.sub(y);
        INDArray mseArr = diff.mul(diff);

        // Turnover (paires adjacentes)
        double turnover = 0.0;
        if (n > 1) {
            INDArray shifted = p.get(interval(1, n));
            INDArray prev = p.get(interval(0, n - 1));
            INDArray turnoverVec = Transforms.abs(shifted.sub(prev), false);
            turnover = turnoverVec.meanNumber().doubleValue();
        }

        // Drawdown penalty
        INDArray reluNeg = Transforms.relu(y.neg(), false);
        INDArray drawArr = reluNeg.mul(p.mul(p));

        double mse = mseArr.meanNumber().doubleValue();
        double draw = drawArr.meanNumber().doubleValue();
        double score = mse + lambdaTurnover * turnover + lambdaDrawdown * draw;

        if (mask != null) {
            // Appliquer masque en moyenne simple sur MSE + draw; turnover reste global sur batch
            INDArray m = mask;
            if (m.rank()==2 && m.size(1)==1) m = m.reshape(m.size(0));
            double maskedCount = m.sumNumber().doubleValue();
            if (maskedCount > 0) {
                double mseMasked = mseArr.mul(m).sumNumber().doubleValue() / maskedCount;
                double drawMasked = drawArr.mul(m).sumNumber().doubleValue() / maskedCount;
                score = mseMasked + lambdaTurnover * turnover + lambdaDrawdown * drawMasked;
            }
        }
        if (!average) {
            // score total (somme) au lieu de moyenne
            score *= n;
        }
        return score;
    }

    @Override
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray p = activate(preOutput, activationFn);
        INDArray y = labels;
        if (p.rank()==2 && p.size(1)==1) p = p.reshape(p.size(0));
        if (y.rank()==2 && y.size(1)==1) y = y.reshape(y.size(0));
        int n = (int) p.length();
        if (n == 0) return Nd4j.create(0);

        INDArray diff = p.sub(y);
        INDArray mseArr = diff.mul(diff);

        INDArray perExampleTurn = Nd4j.zerosLike(p);
        if (n > 1) {
            INDArray shifted = p.get(interval(1, n));
            INDArray prev = p.get(interval(0, n - 1));
            INDArray pair = Transforms.abs(shifted.sub(prev), false);
            // répartir moitié-moitié sur les 2 exemples concernés
            perExampleTurn.get(interval(0, n - 1)).addi(pair.mul(0.5));
            perExampleTurn.get(interval(1, n)).addi(pair.mul(0.5));
            // moyenne sera gérée en amont (ici c'est contribution brute par exemple)
            perExampleTurn.divi(Math.max(1, n - 1));
        }

        INDArray reluNeg = Transforms.relu(y.neg(), false);
        INDArray drawArr = reluNeg.mul(p.mul(p));

        INDArray perExample = mseArr.addi(drawArr.muli(lambdaDrawdown)).addi(perExampleTurn.muli(lambdaTurnover));

        if (mask != null) {
            INDArray m = mask;
            if (m.rank()==2 && m.size(1)==1) m = m.reshape(m.size(0));
            perExample.muli(m);
        }
        return perExample;
    }

    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        // p = activation(preOutput)
        INDArray p = activate(preOutput, activationFn);
        INDArray y = labels;
        if (p.rank()==2 && p.size(1)==1) p = p.reshape(p.size(0));
        if (y.rank()==2 && y.size(1)==1) y = y.reshape(y.size(0));
        int n = (int) p.length();
        if (n == 0) return Nd4j.zerosLike(preOutput);

        // dL/dp pour chaque composant
        INDArray gradOut = p.sub(y).mul(2.0 / Math.max(1, n)); // MSE gradient

        // Drawdown gradient: (2/n) * relu(-y) * p
        INDArray reluNeg = Transforms.relu(y.neg(), false);
        gradOut.addi(reluNeg.mul(p).mul(2.0 * lambdaDrawdown / Math.max(1, n)));

        // Turnover gradient (approx via signe des différences adjacentes)
        if (n > 1 && lambdaTurnover != 0.0) {
            INDArray gTurn = Nd4j.zerosLike(p);
            INDArray diffNext = p.get(interval(1, n)).sub(p.get(interval(0, n - 1)));
            INDArray sNext = Transforms.sign(diffNext, false); // sign(p_i - p_{i-1}) for i>=1
            // i = 0: -(sign(p1 - p0))
            gTurn.get(interval(0, 1)).subi(sNext.get(interval(0, 1)));
            // middle: sign(p_i - p_{i-1}) - sign(p_{i+1} - p_i)
            if (n > 2) {
                INDArray sNextShift = Transforms.sign(p.get(interval(2, n)).sub(p.get(interval(1, n - 1))), false);
                gTurn.get(interval(1, n - 1)).addi(sNext.get(interval(0, n - 2))).subi(sNextShift);
            }
            // i = n-1: +sign(p_{n-1} - p_{n-2})
            gTurn.get(interval(n - 1, n)).addi(sNext.get(interval(n - 2, n - 1)));
            gTurn.divi(Math.max(1, n - 1));
            gradOut.addi(gTurn.muli(lambdaTurnover));
        }

        // Propager via dérivée d'activation pour obtenir dL/dZ (mêmes dimensions que preOutput)
        INDArray gradOutReshaped = gradOut.reshape(preOutput.shape());
        Pair<INDArray, INDArray> bp = activationFn.backprop(preOutput.dup(), gradOutReshaped);
        INDArray gradPreOut = bp.getFirst();

        if (mask != null) {
            gradPreOut.muli(mask);
        }
        return gradPreOut;
    }

    @Override
    public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
        double score = computeScore(labels, preOutput, activationFn, mask, average);
        INDArray grad = computeGradient(labels, preOutput, activationFn, mask);
        return Pair.of(score, grad);
    }

    @Override
    public String name() {
        return "PortfolioCustomLoss";
    }
}
