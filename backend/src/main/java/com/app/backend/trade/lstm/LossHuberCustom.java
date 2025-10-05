package com.app.backend.trade.lstm;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.common.primitives.Pair; // import correct pour Pair
import org.nd4j.shade.jackson.annotation.JsonCreator;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Perte Huber (Smooth L1) custom pour DL4J 1.0.0-M2.1.
 */
public class LossHuberCustom implements ILossFunction {
    private final double delta;

    // Constructeur JSON rétrocompatible: si delta manquant ou <=0, fallback 1.0
    @JsonCreator
    public LossHuberCustom(@JsonProperty("delta") Double delta){
        if (delta == null || delta <= 0) {
            this.delta = 1.0; // valeur par défaut
        } else {
            this.delta = delta;
        }
    }

    // Constructeur no-arg (utilisation directe éventuelle)
    public LossHuberCustom(){
        this.delta = 1.0;
    }

    public double getDelta(){
        return delta;
    }

    private INDArray activated(INDArray preOutput, IActivation activationFn){
        return activationFn.getActivation(preOutput.dup(), true);
    }

    private INDArray error(INDArray labels, INDArray preOutput, IActivation activationFn){
        return activated(preOutput, activationFn).sub(labels); // y_pred - y_true
    }

    private INDArray huberElementWise(INDArray e){
        INDArray abs = Transforms.abs(e, true);
        INDArray quadMask = abs.lte(delta); // bool 0/1
        INDArray quad = e.mul(e).muli(0.5).muli(quadMask);
        INDArray linMask = abs.gt(delta);
        INDArray lin = abs.sub(delta * 0.5).muli(delta).muli(linMask);
        return quad.addi(lin);
    }

    private INDArray huberGradient(INDArray e){
        INDArray abs = Transforms.abs(e, true);
        INDArray quadMask = abs.lte(delta);
        INDArray linMask = abs.gt(delta);
        INDArray gradQuad = e.mul(quadMask);
        INDArray gradLin = Transforms.sign(e, true).muli(delta).muli(linMask);
        return gradQuad.addi(gradLin);
    }

    private INDArray applyMask(INDArray arr, INDArray mask){
        if (mask == null) return arr;
        return arr.mul(mask); // broadcast si nécessaire
    }

    @Override
    public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
        INDArray e = error(labels, preOutput, activationFn);
        INDArray loss = huberElementWise(e);
        loss = applyMask(loss, mask);
        double s = loss.sumNumber().doubleValue();
        if (average) s /= labels.size(0);
        return s;
    }

    @Override
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray e = error(labels, preOutput, activationFn);
        INDArray loss = huberElementWise(e);
        loss = applyMask(loss, mask);
        // Réduit sur dimensions restantes pour obtenir vecteur batch
        if (loss.rank() > 2) {
            long b = loss.size(0);
            return loss.reshape(b, -1).sum(1);
        } else if (loss.rank() == 2) {
            return loss.sum(1);
        }
        return loss; // vecteur ou scalaire
    }

    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray e = error(labels, preOutput, activationFn);
        INDArray dLdy = huberGradient(e); // gradient w.r.t y_pred (activé)
        dLdy = applyMask(dLdy, mask);
        // Backprop activation
        return activationFn.backprop(preOutput.dup(), dLdy).getFirst();
    }

    @Override
    public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
        double score = computeScore(labels, preOutput, activationFn, mask, average);
        INDArray grad = computeGradient(labels, preOutput, activationFn, mask);
        return Pair.of(score, grad);
    }

    @Override
    public String name() { return "LossHuberCustom(delta=" + delta + ")"; }

    @Override
    public String toString(){ return name(); }
}
