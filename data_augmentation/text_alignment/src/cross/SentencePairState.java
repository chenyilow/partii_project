package cross;

import fig.basic.*;
import static fig.basic.LogInfo.*;
import java.util.*;

/**
 * Performs operations with respect to a particular pair of aligned sentences.
 */
public abstract class SentencePairState {
  public abstract static class Factory {
    public abstract SentencePairState create(List<String> enWords, List<String> frWords, EMWordAligner wa);
    public abstract TrainingCache createCache();
    public abstract String getName();
  }

  public SentencePairState(List<String> enWords, List<String> frWords, EMWordAligner wa) {
    this.enWords = enWords;
    this.frWords = frWords;
    this.wa = wa;
    I = enWords.size();
    J = frWords.size();
    likelihood = Double.NaN;
  }

  String en(int i) { return i == I ? wa.nullWord : enWords.get(i); }
  String fr(int j) { return frWords.get(j); }

  public abstract ExpAlign computeExpAlign();
  public abstract void updateNewParams(ExpAlign expAlign);

  // Two types of decoding: posterior and viterbi
  public abstract Alignment getViterbi(boolean reverse);
  // If !reverse, return a JxI matrix which is the posterior probability of an alignment (i,j)
  // If reverse, return an IxJ matrix ...
  public double[][] getPosteriors(boolean reverse) {
    ExpAlign expAlign = computeExpAlign();
    // Throw away null
    double[][] posteriors = new double[J][I];
    for(int j = 0; j < J; j++)
      for(int i = 0; i < I; i++)
        posteriors[j][i] = expAlign.get(j, i);
    if(reverse) return NumUtils.transpose(posteriors);
    return posteriors;
  }

  // pos[j] = position i
  public double getLikelihood(int[] pos) {
    throw new UnsupportedOperationException();
  }

  public double logLikelihood() { return Math.log(likelihood); }

  public void updateTransProbs(ExpAlign expAlign) {
    for(int j = 0; j < J; j++) {
      String v = fr(j);
      for(int i = 0; i <= I; i++) {
        String u = en(i);
        double p = expAlign.get(j, i);
        NumUtils.assertIsFinite(p);
        if(Main.useNormedObjective) p /= I*J;
        wa.newParams.transProbs.incr(u, v, p);
      }
    }
  }

  List<String> enWords, frWords;
  EMWordAligner wa;
  int I, J; // Length of English and French words
  double likelihood; // Computed when computeExpAlign() is called
}
