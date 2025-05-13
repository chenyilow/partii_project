package cross;

import fig.basic.*;
import java.util.*;

public class IntersectedWordAligner extends WordAligner {
  WordAligner wa1, wa2;
  boolean usePosteriorDecodingFlag;
  double posteriorDecodingThreshold;
  // How to combine two posterior numbers.
  public enum CombineMethod { multiply, add, min, max };
  public CombineMethod combineMethod = CombineMethod.multiply;

  public IntersectedWordAligner(WordAligner wa1, WordAligner wa2) {
    this.wa1 = wa1;
    this.wa2 = wa2;
    this.modelPrefix = "intersect-" + wa1.modelPrefix + "+" + wa2.modelPrefix;
  }

  // In order to use posterior decoding,
  // the alignment routines for the two word aligners must set
  // the strengths of the alignments to the posteriors
  public void usePosteriorDecoding(double threshold) {
    usePosteriorDecodingFlag = true;
    posteriorDecodingThreshold = threshold;
  }

  public Alignment alignSentencePair(SentencePair sp) {
    return alignSentencePairReturnAll(sp).get(2);
  }

  public List<Alignment> alignSentencePairReturnAll(SentencePair sp) {
    Alignment a1 = wa1.alignSentencePair(sp);
    Alignment a2 = wa2.alignSentencePair(sp);
    Alignment a3;

    if(!usePosteriorDecodingFlag)
      a3 = a1.intersect(a2);
    else {
      // Do posterior decoding
      int I = sp.getEnglishWords().size();
      int J = sp.getFrenchWords().size();
      double[][] posteriors = new double[J][I];
      if(EMWordAligner.exactJointExp) {
        EMWordAligner wa1 = (EMWordAligner)this.wa1;
        EMWordAligner wa2 = (EMWordAligner)this.wa2;
        ComputeExactExpAlign c = new ComputeExactExpAlign(wa1.newSentencePairState(sp),
                                                          wa2.newSentencePairState(sp));
        c.compute();
        ExpAlign expAlign = c.getExpAlign1();
        for(int j = 0; j < J; j++)
          for(int i = 0; i < I; i++)
            posteriors[j][i] = expAlign.get(j, i);
      }
      else {
        for(int j = 0; j < J; j++)
          for(int i = 0; i < I; i++)
            posteriors[j][i] = combine(a1.getStrength(i, j), a2.getStrength(i, j));
      }

      a3 = Alignment.thresholdPosteriors(posteriors, posteriorDecodingThreshold);
    }

    return ListUtils.newList(a1, a2, a3);
  }

  double combine(double a, double b) {
    switch(combineMethod) {
      case add: return a+b;
      case multiply: return a*b;
      case min: return Math.min(a, b);
      case max: return Math.max(a, b);
      default: throw Exceptions.unknownCase;
    }
  }

  public String getName() {
    String s = usePosteriorDecodingFlag ? ", posteriorThreshold=" + posteriorDecodingThreshold : "";
    return "Intersect(" + wa1.getName() + ", " + wa2.getName() + s + ")";
  }
}

