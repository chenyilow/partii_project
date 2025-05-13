package cross;

import fig.basic.*;
import static fig.basic.LogInfo.*;
import java.util.*;

public class Model1SentencePairState extends SentencePairState {
  public static class Factory extends SentencePairState.Factory {
    public SentencePairState create(List<String> enWords, List<String> frWords, EMWordAligner wa) {
      return new Model1SentencePairState(enWords, frWords, wa);
    }
    public TrainingCache createCache() { return new TrainingCache(); }
    public String getName() { return "Model1"; }
  }

  public Model1SentencePairState(List<String> enWords, List<String> frWords, EMWordAligner wa) {
    super(enWords, frWords, wa);

    // The null-word probability is either the same as the probability
    // of any other position in the sentence or a fixed probability
    // independent of the length of the sentence.
    nullProb = (EMWordAligner.nullProb == 1 ?
        1.0/(I+1) : EMWordAligner.nullProb);
  }

  // Return P(a_j = i | f, e)
  double alignProb(int j, int i) {
    return (i == I ? nullProb : (1-nullProb)/I);
  }

  // Compute expected alignments for a particular sentence
  public ExpAlign computeExpAlign() {
    double[][] expAlign = new double[J][I+1];

    likelihood = 1;
    for(int j = 0; j < J; j++) {
      // Compute P(a_j | f, e) \propto P(a_j, f | e) = P(a_j) P(f_j | e_{a_j})
      String v = fr(j);
      double sum = 0;
      for(int i = 0; i <= I; i++) {
        String u = en(i);
        if(EMWordAligner.handleUnknownWords)
          expAlign[j][i] = alignProb(j, i) * wa.params.transProbs.get(u, v, 0);
        else
          expAlign[j][i] = alignProb(j, i) * wa.params.transProbs.getSure(u, v);
        sum += expAlign[j][i];
      }

      // Normalize
      if(sum == 0) {
        // Can't normalize, just zero everything instead of blowing up
        for(int i = 0; i <= I; i++)
          expAlign[j][i] = 0;
        warning("Sum of expected counts = 0, can't normalize (I=%d,J=%d,j=%d)", I, J, j);
      }
      else {
        for(int i = 0; i <= I; i++)
          expAlign[j][i] /= sum;
        likelihood *= sum;
      }

      //for(int i = 0; i <= I; i++)
        //dbg("expAlign[%d:%s][%d:%s] = %f", j, fr(j), i, en(i), expAlign[j][i]);
    }

    // Doesn't work!  Probabilities are too coarse?
    /*if(wa.hardEM) {
      Random rand = new Random();
      // Make the expectations hard (0, 1)
      for(int j = 0; j < J; j++) {
        int besti = rand.nextInt(I+1);
        for(int i = 0; i <= I; i++)
          if(expAlign[j][i] > expAlign[j][besti]) besti = i;
        for(int i = 0; i <= I; i++) {
          if(2*expAlign[j][i] < expAlign[j][besti])
          //if(i != besti)
            expAlign[j][i] = 0;
        }
        //logs(j + " " + besti + ": " + expAlign[j][besti]);
      }
    }*/

    return new Model1ExpAlign(expAlign);
  }

  /**
   * Update the word aligner's new translation parameters.
   */
  public void updateNewParams(ExpAlign expAlign) {
    // Translation parameters
    updateTransProbs(expAlign);

    // Distortion parameters: don't change
    wa.newParams.distortProbs.set(wa.params.distortProbs);
  }

  public Alignment getViterbi(boolean reverse) {
    Alignment alignment = new Alignment();
    //Random rand = new Random();
    for(int j = 0; j < J; j++) {
      // For each a_j, simply pick the maximum
      int besti = -1;
      double bestp = -1;

      String v = fr(j);
      for(int i = 0; i <= I; i++) {
        String u = en(i);
        double p = alignProb(j, i) * wa.params.transProbs.getWithErrorMsg(u, v, 0);
        if(i != I) {
          int realI = reverse ? j : i;
          int realJ = reverse ? i : j;
          alignment.setStrength(realI, realJ, p);
        }
        if(p > bestp) {
          bestp = p;
          besti = i;
        }
      }

      assert besti != -1;
      if(besti == I) continue; // Skip NULL
      if(!reverse)
        alignment.addAlignment(besti, j, true);
      else
        alignment.addAlignment(j, besti, true);
    }
    return alignment;
  }

  public double getLikelihood(int[] pos) {
    double likelihood = 1;
    for(int j = 0; j < J; j++) {
      String v = fr(j);
      // Compute P(a_j | f, e) \propto P(a_j, f | e) = P(a_j) P(f_j | e_{a_j})
      int i = pos[j];
      String u = en(i);
      if(EMWordAligner.handleUnknownWords)
        likelihood *= alignProb(j, i) * wa.params.transProbs.get(u, v, 0);
      else
        likelihood *= alignProb(j, i) * wa.params.transProbs.getSure(u, v);
    }
    return likelihood;
  }

  double nullProb; // Specific to this sentence
}
