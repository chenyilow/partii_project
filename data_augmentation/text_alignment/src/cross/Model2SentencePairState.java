package cross;

import fig.basic.*;
import static fig.basic.LogInfo.*;
import java.util.*;

public class Model2SentencePairState extends Model1SentencePairState {
  public static class Factory extends SentencePairState.Factory {
    public SentencePairState create(List<String> enWords, List<String> frWords, EMWordAligner wa) {
      return new Model2SentencePairState(enWords, frWords, wa);
    }
    public TrainingCache createCache() { return new TrainingCache(); }
    public String getName() { return "Model2"; }
  }

  public Model2SentencePairState(List<String> enWords, List<String> frWords, EMWordAligner wa) {
    super(enWords, frWords, wa);
  }

  // Return the diagonal in I
  int diag(int j) { return I*j/J; }

  // Return P(a_j = i | f, e)
  double alignProb(int j, int i) {
    if(i == I) return nullProb;
    else {
      int diagi = diag(j);
      return (1-nullProb) * wa.params.distortProbs.get(0, i, diagi, I);
    }
  }

  /**
   * Update the word aligner's new translation parameters.
   */
  public void updateNewParams(ExpAlign expAlign) {
    // Translation parameters
    updateTransProbs(expAlign);

    // Distortion parameters
    for(int j = 0; j < J; j++) {
      for(int i = 0; i < I; i++) {
        int diagi = diag(j);
        wa.newParams.distortProbs.add(0, i, diagi, I, expAlign.get(j, i));
      }
    }
  }
}
