package cross;

import fig.basic.*;
import static fig.basic.LogInfo.*;
import java.io.*;
import java.util.*;

/** Models distortion.
 */
public class DistortProbTable implements Serializable {
  static final long serialVersionUID = 42;

  // P(a_j | a_{j-1}) consists of 2*windowSize+1 buckets
  // Anything outside the window gets mapped on to the closest bucket.
  // Also condition on a state.
  public static final int windowSize = 5;

  static int numStates = 0;
  public static void setNumStates(int _numStates) { numStates = _numStates; }

  private double[][] probs = null;
  private transient double[][] sums = null; // Partial sums (for normalization)
  // sums[state][k] = probs[state][0] + ... + probs[state][k-1]

  public DistortProbTable() { alloc(); }

  private void alloc() {
    if(numStates > 0) {
      probs = new double[numStates][2*windowSize+1];
      sums  = new double[numStates][2*windowSize+1+1];
    }
  }

  public void set(DistortProbTable table) {
    if(probs == null) return;
    probs = NumUtils.copy(table.probs);
    sums = NumUtils.copy(table.sums);
  }

  public DistortProbTable copy() {
    DistortProbTable table = new DistortProbTable();
    table.set(this);
    return table;
  }

  public int numStates() { return probs == null ? 0 : probs.length; }

  private double computeNorm(int state, int i, int I) {
    int mind = Math.max(0-i, -windowSize);
    int maxd = Math.min(I-i, windowSize);
    assert mind <= maxd : String.format("mind=%d,maxd=%d", mind, maxd);
    // Return probs[state][mind] + ... + probs[state][maxd]
    return sums[state][maxd+windowSize+1] - sums[state][mind+windowSize];
  }

  // P(a_j = i | a_{j-1} = h) (Assume nulls are taken care of elsewhere)
  // Current position in English is i
  // Previous position in English is h
  // Independent of current position in French j
  // I = length of English sentence
  // If we're in the fringe buckets, split the probability uniformly
  public double get(int state, int h, int i, int I) {
    int d = i-h;
    // div = Number of positions i out of [0, I]
    // that share this bucket d for this given h
    // (so we need to split the probability among these h)
    int div;
    if(d <= -windowSize) {
      d = -windowSize;
      div = (h-windowSize)-0 + 1;
    }
    else if(d >= windowSize) {
      d = windowSize;
      div = I-(h+windowSize) + 1;
    }
    else div = 1;
    double norm = computeNorm(state, h, I);
    if(norm == 0) return 0;
    assert div > 0 :
      String.format("getDDiv: state=%d, h=%d, i=%d, I=%d: div=%d, norm=%f/%f", state, h, i, I, div, norm, sums[state][2*windowSize+1]);
    return probs[state][d+windowSize] / div / norm;
  }

  public void add(int state, int h, int i, int I, double count) {
    //dbg("DistortProbTable.add state=%d, %d -> %d: %f", state, h, i, count);
    int d = i-h;
    if(d < -windowSize) d = -windowSize;
    else if(d > windowSize) d = windowSize;
    // Even if we are using a scaled version of the parameter probs[state][...]
    // by the number of divisions, the maximum likelihood estimate of that
    // parameter does not scale the count.
    probs[state][d+windowSize] += count; ///d_div.second;
  }

  public void computeSums() {
    if(sums == null) sums = new double[numStates][2*windowSize+1+1];
    for(int state = 0; state < probs.length; state++) {
      Arrays.fill(sums[state], 0);
      for(int k = 0; k < probs[state].length; k++)
        sums[state][k+1] = sums[state][k] + probs[state][k];
    }
  }

  public void normalize() {
    if(probs == null) return;
    for(int i = 0; i < probs.length; i++) {
      if(!NumUtils.normalizeForce(probs[i]))
        error("normalize(): distortProbs(state=%d) has sum 0, using uniform", i);
    }
    computeSums();
  }
  public void initUniform() {
    if(probs == null) return;
    for(double[] p : probs) Arrays.fill(p, 1.0/p.length);
    computeSums();
  }
  public void initZero() {
    if(probs == null) return;
    for(double[] p : probs) Arrays.fill(p, 0.0);
    computeSums();
  }

  public void dump(PrintWriter out) {
    for(int state = 0; state < numStates(); state++) {
      for(int i = -windowSize; i <= windowSize; i++) {
        String s;
        if(i == -windowSize) s = "<= " + i;
        else if(i == windowSize) s = ">= " + i;
        else s = "= " + i;
        out.println(state + ": " + s + "\t" + probs[state][i+windowSize]);
      }
    }
  }

  private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
    HMMSentencePairState.StateType currStateType = HMMSentencePairState.stateType;
    HMMSentencePairState.StateType readStateType = (HMMSentencePairState.StateType)in.readObject();
    if(readStateType == HMMSentencePairState.StateType.NONE) {
      alloc();
      initUniform();
    }
    else if(readStateType == currStateType) {
      probs = (double[][])in.readObject();
      computeSums();
    }
    else
      throw new RuntimeException("Mis-match in HMM state types: read "
          + readStateType + ", but running with " + currStateType);

  }

  private void writeObject(ObjectOutputStream out) throws IOException {
    HMMSentencePairState.StateType currStateType = HMMSentencePairState.stateType;
    if(probs == null) currStateType = HMMSentencePairState.StateType.NONE;
    out.writeObject(currStateType);
    out.writeObject(probs);
  }
}
