package cross;

import fig.basic.*;
import java.util.*;
import static fig.basic.LogInfo.*;

public class ComputeExactExpAlign {
  private SentencePairState sps1, sps2;
  private ExpAlign expAlign1, expAlign2;
  private int[] j2i, i2j;
  private int J, I;
  private double total;
  private double[][] table1, table2;

  public ComputeExactExpAlign(SentencePairState sps1, SentencePairState sps2) {
    this.sps1 = sps1;
    this.sps2 = sps2;
  }

  private void checkAndDump() {
    for(int j = 0; j < J; j++) {
      if(j2i[j] < I) assert i2j[j2i[j]] == j;
    }
    for(int i = 0; i < I; i++) {
      if(i2j[i] < J) assert j2i[i2j[i]] == i;
    }
    StringBuffer sb = new StringBuffer();
    for(int j = 0; j < J; j++) {
      int i = j2i[j];
      if(i < I) sb.append(j + "-" + i + " ");
    }
    logs(sb.toString());
  }

  private void search(int j) {
    if(j == J) {
      double l1 = sps1.getLikelihood(j2i);
      double l2 = sps2.getLikelihood(i2j);
      double l = l1 * l2;
      total += l;
      for(    j = 0; j < J; j++) table1[j][j2i[j]] += l;
      for(int i = 0; i < I; i++) table2[i][i2j[i]] += l;
      //checkAndDump();
      return;
    }

    for(int i = 0; i <= I; i++) {
      if(i < I && i2j[i] != J) continue; // i already aligned
      j2i[j] = i; if(i < I) i2j[i] = j; // Align
      search(j+1);
      j2i[j] = I; if(i < I) i2j[i] = J; // Undo align
    }
  }

  // sps1: e -> f, sps2: f -> e
  // We work in the context frame of sps1
  public void compute() {
    J = sps1.J;
    I = sps1.I;
    j2i = new int[J]; Arrays.fill(j2i, I);
    i2j = new int[I]; Arrays.fill(i2j, J);
    table1 = new double[J][I+1];
    table2 = new double[I][J+1];
    total = 0;
    search(0);
    NumUtils.scalarMult(table1, 1/total);
    NumUtils.scalarMult(table2, 1/total);
  }

  public ExpAlign getExpAlign1() { return new Model1ExpAlign(table1); }
  public ExpAlign getExpAlign2() { return new Model1ExpAlign(table2); }
}
