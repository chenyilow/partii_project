package cross;

import java.io.*;
import java.util.*;
import fig.basic.*;

/**
 * Alignments serve two purposes, both to indicate your system's guessed
 * alignment, and to hold the gold standard alignments.  Alignments map index
 * pairs to one of three values, unaligned, possibly aligned, and surely
 * aligned.  Your alignment guesses should only contain sure and unaligned
 * pairs, but the gold alignments contain possible pairs as well.
 *
 * To build an alignemnt, start with an empty one and use
 * addAlignment(i,j,true).  To display one, use the render method.
 */
public class Alignment implements Serializable {
  static final long serialVersionUID = 42;

  Set<Pair<Integer, Integer>> sureAlignments;
  Set<Pair<Integer, Integer>> possibleAlignments;
  Map<Pair<Integer, Integer>, Double> strengths; // Strength of alignments

  public Alignment() {
    sureAlignments = new HashSet<Pair<Integer, Integer>>();
    possibleAlignments = new HashSet<Pair<Integer, Integer>>();
    strengths = new HashMap<Pair<Integer, Integer>, Double>();
  }

  public void condense() { strengths.clear(); }

  // Return list of indices aligned to the ith English word
  private List<Integer> getEnglishAlignments(int i, int J) {
    List<Integer> js = new ArrayList<Integer>();
    for(int j = 0; j < J; j++)
      if(sureAlignments.contains(new Pair<Integer, Integer>(i, j)))
        js.add(j);
    return js;
  }
  private List<Integer> getNullEnglishAlignments(int J) {
    List<Integer> js = new ArrayList<Integer>();
    boolean[] hit = new boolean[J];
    for(Pair<Integer, Integer> p : sureAlignments)
      hit[p.getSecond()] = true;
    for(int j = 0; j < J; j++)
      if(!hit[j]) js.add(j);
    return js;
  }

  private List<Integer> addOne(List<Integer> list) {
    List<Integer> newList = new ArrayList<Integer>();
    for(int x : list) newList.add(x+1);
    return newList;
  }

  public void writeGIZA(PrintWriter out, int idx, SentencePair sp) {
    out.printf("# sentence pair (%d) source length %d target length %d alignment score : 0\n",
        idx, sp.I(), sp.J());
    out.println(StrUtils.join(sp.getFrenchWords()));
    out.printf("NULL ({ %s })", StrUtils.join(addOne(getNullEnglishAlignments(sp.J()))));
    for(int i = 0; i < sp.I(); i++) {
      out.printf(" %s ({ %s })", sp.en(i), StrUtils.join(addOne(getEnglishAlignments(i, sp.J()))));
    }
    out.println("");
  }

  public boolean containsAlignment(int englishPosition, int frenchPosition) {
    Pair<Integer, Integer> p = new Pair<Integer, Integer>(englishPosition, frenchPosition);
    return sureAlignments.contains(p) || possibleAlignments.contains(p);
  }

  public boolean containsSureAlignment(int englishPosition, int frenchPosition) {
    return sureAlignments.contains(new Pair<Integer, Integer>(englishPosition, frenchPosition));
  }

  public boolean containsPossibleAlignment(int englishPosition, int frenchPosition) {
    return possibleAlignments.contains(new Pair<Integer, Integer>(englishPosition, frenchPosition));
  }

  public void addAlignment(int englishPosition, int frenchPosition, boolean sure) {
    Pair<Integer, Integer> alignment = new Pair<Integer, Integer>(englishPosition, frenchPosition);
    if (sure)
      sureAlignments.add(alignment);
    possibleAlignments.add(alignment);
  }

  public void setStrength(int i, int j, double strength) {
    strengths.put(new Pair<Integer, Integer>(i, j), strength);
  }
  public double getStrength(int i, int j) {
    return MapUtils.get(strengths, new Pair<Integer, Integer>(i, j), 0.0);
  }

  public double[][] getPosteriors(int I, int J) {
    double[][] posteriors = new double[J][I];
    for(int j = 0; j < J; j++)
      for(int i = 0; i < I; i++)
        posteriors[j][i] = getStrength(i, j);
    return posteriors;
  }

  // Keep alignments which are above the threshold.
  public static Alignment thresholdPosteriors(double[][] posteriors, double threshold) {
    Alignment alignment = new Alignment();
    int J = posteriors.length;
    int I = posteriors[0].length;
    for(int j = 0; j < J; j++) {
      for(int i = 0; i < I; i++) {
        alignment.setStrength(i, j, posteriors[j][i]);
        if(posteriors[j][i] >= threshold) {
          alignment.addAlignment(i, j, true);
          //rant("strength(%d, %d) := %f", i, j, posteriors[j][i]);
        }
      }
    }
    return alignment;
  }

  // Create a new alignment based on thresholding the strengths of the provided alignment.
  public static Alignment thresholdAlignmentByStrength(Alignment alignment, double threshold) {
    Alignment newAlignment = new Alignment();
    for(Pair<Integer, Integer> ij : alignment.strengths.keySet()) {
      int i = ij.getFirst(), j = ij.getSecond();
      double strength = alignment.strengths.get(ij);
      newAlignment.setStrength(i, j, strength);
      if(strength >= threshold)
        newAlignment.addAlignment(i, j, true);
    }
    return newAlignment;
  }

  // Create new alignments for a whole set of alignments.
  public static Map<Integer, Alignment> thresholdAlignmentsByStrength(Map<Integer, Alignment> alignments, double threshold) {
    Map<Integer, Alignment> newAlignments = new HashMap<Integer, Alignment>();
    for(int sid : alignments.keySet()) {
      Alignment alignment = alignments.get(sid);
      alignment = alignment.thresholdAlignmentByStrength(alignment, threshold);
      newAlignments.put(sid, alignment);
    }
    return newAlignments;
  }

  public Alignment intersect(Alignment a) {
    Alignment ia = new Alignment();
    for(Pair<Integer, Integer> p : sureAlignments)
      if(a.sureAlignments.contains(p))
        ia.sureAlignments.add(p);
    for(Pair<Integer, Integer> p : possibleAlignments)
      if(a.possibleAlignments.contains(p))
        ia.possibleAlignments.add(p);
    return ia;
  }
  public Alignment subtract(Alignment a) {
    Alignment ia = new Alignment();
    for(Pair<Integer, Integer> p : sureAlignments)
      if(!a.sureAlignments.contains(p))
        ia.sureAlignments.add(p);
    for(Pair<Integer, Integer> p : possibleAlignments)
      if(!a.possibleAlignments.contains(p))
        ia.possibleAlignments.add(p);
    return ia;
  }
  public Alignment union(Alignment a) {
    Alignment ua = new Alignment();
    for(Pair<Integer, Integer> p : sureAlignments)   ua.sureAlignments.add(p);
    for(Pair<Integer, Integer> p : a.sureAlignments) ua.sureAlignments.add(p);
    for(Pair<Integer, Integer> p : possibleAlignments)   ua.possibleAlignments.add(p);
    for(Pair<Integer, Integer> p : a.possibleAlignments) ua.possibleAlignments.add(p);
    return ua;
  }

  public Alignment reverse() {
    Alignment a2 = new Alignment();
    for(Pair<Integer, Integer> p : sureAlignments)
      a2.sureAlignments.add(p.reverse());
    for(Pair<Integer, Integer> p : possibleAlignments)
      a2.possibleAlignments.add(p.reverse());
    return a2;
  }

  public static String render(Alignment alignment, SentencePair sentencePair) {
    return render(alignment, alignment, sentencePair, null);
  }

  static String singleCharCount(double x) {
    int n = (int)(x+0.5);
    if(n < 10) return "" + n;
    if(n < 20) return "@";
    if(n < 50) return "%";
    if(n < 100) return "*";
    return "+";
  }

  public static String render(Alignment reference, Alignment proposed,
      SentencePair sentencePair, WordPairStats wpStats) {

    StringBuilder sb = new StringBuilder();
    for (int frenchPosition = 0; frenchPosition < sentencePair.getFrenchWords().size(); frenchPosition++) {
      for (int englishPosition = 0; englishPosition < sentencePair.getEnglishWords().size(); englishPosition++) {
        boolean sure = reference != null && reference.containsSureAlignment(englishPosition, frenchPosition);
        boolean possible = reference != null && reference.containsPossibleAlignment(englishPosition, frenchPosition);
        char proposedChar = ' ';
        if (proposed.containsSureAlignment(englishPosition, frenchPosition))
          proposedChar = '#';
        if (sure) {
          sb.append('[');
          sb.append(proposedChar);
          sb.append(']');
        } else {
          if (possible) {
            sb.append('(');
            sb.append(proposedChar);
            sb.append(')');
          } else {
            sb.append(' ');
            sb.append(proposedChar);
            sb.append(' ');
          }
        }
      }
      sb.append("| ");
      String fr = sentencePair.getFrenchWords().get(frenchPosition); 
      if(wpStats != null)
        sb.append(String.format("%-15s\t%s", fr, singleCharCount(wpStats.frCount(fr))));
      else
        sb.append(fr);
      sb.append('\n');
    }
    for (int englishPosition = 0; englishPosition < sentencePair.getEnglishWords().size(); englishPosition++) {
      sb.append("---");
    }
    sb.append("'\n");
    boolean printed = true;
    int index = 0;
    while (printed) {
      printed = false;
      StringBuilder lineSB = new StringBuilder();
      for (int englishPosition = 0; englishPosition < sentencePair.getEnglishWords().size(); englishPosition++) {
        String englishWord = sentencePair.getEnglishWords().get(englishPosition);
        if (englishWord.length() > index) {
          printed = true;
          lineSB.append(' ');
          lineSB.append(englishWord.charAt(index));
          lineSB.append(' ');
        } else {
          lineSB.append("   ");
        }
      }
      index += 1;
      if (printed) {
        sb.append(lineSB);
        sb.append('\n');
      }
    }

    // Print counts of English words
    if(wpStats != null) {
      StringBuilder lineSB = new StringBuilder();
      for(int i = 0; i < sentencePair.getEnglishWords().size(); i++) {
        String en = sentencePair.getEnglishWords().get(i);
        lineSB.append(' ');
        lineSB.append(singleCharCount(wpStats.enCount(en)));
        lineSB.append(' ');
      }
      sb.append(lineSB);
      sb.append('\n');
    }

    return sb.toString();
  }

  public Alignment chop(int i1, int i2, int j1, int j2) {
    Alignment choppedAlignment = new Alignment();
    for (int i = i1; i < i2; i++) {
      for (int j = j1; j < j2; j++) {
        boolean isPossible = containsPossibleAlignment(i, j);
        boolean isSure = containsSureAlignment(i, j);
        if (isPossible) {
          choppedAlignment.addAlignment(i-i1, j-j1, isSure);
        }
        choppedAlignment.setStrength(i-i1, j-j1, getStrength(i, j));
      }
    }
    return choppedAlignment;
  }

  public Set<Pair<Integer, Integer>> getSureAlignments() { return sureAlignments; }
  public Set<Pair<Integer, Integer>> getPossibleAlignments() { return possibleAlignments; }
  public Map<Pair<Integer, Integer>, Double> getStrengths() { return strengths; }
}
