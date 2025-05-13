package cross;

import java.io.*;
import java.util.*;

/**
 * A holder for a pair of sentences, each a list of strings.  Sentences in
 * the test sets have integer IDs, as well, which are used to retreive the
 * gold standard alignments for those sentences.
 */
public class SentencePair implements Serializable {
  static final long serialVersionUID = 42;

  int sentenceID;
  String sourceFile;
  List<String> englishWords;
  List<String> frenchWords;

  public SentencePair reverse() {
    return new SentencePair(sentenceID, sourceFile, frenchWords, englishWords);
  }

  public SentencePair(int sentenceID, String sourceFile, List<String> englishWords, List<String> frenchWords) {
    this.sentenceID = sentenceID;
    this.sourceFile = sourceFile;
    this.englishWords = englishWords;
    this.frenchWords = frenchWords;
  }

  public int getSentenceID() { return sentenceID; }
  public String getSourceFile() { return sourceFile; }
  public List<String> getEnglishWords() { return englishWords; }
  public List<String> getFrenchWords() { return frenchWords; }
  public int I() { return englishWords.size(); }
  public int J() { return frenchWords.size(); }
  public String en(int i) { return englishWords.get(i); }
  public String fr(int j) { return frenchWords.get(j); }

  public String toString() {
    StringBuilder sb = new StringBuilder();
    for (int englishPosition = 0; englishPosition < englishWords.size(); englishPosition++) {
      String englishWord = englishWords.get(englishPosition);
      sb.append(englishPosition);
      sb.append(":");
      sb.append(englishWord);
      sb.append(" ");
    }
    sb.append("\n");
    for (int frenchPosition = 0; frenchPosition < frenchWords.size(); frenchPosition++) {
      String frenchWord = frenchWords.get(frenchPosition);
      sb.append(frenchPosition);
      sb.append(":");
      sb.append(frenchWord);
      sb.append(" ");
    }
    sb.append("\n");
    return sb.toString();
  }

  // Return the set of words used in these sentences.
  public static Set<String> getWordSet(List<SentencePair> sentencePairs, boolean isFrench) {
    Set<String> set = new HashSet<String>();
    for(SentencePair sp : sentencePairs) {
      List<String> words = isFrench ? sp.getFrenchWords() : sp.getEnglishWords();
      for(String w : words) set.add(w);
    }
    return set;
  }

  public SentencePair chop(int i1, int i2, int j1, int j2) {
    return new SentencePair(sentenceID, sourceFile,
        englishWords.subList(i1, i2), frenchWords.subList(j1, j2));
  }
}
