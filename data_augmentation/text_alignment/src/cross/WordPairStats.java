package cross;

import fig.basic.*;
import static fig.basic.LogInfo.*;
import java.io.*;
import java.util.*;

/** Stores statistics about English and French words.
 * Importantly, it knows about all pairs of English/French words
 * that appear in the same sentence (thus, worth knowing about).
 */
public class WordPairStats implements Serializable {
  static final long serialVersionUID = 42;

  // Keeps track of and provides access to the following
  //   - Set of English words
  //   - Set of French words
  //   - Counts of English words
  //   - Counts of French words
  //   - Co-occurrence counts of English/French word pair
  //   - Dice score of English/French word pair
  // TODO: make these doubles ints.
  // Can be used to prune.
  StringDoubleMap enCounts = new StringDoubleMap();
  StringDoubleMap frCounts = new StringDoubleMap();
  String2DoubleMap counts = new String2DoubleMap();
  List<SentencePair> sentencePairs;

  public int numEn() { return enCounts.size(); }
  public int numFr() { return frCounts.size(); }
  public Set<String> enWords() { return enCounts.keySet(); }
  public Set<String> frWords() { return frCounts.keySet(); }
  public StringDoubleMap getEnCounts(boolean reverse) { return !reverse ? enCounts : frCounts; }
  public StringDoubleMap getFrCounts(boolean reverse) { return !reverse ? frCounts : enCounts; }
  public boolean hasStats() { return counts.size() > 0; }

  public WordPairStats() { }
  public WordPairStats(List<SentencePair> sentencePairs) {
    this.sentencePairs = sentencePairs;
  }

  public StringDoubleMap allocateForSentencePairs(StringDoubleMap map, boolean isFrench) {
    for(SentencePair sp : sentencePairs) {
      List<String> words = !isFrench ? sp.getEnglishWords() : sp.getFrenchWords();
      for(String w : words) map.put(w, 0);
    }
    return map;
  }
  public String2DoubleMap allocateForSentencePairs(String2DoubleMap map, boolean reverse) {
    // Allocate the memory for the sentences in map.
    for(SentencePair sp : sentencePairs) {
      List<String> enWords = sp.getEnglishWords();
      List<String> frWords = sp.getFrenchWords();
      if(!reverse) {
        for(String en : enWords)
          for(String fr : frWords)
            map.put(en, fr, 0);
      }
      else {
        for(String fr : frWords)
          for(String en : enWords)
            map.put(fr, en, 0);
      }
    }
    return map;
  }

  public void lock() {
    counts.switchToSortedList(); // To save space
    counts.lock();
    enCounts.lock();
    frCounts.lock();
  }

  public double enCount(String en) { return enCounts.get(en, 0.0); }
  public double frCount(String fr) { return frCounts.get(fr, 0.0); }
  public double count(String en, String fr) { return counts.get(en, fr, 0.0); }
  public double dice(String en, String fr) {
    double n = count(en, fr);
    if(n < 1e-10) return 0;
    return n / (enCount(en) + frCount(fr));
  }
  public double enCount(boolean reverse, String w) { return !reverse ? enCount(w) : frCount(w); }
  public double frCount(boolean reverse, String w) { return !reverse ? frCount(w) : enCount(w); }
  public double dice(boolean reverse, String w1, String w2) {
    return !reverse ? dice(w1, w2) : dice(w2, w1);
  }

  public void computeStats(List<SentencePair> sentencePairs) {
    track("WordPairStats.computeStats(): " + sentencePairs.size() + " sentences");

    allocateForSentencePairs(enCounts, false);
    allocateForSentencePairs(frCounts, true);
    allocateForSentencePairs(counts, false);

    for(SentencePair sp : sentencePairs) {
      List<String> enWords = sp.getEnglishWords();
      List<String> frWords = sp.getFrenchWords();

      for(String en : enWords) enCounts.incr(en, 1.0);
      for(String fr : frWords) frCounts.incr(fr, 1.0);

      // If English word u occurs n_u times and French word v occurs n_v times
      // in this sentence, then we increment (u, v) by min(n_u, n_v)
      Map<String, Integer> enHist = ListUtils.buildHistogram(enWords);
      Map<String, Integer> frHist = ListUtils.buildHistogram(frWords);
      for(String en : enHist.keySet())
        for(String fr : frHist.keySet())
          counts.incr(en, fr, Math.min(enHist.get(en), frHist.get(fr)));
    }
    end_track();
  }

  // Return a word pair stats that contains all the information in this instance,
  // but restricted to the words that only appear in the given sentences.
  public WordPairStats restrict(List<SentencePair> sentencePairs) {
    WordPairStats stats = new WordPairStats();

    // Figure out which words we need to save
    Set<String> enSet = SentencePair.getWordSet(sentencePairs, false);
    Set<String> frSet = SentencePair.getWordSet(sentencePairs, true);

    // Go through our counts and save the specified words
    stats.enCounts = enCounts.restrict(enSet);
    stats.frCounts = frCounts.restrict(frSet);
    stats.counts = counts.restrict(enSet, frSet);

    return stats;
  }

  public void dump(String path) throws IOException { dump(new File(path)); }
  public void dump(File path) throws IOException {
    PrintWriter out = IOUtils.openOut(path);
    dump(out);
    out.close();
  }
  public void dump(PrintWriter out) {
    out.println("# English " + enCounts.size());
    for(StringDoubleMap.Entry e : enCounts)
      out.println(e.getKey() + "\t" + e.getValue());
    out.println("# French " + frCounts.size());
    for(StringDoubleMap.Entry e : frCounts)
      out.println(e.getKey() + "\t" + e.getValue());
    out.println("# English, French");
    for(Map.Entry<String, StringDoubleMap> e1 : counts) {
      String en = e1.getKey();
      StringDoubleMap m = e1.getValue();
      out.println(en);
      for(StringDoubleMap.Entry e2 : m)
        out.println("  " + e2.getKey() + "\t" + e2.getValue());
    }
  }

  public static WordPairStats load(String file) {
    logs("WordPairStats.load(" + file + ")");
    return (WordPairStats)IOUtils.readObjFileEasy(file);
  }
  public void save(String file) {
    if(file == null) return;
    logs("WordPairStats.save(" + file + ")");
    IOUtils.writeObjFileEasy(file, this);
  }

  public static void main(String[] args) {
    LogInfo.init();
    load(args[0]).dump(stdout);
  }
}
