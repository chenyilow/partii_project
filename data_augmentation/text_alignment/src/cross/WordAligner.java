package cross;

import fig.basic.*;
import static fig.basic.LogInfo.*;

import java.util.*;

/**
 * WordAligners have one method: alignSentencePair, which takes a sentence
 * pair and produces an alignment which specifies an english source for each
 * french word which is not aligned to "null".  Explicit alignment to
 * position -1 is equivalent to alignment to "null".
 */
public abstract class WordAligner {
  String modelPrefix; // Identifies the model (for writing any files)

  public abstract String getName();

  // IMPORTANT: At least one of the following methods should be overridden.
  public Alignment alignSentencePair(SentencePair sp) {
    ArrayList<SentencePair> spList = new ArrayList<SentencePair>();
    spList.add(sp);
    Map<Integer, Alignment> alignments = alignSentencePairs(spList);
    return alignments.get(sp.sentenceID);
  }
  public Map<Integer, Alignment> alignSentencePairs(List<SentencePair> sentencePairs) {
    track("alignSentencePairs(%d sentences)", sentencePairs.size());
    Map<Integer, Alignment> alignments = new HashMap<Integer, Alignment>();
    int idx = 0;
    boolean condense = Main.condenseAlignOutput;

    for(SentencePair sp : sentencePairs) {
      logs("Sentence %d/%d", idx++, sentencePairs.size());
      Alignment alignment = alignSentencePair(sp);
      if(condense) alignment.condense();
      alignments.put(sp.sentenceID, alignment);
    }
    end_track();
    return alignments;
  }
}
