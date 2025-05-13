package cross;

import java.io.*;
import java.util.*;

import fig.exec.*;
import fig.basic.*;
import fig.record.*;
import static fig.basic.LogInfo.*;

public class Evaluator {
  public Evaluator(List<SentencePair> testSentencePairs, Map<Integer, Alignment> referenceAlignments, WordPairStats wpStats) {
    this.testSentencePairs = testSentencePairs;
    this.referenceAlignments = referenceAlignments;
    this.wpStats = wpStats;
  }

  public Performance test(WordAligner wordAligner, boolean output, boolean evalPRTradeoff) {
    track("Evaluator.test(" + wordAligner.getName() + ")");

    // Main computation: align sentences!
    Map<Integer, Alignment> proposedAlignments =
      wordAligner.alignSentencePairs(testSentencePairs);

    // Do precision/recall tradeoff: only meaningful if we were using posterior decoding
    double bestThreshold = -1;
    double bestAER = 2;
    if(evalPRTradeoff) {
      Record.begin("PRTradeoff");
      Record.setStruct("threshold", "precision", "recall", "aer");
      track("Eval precision/recall tradeoff");
      // Get an entire curve
      OutputOrderedMap<Double, String> postMap = new OutputOrderedMap<Double, String>(Execution.getFile(wordAligner.modelPrefix+".PRTradeoff"));
      int numIntervals = 100;
      for(int i = 0; i < numIntervals; i++) {
        double threshold = 1.0*i/numIntervals;
        Performance perf = eval(testSentencePairs, referenceAlignments,
            Alignment.thresholdAlignmentsByStrength(proposedAlignments, threshold));
        postMap.put(threshold, perf.simpleString());
        logs("Threshold = %f; AER = %f", threshold, perf.aer);
        Record.add(""+threshold, perf.precision, perf.recall, perf.aer);
        if(perf.aer < bestAER) {
          bestAER = perf.aer;
          bestThreshold = threshold;
        }
      }
      logss("Best threshold = %f, AER = %f", bestThreshold, bestAER);
      Record.end();
      end_track();
    }

    // Output alignments
    track("Output alignments");
    String file = Execution.getFile(wordAligner.modelPrefix);
    if(output && file != null) {
      AlignmentsInfo ainfo = 
        new AlignmentsInfo(wordAligner.getName(), testSentencePairs, referenceAlignments, proposedAlignments);
      ainfo.writeBinary(file + ".alignOutput.bin");
      if(!Main.condenseAlignOutput)
        ainfo.writeText(file + ".alignOutput.txt", wpStats);

      // Evaluate the best possible threshold
      if(EMWordAligner.usePosteriorDecoding) {
        double saveThreshold = EMWordAligner.posteriorDecodingThreshold;
        EMWordAligner.posteriorDecodingThreshold = bestThreshold;
        proposedAlignments = wordAligner.alignSentencePairs(testSentencePairs);
        EMWordAligner.posteriorDecodingThreshold = saveThreshold;
        ainfo = 
          new AlignmentsInfo(wordAligner.getName(), testSentencePairs, referenceAlignments, proposedAlignments);
        ainfo.writeBinary(file + ".alignOutput.best.bin");
        if(!Main.condenseAlignOutput)
          ainfo.writeText(file + ".alignOutput.best.txt", wpStats);
      }
    }
    end_track();

    // Evaluate performance
    Performance perf = eval(testSentencePairs, referenceAlignments, proposedAlignments);
    perf.bestThreshold = bestThreshold; // Not very clean place to put this
    perf.bestAER = bestAER;
    perf.dump();

    end_track();
    return perf;
  }

  // Evaluate the proposed alignments against the reference alignments.
  public static Performance eval(List<SentencePair> testSentencePairs, Map<Integer, Alignment> referenceAlignments,
      Map<Integer, Alignment> proposedAlignments) {
    Performance perf = new Performance();

    //int idx = 0;
    for(SentencePair sentencePair : testSentencePairs) {
      //logs("Sentence %d/%d", idx++, testSentencePairs.size());

      int I = sentencePair.I();
      int J = sentencePair.J();

      Alignment proposedAlignment = proposedAlignments.get(sentencePair.getSentenceID());
      Alignment referenceAlignment = referenceAlignments.get(sentencePair.getSentenceID());

      // Silently ignore alignments that aren't there
      if(proposedAlignments == null || referenceAlignment == null) continue;

      boolean[] hit1 = new boolean[I];
      boolean[] hit2 = new boolean[J];

      for (int j = 0; j < J; j++) {
        for (int i = 0; i < I; i++) {
          boolean proposed = proposedAlignment.containsSureAlignment(i, j);
          boolean sure = referenceAlignment.containsSureAlignment(i, j);
          boolean possible = referenceAlignment.containsPossibleAlignment(i, j);
          double strength = proposedAlignment.getStrength(i, j);

          perf.addPoint(proposed, sure, possible, strength);
          if(proposed) hit1[i] = hit2[j] = true;
        }
      }

      for(int i = 0; i < I; i++) if(!hit1[i]) perf.numNull1++;
      for(int j = 0; j < J; j++) if(!hit2[j]) perf.numNull2++;
    }

    perf.computeFromCounts();
    return perf;
  }

  List<SentencePair> testSentencePairs;
  Map<Integer, Alignment> referenceAlignments; // Gold alignments
  WordPairStats wpStats;
}
