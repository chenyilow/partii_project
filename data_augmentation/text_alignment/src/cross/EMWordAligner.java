package cross;

import java.io.*;
import java.util.*;

import fig.exec.*;
import fig.basic.*;
import static fig.basic.LogInfo.*;
import fig.record.*;

/**
 * Generic template for the EM-based IBM models.
 */
public class EMWordAligner extends IterWordAligner {
  // Alignments always go from English to French: P(F|E)
  SentencePairState.Factory spsFactory; // Used to create objects of a particular model
  final String nullWord = "(NULL)";

  // Options
  @Option(gloss="How to assign null-word probabilities (=1 means 1/n)")
    public static double nullProb = 1;
  @Option(gloss="Use posterior decoding.")
    public static boolean usePosteriorDecoding = true;
  @Option(gloss="Threshold in [0,1] for deciding whether an alignment should exist.")
    public static double posteriorDecodingThreshold = 0.5;
  @Option(gloss="When merging expected sufficient statistics, take into account the NULL (fix).")
    public static boolean mergeConsiderNull = false;
  @Option(gloss="Merge only on last iteration (test)")
    public static boolean mergeOnlyOnLastIteration = false;
  @Option(gloss="Actually compute the exact posterior (q) when doing joint training")
    public static boolean exactJointExp = false;
  @Option(gloss="Don't puke with unknown words")
    public static boolean handleUnknownWords = false;

  @Option(gloss="Use variational inference")
    public static boolean useVariational = false;
  @Option(gloss="Sample the parameters")
    public static boolean useSampling = false;
  @Option(gloss="Concentration per element of the Dirichlet")
    public static double elementConcentration = Double.NaN;
  @Option(gloss="Total concentration of the Dirichlet")
    public static double totalConcentration = Double.NaN;
  @Option(gloss="Total concentration of the Dirichlet")
    public static double distortTotalConcentration = Double.NaN;
  @Option(gloss="Randomness for sampling")
    public static Random random = new Random(1);

  public EMWordAligner(SentencePairState.Factory spsFactory, Evaluator evaluator,
      boolean reverse) {
    this.spsFactory = spsFactory;
    this.evaluator = evaluator;
    this.reverse = reverse;
    this.modelPrefix = !reverse ? "1" : "2";
    this.trainingCache = spsFactory.createCache();
  }

  List<String> getEnWords(SentencePair sp) {
    return !reverse ? sp.getEnglishWords() : sp.getFrenchWords();
  }
  List<String> getFrWords(SentencePair sp) {
    return !reverse ? sp.getFrenchWords() : sp.getEnglishWords();
  }

  public String getName() {
    return spsFactory.getName() + (reverse ? ":reversed" : ":normal");
  }

  /*double logLikelihood(List<SentencePair> sentences) {
    double ll = 0;
    for(SentencePair sp : sentences) {
      ll += spsFactory.create(getEnWords(sp), getFrWords(sp), this).logLikelihood();
      logs("LL " + ll);
    }
    return ll;
  }*/

  void initParams(WordPairStats wpStats) {
    initParams(wpStats, nullWord, true);
  }

  // If nullWord is not null, then consider nullWord in initialization
  /*void initPruneParams(List<SentencePair> sentences) {
    // Intialize parameters (set those we want to consider)
    track("initPruneParams(" + sentences.size() + " sentences): pruning parameters with dice < " + transParamsInitDiceThreshold);
    params.transProbs.clear();
    double initVal = 1.0;
    int numPruned = 0;
    for(SentencePair sp : sentences) {
      for(String fr : getFrWords(sp)) { // Reversed if applicable
        for(String en : getEnWords(sp)) { // Reversed if applicable
          String realEn = reverse ? fr : en;
          String realFr = reverse ? en : fr;
          //dbg("dice(%s, %s) = %f", realEn, realFr, evaluator.wpStats.dice(realEn, realFr));
          if(evaluator.wpStats == null || transParamsInitDiceThreshold < 1e-10 ||
              evaluator.wpStats.dice(realEn, realFr) >= transParamsInitDiceThreshold) {
            params.transProbs.set(en, fr, initVal);
            //dbg("transProbs(%s, %s) = %f\n", en, fr, initVal);
          }
          else
            numPruned++;
        }
        params.transProbs.set(nullWord, fr, initVal);
      }
    }
    params.transProbs.normalize();
    logs("Pruned " + numPruned + " tokens");

    // Init distortion parameters
    logs("Initializing distortion parameters to uniform");
    params.distortProbs.initUniform();

    end_track();
  }*/

  public void train(List<SentencePair> sentences, int numIters) {
    track("EMWordAligner.train(): " + sentences.size() + " sentences");

    initTrain(numIters);
    Record.begin("train");
    while(!trainDone()) {
      track("Iteration " + iter + "/" + numIters);
      Record.begin("iteration", iter);

      initNewParams();
      double logLikelihood = 0;
      for(int t = 0; t < sentences.size(); t++) {
        logs("Sentence " + t + "/" + sentences.size());

        SentencePair sp = sentences.get(t);
        SentencePairState sps = newSentencePairState(sp);

        // E-step
        StopWatch.start("E-step");
        ExpAlign expAlign = sps.computeExpAlign();
        logLikelihood += sps.logLikelihood();
        StopWatch.accumStop("E-step");

        if(Main.rantOutput) expAlign.dump();

        // M-step (partial)
        StopWatch.start("M-step");
        sps.updateNewParams(expAlign);
        StopWatch.accumStop("M-step");

      }
      StopWatch.start("M-step");
      newParams.finish(); // M-step (finish)
      switchToNewParams();
      StopWatch.accumStop("M-step");

      logss("Log-likelihood = " + Fmt.D(logLikelihood));
      if(Main.rantOutput) params.dump(stdout, null, reverse);

      Record.end();
      end_track();
    }
    Record.end();

    end_track();
  }

  public static IntersectedWordAligner newIntersectedWordAligner(WordAligner wa1, WordAligner wa2) {
    IntersectedWordAligner intwa = new IntersectedWordAligner(wa1, wa2);
    if(usePosteriorDecoding)
      intwa.usePosteriorDecoding(posteriorDecodingThreshold);
    return intwa;
  }

  public SentencePairState newSentencePairState(SentencePair sp) {
    return spsFactory.create(getEnWords(sp), getFrWords(sp), this);
  }

  /**
   * Train the two models jointly.
   * Use modified EM algorithm (key step: merging of expectations).
   */
  public static void jointTrain(EMWordAligner wa1, EMWordAligner wa2,
      List<SentencePair> sentences, int numIters, boolean merge) {
    track("jointTrain(): " + sentences.size() + " sentences; merge = " + merge);

    IntersectedWordAligner intwa = newIntersectedWordAligner(wa1, wa2);
    OutputOrderedMap<Integer, String> aerMap
      = new OutputOrderedMap<Integer, String>(Execution.getFile(intwa.modelPrefix+".alignErrorRate"));

    wa1.initTrain(numIters);
    wa2.initTrain(numIters);
    Record.begin("train");
    while(!wa1.trainDone() && !wa2.trainDone()) {
      track("Iteration " + wa1.iter + "/" + numIters);
      Record.begin("iteration", wa1.iter);

      wa1.initNewParams();
      wa2.initNewParams();

      double logLikelihood1 = 0;
      double logLikelihood2 = 0;
      for(int t = 0; t < sentences.size(); t++) {
        logs("Sentence " + t + "/" + sentences.size());

        SentencePair sp = sentences.get(t);
        SentencePairState sps1 = wa1.newSentencePairState(sp);
        SentencePairState sps2 = wa2.newSentencePairState(sp);

        ExpAlign expAlign1, expAlign2;
        if(exactJointExp && merge) {
          ComputeExactExpAlign c = new ComputeExactExpAlign(sps1, sps2);
          c.compute();
          expAlign1 = c.getExpAlign1();
          expAlign2 = c.getExpAlign2();

          // Compute difference in ratio stats
          ExpAlign approxExpAlign1 = sps1.computeExpAlign();
          ExpAlign approxExpAlign2 = sps2.computeExpAlign();
          approxExpAlign1.merge(approxExpAlign1, approxExpAlign2);
          BigStatFig fig = new BigStatFig();
          for(int j = 0; j < expAlign1.J(); j++) {
            for(int i = 0; i < expAlign1.I(); i++) {
              double a1 = approxExpAlign1.get(j, i);
              double b1 =       expAlign1.get(j, i);
              double a2 = approxExpAlign2.get(i, j);
              double b2 =       expAlign2.get(i, j);
              if(NumUtils.isFinite(1/a1)) fig.add(a2/a1);
              if(NumUtils.isFinite(1/b1)) fig.add(a2/b1);
            }
          }
          logs("expAlign ratio: " + fig);
        }
        else {
          // E-step
          StopWatch.start("E-step");
          expAlign1 = sps1.computeExpAlign();
          expAlign2 = sps2.computeExpAlign();
          logLikelihood1 += sps1.logLikelihood();
          logLikelihood2 += sps2.logLikelihood();
          StopWatch.accumStop("E-step");

          //if(merge)
          if(merge && (!mergeOnlyOnLastIteration || wa1.iter == numIters-1))
            expAlign1.merge(expAlign1, expAlign2);
        }

        if(Main.rantOutput) {
          rant("=== expAlign1 ==="); expAlign1.dump();
          rant("=== expAlign2 ==="); expAlign2.dump();
        }

        // M-step (partial)
        StopWatch.start("M-step");
        sps1.updateNewParams(expAlign1);
        sps2.updateNewParams(expAlign2);
        StopWatch.accumStop("M-step");
       
      }

      // M-step (finish)
      StopWatch.start("M-step");
      wa1.newParams.finish();
      wa2.newParams.finish();
      wa1.switchToNewParams();
      wa2.switchToNewParams();
      StopWatch.accumStop("M-step");

      logss("Log-likelihood 1 = " + Fmt.D(logLikelihood1));
      logss("Log-likelihood 2 = " + Fmt.D(logLikelihood2));

      // Evaluate joint model
      double aer = wa1.evaluator.test(intwa, false, false).aer;
      logss("AER 1+2 = " + Fmt.D(aer));
      aerMap.put(wa1.iter-1, Fmt.D(wa1.aer) + " " + Fmt.D(wa1.aer) + " " + Fmt.D(aer));
      Execution.putOutput("AER", Fmt.D(aer));
      Record.add("aer1", wa1.aer);
      Record.add("aer2", wa2.aer);
      Record.add("aer", aer);

      Record.end();
      end_track();
    }
    Record.end();

    end_track();
  }

  public Alignment alignSentencePair(SentencePair sp) {
    SentencePairState sps = newSentencePairState(sp);
    if(usePosteriorDecoding)
      return Alignment.thresholdPosteriors(sps.getPosteriors(reverse), posteriorDecodingThreshold);
    else
      return sps.getViterbi(reverse);
  }
}
