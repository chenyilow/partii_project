package cross;

import java.util.*;

import fig.exec.*;
import fig.basic.*;
import static fig.basic.LogInfo.*;
import fig.record.*;

/**
 * WordAligner with parameters which are trained via some number of iterations.
 */
public abstract class IterWordAligner extends WordAligner {
  /*@Option(gloss="Prune translation parameters if below this threshold")
    public static double transParamsThreshold = 0;
  @Option(gloss="Prune translation parameters if dice below this threshold")
    public static double transParamsInitDiceThreshold = 0;
  @Option(gloss="Keep this many translation parameters per word")
    public static int numTransParamsPerWord = -1;*/

  // Used to evaluate performance during training time
  Evaluator evaluator;
  boolean reverse; // Whether to reverse roles of French and English
  TrainingCache trainingCache;

  // Record performance over iterations
  OutputOrderedMap<String, String> aerMap, changeMap;

  // Parameters
  Params params, newParams;
  int iter, numIters;
  double aer;

  public abstract String getName();

  protected void initNewParams() {
    if(newParams == null) newParams = new Params(params);
    newParams.initZero();
  }

  /**
   * Write the model parameters to disk.
   * Output files: <prefix>.bin, <prefix>.txt
   */
  void saveParams() {
    String file = Execution.getFile(modelPrefix + ".params");
    if(file == null) return;
    track("saveParams(" + file + ")");
    track("Text"); {
      params.dump(file+".txt", evaluator.wpStats, reverse);
      params.restrict(evaluator.testSentencePairs, reverse).dump(
        file+"-test.txt", evaluator.wpStats, reverse);
    } end_track();
    track("Binary"); {
      params.save(file+".bin");
      params.restrict(evaluator.testSentencePairs, reverse).save(file+"-test.bin");
    } end_track();
    end_track();
  }
  void loadParams(String dir) {
    if(dir == null || dir.equals("")) return;
    String file = dir + "/" + modelPrefix + ".params.bin";
    track("loadParams(" + file + ")");
    params = Params.load(file);
    params.transProbs.lock();
    logs("Loaded " + params);
    Execution.linkFileToExec(file, modelPrefix + "-loaded.params.bin");
    end_track();
  }

  // Keep only the top translation parameters (u, v).
  // Right now, only use the top with respect to u.
  /*void pruneNumParams() {
    if(numTransParamsPerWord == -1) return;
    track("pruneNumParams(" + numTransParamsPerWord + ")");
    //params.transProbs.truncateParamsByNumber(numTransParamsPerWord);
    end_track();
  }*/

  abstract void initParams(WordPairStats wpStats);

  protected void initParams(WordPairStats wpStats, String nullWord, boolean initUniform) {
    params = new Params(getName(), reverse);

    params.transProbs = (StrCondProbTable)wpStats.allocateForSentencePairs(new StrCondProbTable(), reverse);
    if(nullWord != null)
      params.transProbs.put(nullWord, wpStats.allocateForSentencePairs(new StringDoubleMap(), !reverse));
    params.transProbs.switchToSortedList();

    if(initUniform)
      params.initUniform();
    else
      params.initZero();
    params.transProbs.lock();
    trainingCache.clear();
  }

  void initTrain(int numIters) {
    params.name = getName();

    aerMap    = new OutputOrderedMap<String, String>(Execution.getFile(modelPrefix+".alignErrorRate"));
    changeMap = new OutputOrderedMap<String, String>(Execution.getFile(modelPrefix+".changeInParams"));
    Execution.putOutput("Iterations", "0");
    this.numIters = numIters;
    this.iter = 1;
  }

  boolean trainDone() {
    return iter > numIters;
  }

  void switchToNewParams() {
    // Change in parameters
    StatFig changeFig = params.getDiff(newParams);
    logss("Change in parameters: " + changeFig);
    changeMap.put(""+iter, ""+changeFig);
    Execution.putOutput("Change", changeFig.toString());
    //Record.add("changeInParams", changeFig.mean());

    // Switch the two
    Params tmpParams = params;
    params = newParams;
    newParams = tmpParams;
    trainingCache.clear();
    //pruneParams();
    //pruneNumParams();

    // Alignment error rate
    aer = evaluator.test(this, false, false).aer;
    logss("AER = %f", aer);
    aerMap.put(""+iter, ""+aer);
    Execution.putOutput("AER", Fmt.D(aer));
    //Record.add("aer", aer);

    // Misc. input/output
    Execution.putOutput("Iterations", ""+iter);
    if(Execution.getBooleanInput("save")) saveParams();
    if(Execution.getBooleanInput("eval")) evaluator.test(this, true, false);
    if(Execution.getBooleanInput("kill")) kill();
    printProgStatus();

    iter++;
  }

  public void kill() {
    iter = numIters;
    Execution.setExecStatus("killed", true);
  }
}
