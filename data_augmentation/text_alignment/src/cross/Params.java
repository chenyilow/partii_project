package cross;

import fig.basic.*;
import fig.exec.*;
import static fig.basic.LogInfo.*;
import java.io.*;
import java.util.*;

public class Params implements Serializable {
  static final long serialVersionUID = 42;

  public String name;
  public boolean reverse; // Whether the transProbs are reversed
  public StrCondProbTable transProbs = new StrCondProbTable();
  public DistortProbTable distortProbs = new DistortProbTable();

  public Params(String name, boolean reverse) {
    this.name = name;
    this.reverse = reverse;
  }

  public Params(Params params) {
    this.name = params.name;
    this.reverse = params.reverse;
    this.transProbs = (StrCondProbTable)params.transProbs.copy();
    this.distortProbs = params.distortProbs.copy();
  }

  // Called after the M-step
  public void finish() {
    transProbs.normalize();
    distortProbs.normalize();
  }

  public void dump(String path, WordPairStats wpStats, boolean reverse) {
    PrintWriter out = IOUtils.openOutHard(path);
    dump(out, wpStats, reverse);
    out.close();
  }

  public void dump(PrintWriter out, WordPairStats wpStats, boolean reverse) {
    out.println("# Name: " + name);
    out.println("# Reverse: " + reverse);
    out.println("# Translation probabilities");
    transProbs.dump(out, wpStats, reverse);
    out.println("# Distortion probabilities");
    distortProbs.dump(out);
  }

  public void save(String file) {
    IOUtils.writeObjFileHard(file, this);
  }
  public static Params load(String file) {
    return (Params)IOUtils.readObjFileHard(file);
  }

  public Params restrict(List<SentencePair> sentences, boolean reverse) {
    Params subParams = new Params(name, reverse);
    subParams.transProbs = transProbs.restrict(sentences, reverse);
    subParams.distortProbs = distortProbs;
    return subParams;
  }

  public void initZero() {
    transProbs.initZero();
    distortProbs.initZero();
  }
  public void initUniform() {
    transProbs.initUniform();
    distortProbs.initUniform();
  }

  // Measure how much the parameters changed
  public BigStatFig getDiff(Params otherParams) {
    return transProbs.getDiff(otherParams.transProbs);
  }

  public String toString() { return name; }

  @Option(gloss="Input parameters file", required=true)
    public static String inFile;

  public static void main(String[] args) {
    OptionsParser.register("hmm", HMMSentencePairState.class);
    OptionsParser.register("main", Params.class);
    Execution.init(args);

    Params params = load(inFile);

    stdout.println("# Name: " + params.name);
    stdout.println("# Reverse: " + params.reverse);
    stdout.println("# Distortion probabilities");
    params.distortProbs.dump(stdout);
    stdout.println("# Translation probabilities");
    params.transProbs.dump(stdout);

    Execution.finish();
  }
}
