package cross;

import java.io.*;
import java.util.*;

import fig.exec.*;
import fig.basic.*;
import static fig.basic.LogInfo.*;

/**
 * Entry point for word alignment.
 */
public class Main {
  public enum ModelType { BASELINE, OUTPUT_WPSTATS, MODEL1, MODEL2, HMM };
  public enum TrainingMode { NORMAL, REVERSE, BOTH_INDEP, BOTH_JOINT };

  // Options {
  @Option(gloss="File extension for English files")
    public static String enExt = "e";
  @Option(gloss="File extension for French files")
    public static String frExt = "f";
  @Option(gloss="Directory or file containing all the training files (used to compute statistics).")
    public static String allTrainSource;
  @Option(gloss="Directory or file containing training files.")
    public static ArrayList<String> trainSources = new ArrayList<String>();
  @Option(gloss="Directory or file containing testing files.")
    public static ArrayList<String> testSources = new ArrayList<String>();
  @Option(gloss="Directory or file containing all the testing files (used to output a large thing).")
    public static ArrayList<String> allTestSources = null;

  @Option(name="sentences", gloss="Maximum number of the training sentences to use", required=true)
    public static int maxTrainSentences = Integer.MAX_VALUE;
  @Option(gloss="Maximum number of the test sentences to use")
    public static int maxTestSentences = Integer.MAX_VALUE;
  @Option(gloss="Skip the first few sentences")
    public static int offsetTestSentences = 0;
  @Option(name="iters", gloss="Number of iterations to run the model.")
    public static int numIters = 5;
  @Option(gloss="Which word alignment model to use.", required=true)
    public static ModelType model = ModelType.BASELINE;
  @Option(name="mode", gloss="Whether to train the two models jointly or independently.", required=true)
    public static TrainingMode trainingMode = TrainingMode.NORMAL;
  @Option(gloss="Whether to normalize the expected counts by I*J, number of possible alignments")
    public static boolean useNormedObjective = false;
  @Option(gloss="Whether to convert all words to lowercase")
    public static boolean lowercaseWords = false;
  @Option(gloss="Whether to append _<i> to the ith occurence of a word in a sentence")
    public static boolean appendOccurToWord = false;

  @Option(gloss="Directory to load parameters from.")
    public static String loadParamsDir = "";
  @Option(gloss="File to write word pair statistics (for reading and writing).")
    public static String wpStatsFile = "";
  @Option(gloss="Whether to save parameters.")
    public static boolean saveParams = true;
  @Option(gloss="Whether to save alignments produced by the system.")
    public static boolean saveAlignOutput = true;
  @Option(gloss="Produce two GIZA-like files (for phrase translation)")
    public static boolean alignIntUnion = false;
  @Option(gloss="Print out combined posteriors at test time")
    public static boolean outputTestPosteriors = false;
  @Option(gloss="At test time, the intersected word aligner uses this to combine posteriors")
    public static IntersectedWordAligner.CombineMethod combineMethod = IntersectedWordAligner.CombineMethod.multiply;

  @Option(gloss="Save memory: compute statistics (if not reading in from a file)")
    public static boolean computeWpStats = false;
  @Option(gloss="Save memory: just store proposed alignments, not strengths, etc.")
    public static boolean condenseAlignOutput = false;
  @Option(gloss="Output a lot of junk")
    public static boolean rantOutput = false;
  // }

  public static void main(String[] args) {
    OptionsParser.register("main", Main.class);
    OptionsParser.register("em", EMWordAligner.class);
    OptionsParser.register("hmm", HMMSentencePairState.class);
    OptionsParser.register("iter", IterWordAligner.class);

    Execution.init(args);
    try {
      doMain();
    } catch(Throwable t) {
      Execution.raiseException(t);
    }
    Execution.finish();
  }

  private static void doMain() {
    // Read training and testing sentence pairs and alignments.
    track("Reading files");
    Interner<String> strdb = new Interner<String>();
    List<SentencePair> trainSentencePairs = new ArrayList<SentencePair>();
    List<SentencePair> testSentencePairs = new ArrayList<SentencePair>();
    for(String trainSource : trainSources)
      readSentencePairsFromSource(trainSource, 0, maxTrainSentences, strdb, trainSentencePairs);
    for(String testSource : testSources)
      readSentencePairsFromSource(testSource, offsetTestSentences, maxTestSentences, strdb, testSentencePairs);

    maxTrainSentences = Math.min(maxTrainSentences, trainSentencePairs.size());
    maxTestSentences = Math.min(maxTestSentences, testSentencePairs.size());

    Map<Integer, Alignment> testAlignments = new HashMap<Integer, Alignment>();
    for(String testSource : testSources)
      readAlignmentsFromSource(testSource, testAlignments);
    logs("%d training sentences, %d test sentences, %d alignments",
        trainSentencePairs.size(), testSentencePairs.size(), testAlignments.size());
    assertSentenceIDsAreUnique(trainSentencePairs);
    assertSentenceIDsAreUnique(testSentencePairs);
    end_track();

    // Load word pair stats
    WordPairStats wpStats = null;
    if(model != ModelType.OUTPUT_WPSTATS) {
      if(!wpStatsFile.equals("")) { // Read from file
        wpStats = WordPairStats.load(wpStatsFile);
        Execution.linkFileToExec(wpStatsFile, "wpstats.bin");
        wpStats.restrict(testSentencePairs).save(Execution.getFile("wpstats-test.bin"));
      }
      else {
        wpStats = new WordPairStats(trainSentencePairs);
        if(computeWpStats) { // Compute from these sentences
          wpStats.computeStats(trainSentencePairs);
          wpStats.restrict(testSentencePairs).save(Execution.getFile("wpstats-test.bin"));
        }
      }
    }

    // Build model
    Evaluator evaluator = new Evaluator(testSentencePairs, testAlignments, wpStats);
    WordAligner theWordAligner = null;
    if(model == ModelType.BASELINE) {
      // Stupid model
      WordAligner wordAligner = new BaselineWordAligner();
      evaluator.test(wordAligner, saveAlignOutput, false);
      theWordAligner = wordAligner;
    }
    else if(model == ModelType.OUTPUT_WPSTATS) {
      // Compute word pair stats
      List<SentencePair> allTrainSentencePairs = new ArrayList<SentencePair>();
      readSentencePairsFromSource(allTrainSource, 0, Integer.MAX_VALUE, strdb, allTrainSentencePairs);

      wpStats = new WordPairStats(allTrainSentencePairs);
      wpStats.computeStats(allTrainSentencePairs);
      WordPairStats subWpStats = wpStats.restrict(trainSentencePairs);

      subWpStats.save(Execution.getFile("wpstats.bin"));
      Execution.linkFileFromExec("wpstats.bin", wpStatsFile);
    }
    else if(model == ModelType.MODEL1 || model == ModelType.MODEL2 || model == ModelType.HMM) {
      // Decide which model to use
      SentencePairState.Factory spsFactory = null;
      if(model == ModelType.MODEL1)
        spsFactory = new Model1SentencePairState.Factory();
      else if(model == ModelType.MODEL2) {
        spsFactory = new Model2SentencePairState.Factory();
        DistortProbTable.setNumStates(1);
      }
      else if(model == ModelType.HMM) {
        spsFactory = new HMMSentencePairState.Factory();
        HMMSentencePairState.setFactory(HMMSentencePairState.stateType);
        DistortProbTable.setNumStates(HMMSentencePairState.factory.numDistortionGroups());
      }

      EMWordAligner wa1 = new EMWordAligner(spsFactory, evaluator, false);
      EMWordAligner wa2 = new EMWordAligner(spsFactory, evaluator, true);

      // Which models to train (1 normal vs. 2 reversed)
      boolean b1, b2;
      if(trainingMode == TrainingMode.BOTH_JOINT)
        b1 = b2 = true;
      else {
        b1 = trainingMode != TrainingMode.REVERSE;
        b2 = trainingMode != TrainingMode.NORMAL;
      }

      // Initialize
      if(b1) {
        if(!loadParamsDir.equals("")) wa1.loadParams(loadParamsDir);
        else wa1.initParams(wpStats);
      }
      if(b2) {
        if(!loadParamsDir.equals("")) wa2.loadParams(loadParamsDir);
        else wa2.initParams(wpStats);
      }

      // Train
      if(trainingMode == TrainingMode.BOTH_INDEP) {
        EMWordAligner.jointTrain(wa1, wa2, trainSentencePairs, numIters, false);
      }
      else if(trainingMode == TrainingMode.BOTH_JOINT) {
        EMWordAligner.jointTrain(wa1, wa2, trainSentencePairs, numIters, true);
      }
      else {
        if(b1) wa1.train(trainSentencePairs, numIters);
        if(b2) wa2.train(trainSentencePairs, numIters);
      }
      IntersectedWordAligner intwa = EMWordAligner.newIntersectedWordAligner(wa1, wa2);
      intwa.combineMethod = combineMethod;

      // Test
      if(testSentencePairs.size() > 0) {
        if(outputTestPosteriors) {
          track("Outputting test posteriors");
          // Decode a lot of sentences
          PrintWriter out = null;
          int sid = 0;
          int numBlock = 0;
          int numInBlock = 0;
          new File(Execution.getFile("testPosteriors")).mkdir();
          for(SentencePair sp : testSentencePairs) {
            logs("Sentence %d/%d", sid, testSentencePairs.size());
            if(out == null) {
              out = IOUtils.openOutHard(Execution.getFile(String.format("testPosteriors/file%04d", numBlock)));
              numBlock++;
            }

            List<Alignment> alignments = intwa.alignSentencePairReturnAll(sp);
            for(int j = 0; j < sp.J(); j++) {
              for(int i = 0; i < sp.I(); i++) {
                double s0 = alignments.get(0).getStrength(i, j);
                double s1 = alignments.get(1).getStrength(i, j);
                double s2 = alignments.get(2).getStrength(i, j);
                if(s0+s1+s2 > 1e-8)
                  out.printf("%d %d %d %s %s %s\n", sid+1, i+1, j+1,
                      Fmt.D(s0), Fmt.D(s1), Fmt.D(s2));
              }
            }
            sid++;
            numInBlock++;

            if(numInBlock >= 200) {
              out.close();
              out = null;
              numInBlock = 0;
            }
          }
          if(out != null) out.close();
          end_track();
        }
        else {
          boolean evalPRTradeoff = EMWordAligner.usePosteriorDecoding; 
          double aer = Double.NaN;
          if(b1) aer = evaluator.test(wa1, saveAlignOutput, evalPRTradeoff).aer;
          if(b2) aer = evaluator.test(wa2, saveAlignOutput, evalPRTradeoff).aer;
          if(b1 && b2) {
            Performance perf = evaluator.test(intwa, saveAlignOutput, evalPRTradeoff);
            aer = perf.aer;
            if(perf.hasBestAER()) Execution.putOutput("bestAER", Fmt.D(perf.bestAER));
            Execution.putOutput("bestThreshold", Fmt.D(perf.bestThreshold));
          }
          Execution.putOutput("AER", Fmt.D(aer));
        }
      }

      writeIntUnionAlignments(trainSentencePairs, intwa);

      // Save parameters
      if(saveParams) {
        if(b1) wa1.saveParams();
        if(b2) wa2.saveParams();
      }
      if(!b2)      theWordAligner = wa1;
      else if(!b1) theWordAligner = wa2;
      else         theWordAligner = intwa;
    }

    //dumpAlignedWordsParams(testAlignments, testSentencePairs);
  }

  // GIZA {
  /**
   * The goal of this exercise is to simulate what GIZA does with our intersected model.
   * Basically, we want to produce two sets of alignments which look like they
   * were produced by GIZA.
   * These alignments will be used to construct phrases.
   * The output should have the property that the intersection is the output
   * of the intersected model, and the union is the union of the two models.
   */
  private static void writeIntUnionAlignments(List<SentencePair> sentencePairs, IntersectedWordAligner intwa) {
    if(!alignIntUnion) return;
    track("Writing intersect/union alignments on %d sentences", sentencePairs.size());

    PrintWriter efOut = IOUtils.openOutHard(Execution.getFile("englishToFrench.giza"));
    PrintWriter feOut = IOUtils.openOutHard(Execution.getFile("frenchToEnglish.giza"));

    int idx = 0;
    for(SentencePair sp : sentencePairs) {
      logs("Sentence %d/%d", idx, sentencePairs.size());
      idx++;
      List<Alignment> a123 = intwa.alignSentencePairReturnAll(sp);
      /*Alignment a1 = wa1.alignSentencePair(sp);
      Alignment a2 = wa2.alignSentencePair(sp);
      // Alignments that should appear in both
      Alignment a3 = intwa.alignSentencePair(sp);*/
      Alignment a1 = a123.get(0); // E->F
      Alignment a2 = a123.get(1); // F->E
      Alignment a3 = a123.get(2); // Combined
      a3.writeGIZA(efOut, idx, sp);
      a1.union(a2).union(a3).reverse().writeGIZA(feOut, idx, sp.reverse());
    }

    efOut.close();
    feOut.close();

    end_track();
  }
  // }

  // Dump out the induced parameters from the alignments
  private static void dumpAlignedWordsParams(Map<Integer, Alignment> alignments,
      List<SentencePair> sentences) {
    Params params = new Params("Gold", false);
    for(SentencePair sp : sentences) {
      Alignment a = alignments.get(sp.sentenceID);
      for(Pair<Integer, Integer> pair : a.sureAlignments)
        params.transProbs.incr(sp.en(pair.getFirst()), sp.fr(pair.getSecond()), 1000);
      for(Pair<Integer, Integer> pair : a.possibleAlignments)
        params.transProbs.incr(sp.en(pair.getFirst()), sp.fr(pair.getSecond()), 1);
    }
    params.dump(IOUtils.openOutHard("from-alignments.params"), null, false);
  }

  private static void assertSentenceIDsAreUnique(List<SentencePair> sentencePairs) {
    Map<Integer, SentencePair> map = new HashMap<Integer, SentencePair>();
    for(SentencePair sp : sentencePairs) {
      int sid = sp.getSentenceID();
      if(map.containsKey(sid)) {
        throw new RuntimeException("Two sentences have same sentence ID: " + sid);
      }
      map.put(sid, sp);
    }
  }

  private static void readAlignmentsFromSource(String path, Map<Integer, Alignment> alignments) {
    track("readAlignments(" + path + ")");

    for(String baseFileName : getBaseFileNamesFromSource(path))
      readAlignmentsFromFile(alignments, baseFileName+".wa");

    end_track();
  }

  private static void readAlignmentsFromFile(Map<Integer, Alignment> alignments, String fileName) {
    if(!new File(fileName).exists()) return;

    try {
      BufferedReader in = IOUtils.openIn(fileName);
      while (in.ready()) {
        String line = in.readLine();
        String[] words = line.split("\\s+");
        if (words.length != 4)
          throw new RuntimeException("Bad alignment file "+fileName+", bad line was "+line);

        Integer sentenceID = Integer.parseInt(words[0]);
        Integer englishPosition = Integer.parseInt(words[1])-1;
        Integer frenchPosition = Integer.parseInt(words[2])-1;
        String type = words[3];

        Alignment alignment = alignments.get(sentenceID);
        if (alignment == null) {
          alignment = new Alignment();
          alignments.put(sentenceID, alignment);
        }
        alignment.addAlignment(englishPosition, frenchPosition, type.equals("S"));
      }
    } catch(IOException e) {
      throw new RuntimeException(e);
    }
  }

  // If path is a directory, return the list of files in the directory
  // If path is a file, return the files whose names are in the path file
  private static List<String> getBaseFileNamesFromSource(String path) {
    if(new File(path).isDirectory()) 
      return getBaseFileNamesFromDir(path);
    else {
      try {
        return OrderedStringMap.fromFile(path).keys();
      } catch(IOException e) {
        throw new RuntimeException(e);
      }
    }
  }

  private static void readSentencePairsFromSource(String path,
      int offset, int maxSentencePairs,
      Interner<String> strdb, List<SentencePair> sentencePairs) {
    track("readSentencePairs(" + path + ")");
    readSentencePairsUsingList(getBaseFileNamesFromSource(path), offset, maxSentencePairs, strdb, sentencePairs);
  }

  private static void readSentencePairsUsingList(List<String> baseFileNames,
      int offset, int maxSentencePairs, Interner<String> strdb, List<SentencePair> sentencePairs) {
    int numFiles = 0;
    for (String baseFileName : baseFileNames) {
      if (sentencePairs.size() >= maxSentencePairs)
        continue;
      numFiles++;
      logs("Reading " + numFiles + "/" + baseFileNames.size() + ": " + baseFileName);
      List<SentencePair> subSentencePairs = readSentencePairsFromFile(baseFileName,
        offset, maxSentencePairs - sentencePairs.size(), strdb);
      sentencePairs.addAll(subSentencePairs.subList(NumUtils.bound(offset, 0, subSentencePairs.size()), subSentencePairs.size()));
      offset -= subSentencePairs.size();
    }
    end_track();
  }

  // For sentence pairs before offset, just stick null instead of the actual sentence
  private static List<SentencePair> readSentencePairsFromFile(String baseFileName,
      int offset, int maxSentencePairs, Interner<String> strdb) {
    List<SentencePair> sentencePairs = new ArrayList<SentencePair>();
    String englishFileName = baseFileName + "." + enExt;
    String frenchFileName = baseFileName + "." + frExt;
    try {
      BufferedReader englishIn = IOUtils.openIn(englishFileName);
      BufferedReader frenchIn = IOUtils.openIn(frenchFileName);
      while (englishIn.ready() && frenchIn.ready()) {
        if(sentencePairs.size() >= maxSentencePairs) break;
        String englishLine = englishIn.readLine();
        String frenchLine = frenchIn.readLine();
        currSentenceID--;

        if(sentencePairs.size() < offset) {
          sentencePairs.add(null);
          continue; // Skip sentences before offset
        }

        Pair<Integer,List<String>> englishIDAndSentence = readSentence(englishLine, strdb);
        Pair<Integer,List<String>> frenchIDAndSentence = readSentence(frenchLine, strdb);

        int enID = englishIDAndSentence.getFirst();
        int frID = frenchIDAndSentence.getFirst();
        if(enID != frID)
          throw new RuntimeException("Sentence ID confusion in file "+baseFileName+", lines were:\n\t"+englishLine+"\n\t"+frenchLine);
        if(enID == -1) enID = frID = currSentenceID;
        sentencePairs.add(new SentencePair(enID, baseFileName, englishIDAndSentence.getSecond(), frenchIDAndSentence.getSecond()));
      }
    } catch(IOException e) {
      throw new RuntimeException(e);
    }
    return sentencePairs;
  }
  private static int currSentenceID = 0;

  private static Pair<Integer, List<String>> readSentence(String line, Interner<String> strdb) {
    int id = -1;
    List<String> words = new ArrayList<String>();
    String[] tokens = line.split("\\s+");
    for (int i = 0; i < tokens.length; i++) {
      String token = tokens[i];
      if (token.equals("<s")) continue;
      if (token.equals("</s>")) continue;
      if (token.startsWith("snum=")) {
        String idString = token.substring(5,token.length()-1);
        id = Integer.parseInt(idString);
        continue;
      }
      if(lowercaseWords)
        token = token.toLowerCase();
      words.add(token.intern());
      //words.add(strdb.intern(token));
      //words.add(token);
    }

    // For each ith occurence of the word (i > 1), append _<i>
    if(appendOccurToWord) {
      Map<String, Integer> hist = new HashMap<String, Integer>();
      for(int i = 0; i < words.size(); i++) {
        String word = words.get(i);
        MapUtils.incr(hist, word);
        int n = hist.get(word);
        if(n > 1) words.set(i, word + "_"+n);
      }
    }
    
    return new Pair<Integer, List<String>>(id, words);
  }

  private static List<String> getBaseFileNamesFromDir(String path) {
    //logs("getBaseFileNames(%s)", path);
    List<File> englishFiles = IOUtils.getFilesUnder(path, new FileFilter() {
      public boolean accept(File pathname) {
        if (pathname.isDirectory())
          return true;
        String name = pathname.getName();
        return name.endsWith(enExt);
      }
    });
    List<String> baseFileNames = new ArrayList<String>();
    for (File englishFile : englishFiles) {
      String baseFileName = chop(englishFile.getAbsolutePath(), "."+enExt);
      baseFileNames.add(baseFileName);
    }
    return baseFileNames;
  }

  private static String chop(String name, String extension) {
    if (! name.endsWith(extension)) return name;
    return name.substring(0, name.length()-extension.length());
  }
}
