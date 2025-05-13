package cross;

import java.io.*;
import java.util.*;

import fig.exec.*;
import fig.basic.*;
import static fig.basic.LogInfo.*;

/**
 * GizaAlignments store additional information extracted from GIZA++ data files,
 * including the text, alignment score, sentence number, and filename.
 * 
 * @author John DeNero
 * 
 */
public class GizaReader {
  protected String fileName;
  protected BufferedReader fileReader;
  
  public GizaReader(String fileName) throws IOException /*FileNotFoundException*/ {
    this.fileName = fileName;
    //fileReader = new BufferedReader(new FileReader(fileName));
    fileReader = IOUtils.openIn(fileName);
  }

  /**
   * Takes a GIZA++ alignment output file and returns a list of Alignments.
   * 
   * @param fileName
   *           the input file name
   * @return Returns a list of GizaAlignment objects.
   */
  public static AlignmentsInfo readAlignments(String fileName) throws IOException, RuntimeException {
    List<SentencePair> sentencePairs = new ArrayList<SentencePair>();
    Map<Integer, Alignment> alignments = new HashMap<Integer, Alignment>();
    //BufferedReader in = new BufferedReader(new FileReader(fileName));
    BufferedReader in = IOUtils.openIn(fileName);
    while (in.ready()) {
      if(sentencePairs.size() >= maxSentences) break;
      /*
       * GIZA++ alignments come in sets of three lines. An example with
       * added notation for new lines:
       * 
       * (line 1) # Sentence pair (1) source length 2 target length 2
       * alignment score : 0.0457472
       * 
       * (line 2) <CHAPTER ID=1>
       * 
       * (line 3) NULL ({ }) <CHAPTER ({ 1 }) ID=1> ({ 2 })
       */
      String infoline = in.readLine();
      String[] infowords = infoline.split("\\s+");
      if (infowords.length != 14)
        throw new RuntimeException(
            "Bad alignment file "
                + fileName
                + ": wrong number of words in input line. Bad line was "
                + infoline);
      if (!infowords[0].equals("#"))
        throw new RuntimeException("Bad alignment file " + fileName
            + ": input line without initial #. Bad line was "
            + infoline);

      Integer sentenceID = Integer.parseInt(infowords[3].substring(1,
          infowords[3].length() - 1));
      Double score = Double.parseDouble(infowords[13]);

      String frenchline = in.readLine();
      List<String> frenchWords = Arrays.asList(frenchline
          .split("\\s+"));

      String englishline = in.readLine();
      List<String> englishInput = Arrays.asList(englishline
          .split("\\s+"));

      List<String> englishWords = new ArrayList<String>();
      int alignmentsFromFrenchToEnglish[] = new int[frenchWords.size() + 1];
      // First, we extract the alignments and english words from the english.
      // Format: NULL ({ }) <CHAPTER ({ 1 }) ID=1> ({ 2 })
      int englishPosition = 0;
      int inputPosition = 0;
      while (inputPosition < englishInput.size()) {
        if (englishPosition != 0) { // Not the NULL word
          englishWords.add(englishInput.get(inputPosition));
          inputPosition++;
          if (!"({".equals(englishInput.get(inputPosition)))
            throw new RuntimeException(
                "Improperly formed english input string at sentence #"
                    + sentenceID);
          inputPosition++;
          while (!"})".equals(englishInput.get(inputPosition))) {
            try {
              int french = Integer
                  .parseInt(englishInput.get(inputPosition));
              alignmentsFromFrenchToEnglish[french] = englishPosition;
              inputPosition++;
            } catch (NumberFormatException nfe) {
              throw new RuntimeException(
                  "Improperly formed english input string at sentence #"
                   + sentenceID);
            }
          }
        } else { // Skip past NULL information
          while (!"})".equals(englishInput.get(inputPosition))) {
            inputPosition++;
          }
        }
        inputPosition++;
        englishPosition++;
      }

      // Then, we build an alignment structure.
      Alignment alignment = new Alignment();
      for (int frenchPosition = 1; frenchPosition <= frenchWords.size(); frenchPosition++) {
        // Here, we account for the fact that GIZA++ output is
        // 1-indexed while our alignment class is 0-indexed.
        alignment.addAlignment(
            alignmentsFromFrenchToEnglish[frenchPosition] - 1,
            frenchPosition - 1, true);
      }

      SentencePair sp = new SentencePair(sentenceID, fileName, englishWords, frenchWords);
      sentencePairs.add(sp);
      alignments.put(sentenceID, alignment);
    }
    return new AlignmentsInfo(fileName, sentencePairs, new HashMap<Integer, Alignment>(), alignments);
  }

  @Option(gloss="File with giza alignments", required=true)
    public static String inFile;
  @Option(gloss="Output file (text format)")
    public static String txtOutFile = null;
  @Option(gloss="Output file (binary format)")
    public static String binOutFile = null;
  @Option(gloss="Maximum number of sentences to convert")
    public static int maxSentences = Integer.MAX_VALUE;
  
  /**
   * Testing utility that outputs rendered alignment diagrams from a particular
   * GIZA++ output file.
   * 
   * @param args
   */
  public static void main(String args[]) throws Exception {
    OptionsParser.register("main", GizaReader.class);
    Execution.init(args);

    track("Reading alignments");
    AlignmentsInfo ainfo = readAlignments(inFile);
    end_track();

    track("Writing %d alignments", ainfo.proposedAlignments.size());
    if(txtOutFile != null) ainfo.writeText(txtOutFile, null);
    if(binOutFile != null) ainfo.writeBinary(binOutFile);
    end_track();

    Execution.finish();
  }
}
