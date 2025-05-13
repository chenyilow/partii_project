package cross;

public class UnionWordAligner extends WordAligner {
  WordAligner wa1, wa2;
  public UnionWordAligner(WordAligner wa1, WordAligner wa2) {
    this.wa1 = wa1;
    this.wa2 = wa2;
  }

  public Alignment alignSentencePair(SentencePair sentencePair) {
    Alignment a1 = wa1.alignSentencePair(sentencePair);
    Alignment a2 = wa2.alignSentencePair(sentencePair);
    return a1.union(a2);
  }

  public String getName() {
    return "Union(" + wa1.getName() + ", " + wa2.getName() + ")";
  }
}
