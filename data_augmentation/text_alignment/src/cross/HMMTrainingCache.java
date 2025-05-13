package cross;

import java.util.*;

public class HMMTrainingCache extends TrainingCache {
  public WATrellis getTrellis(WAState.Factory factory, int I, Params params) {
    WATrellis trellis = trellisCache.get(I);
    if(trellis == null) {
      trellis = new WATrellis(factory, I, params);
      // TODO: don't put it in the cache if I is too large?
      if(I <= 100)
        trellisCache.put(I, trellis);
    }
    return trellis;
  }

  public void clear() { trellisCache.clear(); }

  Map<Integer, WATrellis> trellisCache = new HashMap<Integer, WATrellis>();
}
