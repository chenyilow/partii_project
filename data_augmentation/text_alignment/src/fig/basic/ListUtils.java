package fig.basic;

import java.util.*;
import java.lang.reflect.*;

public class ListUtils {
  public static ArrayList<Double> toList(double[] xs) {
    ArrayList<Double> list = new ArrayList();
    for(double x : xs) list.add(x);
    return list;
  }
  public static<T> ArrayList<T> toList(Iterable<T> it) {
    if(it instanceof ArrayList) return (ArrayList)it;
    ArrayList<T> list = new ArrayList<T>();
    for(T x : it) list.add(x);
    return list;
  }

  public static<T> ArrayList<T> newList(T... list) {
    return new ArrayList<T>(Arrays.asList(list));
  }

  public static<T> ArrayList<T> newListFill(T x, int n) {
    ArrayList<T> list = new ArrayList<T>(n);
    for(int i = 0; i < n; i++) list.add(x);
    return list;
  }

  public static int maxStringLength(List<String> strings) {
    int l = 0;
    for(String s : strings)
      l = Math.max(l, s.length());
    return l;
  }

  public static <T> Map<T, Integer> buildHistogram(Collection<T> c) {
    Map<T, Integer> counts = new HashMap<T, Integer>();
    for(T x : c) MapUtils.incr(counts, x);
    return counts;
  }

  public static <T> void randomPermute(List<T> l, Random rand) {
    for(int i = 0; i < l.size(); i++) {
      int j = i+rand.nextInt(l.size()-i);
      T x = l.get(i);
      l.set(i, l.get(j));
      l.set(j, x);
    }
  }

  public static <T> T getLast(List<T> l) { return get(l, -1); }
  public static <T> T get(List<T> l, int i) { return get(l, i, null); }
  public static <T> T get(List<T> l, int i, T defValue) {
    if(i < 0) i += l.size();
    if(i < 0 || i >= l.size()) return defValue;
    return l.get(i);
  }
  public static <T> T removeLast(List<T> l) {
    return l.remove(l.size()-1);
  }
  public static <T> T get(T[] l, int i, T defValue) {
    if(i < 0) i += l.length;
    if(i < 0 || i >= l.length) return defValue;
    return l[i];
  }
  public static double get(double[] l, int i, double defValue) {
    if(i < 0) i += l.length;
    if(i < 0 || i >= l.length) return defValue;
    return l[i];
  }

  // Return the array (0, 1, 2, ..., n-1)
  public static int[] identityMapArray(int n) {
    int[] arr = new int[n];
    for(int i = 0; i < n; i++) arr[i] = i;
    return arr;
  }

  public static int maxIndex(double[] list) {
    int bi = -1;
    for(int i = 0; i < list.length; i++)
      if(bi == -1 || list[i] > list[bi]) bi = i;
    return bi;
  }
  public static double max(double[] list) {
    double m = Double.NEGATIVE_INFINITY;
    for(double x : list) m = Math.max(m, x);
    return m;
  }
  public static double sum(double[] list) {
    double sum = 0;
    for(double x : list) sum += x;
    return sum;
  }
  public static double sum(List<Double> list) {
    double sum = 0;
    for(double x : list) sum += x;
    return sum;
  }

  public static double[] expMut(double[] list) {
    for(int i = 0; i < list.length; i++)
      list[i] = Math.exp(list[i]);
    return list;
  }
  public static double[] exp(double[] list) {
    double[] newlist = new double[list.length];
    for(int i = 0; i < list.length; i++)
      newlist[i] = Math.exp(list[i]);
    return newlist;
  }

  // Return a permuted array.
  // Example:
  //   data = (A, B, C), perm = (2, 0, 1)
  //   newData = (C, A, B)
  public static int[] applyPermutation(int[] data, int[] perm) {
    assert data.length == perm.length;
    int[] newData = new int[data.length];
    for(int i = 0; i < data.length; i++)
      newData[i] = data[perm[i]];
    return newData;
  }
  public static double[] applyPermutation(double[] data, int[] perm) {
    assert data.length == perm.length;
    double[] newData = new double[data.length];
    for(int i = 0; i < data.length; i++)
      newData[i] = data[perm[i]];
    return newData;
  }
  public static <T> T[] applyPermutation(T[] data, int[] perm) {
    assert data.length == perm.length;
    T[] newData = newArray(data);
    for(int i = 0; i < data.length; i++)
      newData[i] = data[perm[i]];
    return newData;
  }
  public static double[] applyInversePermutation(double[] data, int[] perm) {
    assert data.length == perm.length;
    double[] newData = new double[data.length];
    for(int i = 0; i < data.length; i++)
      newData[perm[i]] = data[i];
    return newData;
  }
  public static int[] inversePermutation(int[] perm) {
    int[] newperm = new int[perm.length];
    for(int i = 0; i < perm.length; i++)
      newperm[perm[i]] = i;
    return newperm;
  }
  public static void assertIsPermutation(int[] perm) {
    boolean hit[] = new boolean[perm.length];
    for(int i : perm) {
      assert !hit[i];
      hit[i] = true;
    }
  }

  public static int[] append(int[] a, int[] b) {
    int[] c = new int[a.length+b.length];
    int j = 0;
    for(int i = 0; i < a.length; i++, j++) c[j] = a[i];
    for(int i = 0; i < b.length; i++, j++) c[j] = b[i];
    return c;
  }
  public static double[] append(double[] a, double[] b) {
    double[] c = new double[a.length+b.length];
    int j = 0;
    for(int i = 0; i < a.length; i++, j++) c[j] = a[i];
    for(int i = 0; i < b.length; i++, j++) c[j] = b[i];
    return c;
  }
  public static <T> T[] append(T[] a, T[] b) {
    T[] c = newArray(a.length+b.length, a[0]);
    int j = 0;
    for(int i = 0; i < a.length; i++, j++) c[j] = a[i];
    for(int i = 0; i < b.length; i++, j++) c[j] = b[i];
    return c;
  }

  public static Double[] toObjArray(double[] v) {
    Double[] newv = new Double[v.length];
    for(int i = 0; i < v.length; i++) newv[i] = v[i];
    return newv;
  }

  public static boolean[] shallowClone(boolean[] v) {
    if(v == null) return null;
    return (boolean[])v.clone();
  }
  public static int[] shallowClone(int[] v) {
    if(v == null) return null;
    return (int[])v.clone();
  }
  public static double[][] shallowClone(double[][] v) {
    if(v == null) return null;
    double[][] newv = new double[v.length][];
    for(int i = 0; i < v.length; i++)
      newv[i] = shallowClone(v[i]);
    return newv;
  }
  public static double[] shallowClone(double[] v) {
    if(v == null) return null;
    return (double[])v.clone();
  }
  public static <T> T[][] shallowClone(T[][] v) {
    if(v == null) return null;
    T[][] newv = newArray(v);
    for(int i = 0; i < v.length; i++) {
      newv[i] = newArray(v[i]);
      for(int j = 0; j < v[i].length; j++)
        newv[i][j] = v[i][j]; // Don't make a copy
    }
    return newv;
  }
  public static <T> T[] shallowClone(T[] v) {
    if(v == null) return null;
    T[] newv = newArray(v);
    for(int i = 0; i < v.length; i++)
      newv[i] = v[i]; // Don't make a copy
    return newv;
  }

  public static <T> T[] deepClone(T[] v) {
    if(v == null) return null;
    T[] newv = newArray(v);
    for(int i = 0; i < v.length; i++)
      newv[i] = ((DeepCloneable<T>)v[i]).deepClone();
    return newv;
  }

  public static <T> T[] newArray(T[] v) {
    return (T[])Array.newInstance(v.getClass().getComponentType(), v.length);
  }

  public static double[] newDouble(int n, double x) {
    double[] v = new double[n];
    Arrays.fill(v, x);
    return v;
  }
  public static int[] newInt(int n, int x) {
    int[] v = new int[n];
    Arrays.fill(v, x);
    return v;
  }

  public static interface Generator<T> {
    public T generate(int i);
  }
  public static <T> T[] newArray(int n, Class c, Generator<T> gen) {
    T[] a = (T[])Array.newInstance(c, n);
    for(int i = 0; i < n; i++) a[i] = gen.generate(i);
    return a;
  }
  public static <T> T[] newArray(int n, T x) {
    T[] a = (T[])Array.newInstance(x.getClass(), n);
    for(int i = 0; i < n; i++) a[i] = x;
    return a;
  }

  public static double[] mult(double f, double[] vec) {
    double[] newVec = new double[vec.length];
    for(int i = 0; i < vec.length; i++)
      newVec[i] = f * vec[i];
    return newVec;
  }

  // v1 += factor * v2
  public static double[] incr(double[] v1, double factor, double[] v2) {
    for(int i = 0; i < v1.length; i++)
      v1[i] += factor * v2[i];
    return v1;
  }
  public static double[] incr(double[] v1, double x) {
    for(int i = 0; i < v1.length; i++)
      v1[i] += x;
    return v1;
  }

  public static double[] add(double[] v1, double[] v2) {
    double[] sumv = new double[v1.length];
    for(int i = 0; i < v1.length; i++)
      sumv[i] = v1[i] + v2[i];
    return sumv;
  }
  public static double[] add(double[] v1, int[] v2) {
    double[] sumv = new double[v1.length];
    for(int i = 0; i < v1.length; i++)
      sumv[i] = v1[i] + v2[i];
    return sumv;
  }
  public static double[] sub(double[] v1, double[] v2) {
    double[] sumv = new double[v1.length];
    for(int i = 0; i < v1.length; i++)
      sumv[i] = v1[i] - v2[i];
    return sumv;
  }
  public static double[] add(double[] v1, double x) {
    double[] sumv = new double[v1.length];
    for(int i = 0; i < v1.length; i++)
      sumv[i] = v1[i] + x;
    return sumv;
  }
  public static double[] mult(double[] v1, double[] v2) {
    double[] v = new double[v1.length];
    for(int i = 0; i < v1.length; i++)
      v[i] = v1[i] * v2[i];
    return v;
  }

  public static double dot(double[] v1, double[] v2) {
    double sum = 0;
    for(int i = 0; i < v1.length; i++)
      sum += v1[i] * v2[i];
    return sum;
  }

  public static double[] sqrt(double[] v) {
    double[] newv = new double[v.length];
    for(int i = 0; i < v.length; i++)
      newv[i] = Math.sqrt(v[i]);
    return newv;
  }

  public static double[] reverse(double[] v) {
    double[] newv = new double[v.length];
    for(int i = 0; i < v.length; i++)
      newv[i] = v[v.length-i-1];
    return newv;
  }

  public static int[] toArray(List<Integer> list) {
    int[] array = new int[list.size()];
    for(int i = 0; i < array.length; i++)
      array[i] = list.get(i);
    return array;
  }
  public static double[] toArray(List<Double> list) {
    double[] array = new double[list.size()];
    for(int i = 0; i < array.length; i++)
      array[i] = list.get(i);
    return array;
  }
  public static String[] toArray(List<String> list) {
    String[] data = new String[list.size()];
    for(int i = 0; i < data.length; i++)
      data[i] = list.get(i);
    return data;
  }

  // Take subsequence [start, end)
  public static String[] subArray(String[] v, int start) {
    return subArray(v, start, v.length);
  }
  public static String[] subArray(String[] v, int start, int end) {
    String[] subv = new String[end-start];
    for(int i = start; i < end; i++)
      subv[i-start] = v[i];
    return subv;
  }
  public static double[] subArray(double[] v, int start) {
    return subArray(v, start, v.length);
  }
  public static double[] subArray(double[] v, int start, int end) {
    double[] subv = new double[end-start];
    for(int i = start; i < end; i++)
      subv[i-start] = v[i];
    return subv;
  }
  public static <T> T[] subArray(T[] v, int start, int end) {
    T[] subv = newArray(end-start, v[0]);
    for(int i = start; i < end; i++)
      subv[i-start] = v[i];
    return subv;
  }
  public static <T> T[] subArray(T[] v, List<Integer> indices) {
    T[] newv = newArray(indices.size(), v[0]);
    for(int i = 0; i < indices.size(); i++)
      newv[i] = v[indices.get(i)];
    return newv;
  }
  // If bounds are invalid, clip them.
  public static <T> List<T> subList(List<T> list, int start) {
    return subList(list, start, list.size());
  }
  public static <T> List<T> subList(List<T> list, int start, int end) {
    if(end < 0) end += list.size();
    if(start < 0) start += list.size();
    start = NumUtils.bound(start, 0, list.size());
    end = NumUtils.bound(end, 0, list.size());
    return list.subList(start, end);
  }

  public static <T> void partialSort(List<T> list, int numTop, Comparator<? super T> c) {
    Object[] a = list.toArray();
    partialSort(a, numTop, (Comparator)c);
    ListIterator<T> i = list.listIterator();
    for(int j=0; j<a.length; j++) {
      i.next();
      i.set((T)a[j]);
    }
  }

  public static <T> void partialSort(T[] list, int numTop, Comparator<? super T> c) {
    // Select out the numTop-th ranked element
    // TODO
    Arrays.sort(list, c); // For now, sort everything
  }

  // Return the indices: the first element contains the smallest
  public static int[] sortedIndices(double[] list, boolean reverse) {
    int n = list.length;
    // Sort
    List<Pair<Double,Integer>> pairList = new ArrayList<Pair<Double,Integer>>(n);
    for(int i = 0; i < n; i++)
      pairList.add(new Pair<Double,Integer>(list[i], i));
    Collections.sort(pairList,
      reverse ? new Pair.ReverseFirstComparator<Double,Integer>()
              : new Pair.FirstComparator<Double,Integer>());
    // Extract the indices
    int[] indices = new int[n];
    for(int i = 0; i < n; i++)
      indices[i] = pairList.get(i).getSecond();
    return indices;
  }

  public static int[] toInt(boolean[] v) {
    int[] newv = new int[v.length];
    for(int i = 0; i < v.length; i++)
      newv[i] = v[i] ? 1 : 0;
    return newv;
  }
}
