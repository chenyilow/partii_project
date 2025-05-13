package fig.basic;

import java.io.*;
import java.util.*;

public class Utils {
  // Create a random from another mother random.
  // This is useful when a program needs to use randomness
  // for two tasks, each of which requires an unknown
  // number random draws.  For partial reproducibility.
  public static Random randRandom(Random random) {
    return new Random(random.nextInt(Integer.MAX_VALUE));
  }

  public static void sleep(int ms) {
    try { Thread.sleep(ms); }
    catch(InterruptedException e) { }
  }

  public static Properties loadProperties(String path) {
    Properties properties = new Properties();
    try {
      properties.load(new FileInputStream(path));
      return properties;
    } catch(IOException e) {
      throw new RuntimeException("Cannot open " + path);
    }
  }

  public static boolean createSymLink(String src, String dest) {
    try {
      // -n: if destination is a symbolic link (to a directory), can't overwrite it
      String cmd = String.format("ln -sn %s %s", src, dest);
      try { return Runtime.getRuntime().exec(cmd).waitFor() == 0; }
      catch(InterruptedException e) { return false; }
    }
    catch(IOException e) {
      return false;
    }
  }

  // Get stack traces
  // Include the top max stack traces
  // Stop when reach stopClassPrefix
  public static String getStackTrace(Throwable t, int max, String stopClassPrefix) {
    StringBuilder sb = new StringBuilder();
    for(StackTraceElement e : t.getStackTrace()) {
      if(max-- <= 0) break;
      if(stopClassPrefix != null && e.getClassName().startsWith(stopClassPrefix)) break;
      sb.append(e);
      sb.append('\n');
    }
    return sb.toString();
  }
  public static String getStackTrace(Throwable t) {
    return getStackTrace(t, Integer.MAX_VALUE, null);
  }
  public static String getStackTrace(Throwable t, int max) {
    return getStackTrace(t, max, null);
  }
  public static String getStackTrace(Throwable t, String classPrefix) {
    return getStackTrace(t, Integer.MAX_VALUE, classPrefix);
  }

  public static int parseIntEasy(String s, int defaultValue) {
    if(s == null) return defaultValue;
    try { return Integer.parseInt(s); }
    catch(NumberFormatException e) { return defaultValue; }
  }
  public static double parseDoubleEasy(String s) {
    return parseDoubleEasy(s, Double.NaN);
  }
  public static double parseDoubleEasy(String s, double defaultValue) {
    if(s == null) return defaultValue;
    try { return Double.parseDouble(s); }
    catch(NumberFormatException e) { return defaultValue; }
  }
  public static boolean parseBooleanEasy(String s, boolean defaultValue) {
    if(s == null) return defaultValue;
    try { return Boolean.parseBoolean(s); }
    catch(NumberFormatException e) { return defaultValue; }
  }

  public static int parseIntHard(String s) {
    try { return Integer.parseInt(s); }
    catch(NumberFormatException e) { throw new RuntimeException("Invalid format: " + s); }
  }
  public static double parseDoubleHard(String s) {
    try { return Double.parseDouble(s); }
    catch(NumberFormatException e) { throw new RuntimeException("Invalid format: " + s); }
  }
  public static boolean parseBooleanHard(String s) {
    try { return Boolean.parseBoolean(s); }
    catch(NumberFormatException e) { throw new RuntimeException("Invalid format: " + s); }
  }

  public static Object parseEnum(Class c, String s) {
    s = s.toLowerCase();
    for(Object o : c.getEnumConstants())
      if(o.toString().toLowerCase().equals(s))
        return o;
    return null;
  }

  // Convert Integer/Double object to a double
  public static double toDouble(Object o) {
    if(o instanceof Double) return (Double)o;
    if(o instanceof Integer) return (double)((Integer)o);
    throw Exceptions.unknownCase;
  }

  // Return number of seconds
  // 1d2h4m2s
  public static int parseTimeLength(String s) {
    if(StrUtils.isEmpty(s)) return 0;
    int sum = 0;
    int n = 0;
    for(int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      if(Character.isDigit(c))
        n = n * 10 + Integer.parseInt(c+"");
      else if(c == 'd') { sum += n * 60 * 60 * 24; n = 0; }
      else if(c == 'h') { sum += n * 60 * 60;      n = 0; }
      else if(c == 'm') { sum += n * 60;           n = 0; }
      else if(c == 's') { sum += n;                n = 0; }
    }
    return sum;
  }

  // Run shell commands
  public static Process openSystem(String cmd) throws IOException {
    return Runtime.getRuntime().exec(new String[] { "sh", "-c", cmd });
  }
  public static int closeSystem(String cmd, Process p) throws InterruptedException {
    return p.waitFor();
  }
  public static int closeSystemEasy(String cmd, Process p) {
    try { return closeSystem(cmd, p); }
    catch(InterruptedException e) { return -1; }
  }
  public static void closeSystemHard(String cmd, Process p) {
    try {
      int status = closeSystem(cmd, p);
      if(status != 0)
        throw new RuntimeException("Failed: '" + cmd + "' returned status " + status);
    } catch(InterruptedException e) {
      throw new RuntimeException(e);
    }
  }
  // Run the command
  // Assume command takes no stdin
  // Dump command's output to stdout, stderr
  // Throw a nasty exception if anything fails.
  public static void systemHard(String cmd) {
    try {
      Process p = openSystem(cmd);
      PrintWriter out = CharEncUtils.getWriter(p.getOutputStream());
      out.close();
      BufferedReader in = CharEncUtils.getReader(p.getInputStream());
      IOUtils.copy(in, LogInfo.stdout); in.close();
      BufferedReader err = CharEncUtils.getReader(p.getErrorStream());
      IOUtils.copy(err, LogInfo.stderr); err.close();
      closeSystemHard(cmd, p);
    } catch(Exception e) {
      throw new RuntimeException(e);
    }
  }
}
