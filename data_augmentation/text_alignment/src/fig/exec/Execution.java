package fig.exec;

import java.io.*;
import java.util.*;
import java.lang.Thread;

import fig.basic.*;
import static fig.basic.LogInfo.*;
import fig.record.*;

/**
 * Represents all the settings and output of an execution of a program.
 * An execution is defined by all the options registered with OptionsParser.
 * Creates a directory for the execution in the execution pool dir.
 */
public class Execution {
  @Option(gloss="Directory which contains all the executions (or symlinks).")
    public static String execPoolDir = "execs";
  @Option(gloss="Directory which actually holds the executions.")
    public static String actualExecPoolDir = null;
  @Option(gloss="Whether to create a directory for this run; if not, don't generate output files")
    public static boolean create = false;
  @Option(gloss="Whether to create a thread to monitor the status.")
    public static boolean monitor = false;
  @Option(gloss="Simply print options and exit.")
    public static boolean printOptionsAndExit = false;
  @Option(gloss="Write the name of the exec directory to this file.")
    public static String writeExecDirName = "";
  @Option(gloss="Name of jar files to load prior to execution")
    public static ArrayList<String> jarFiles = new ArrayList<String>();
  @Option(gloss="Character encoding")
    public static String charEncoding;
  @Option(gloss="Name of the view to add this execution to in the servlet")
    public static String addToView;
  @Option(gloss="Miscellaneous options (for displaying in servlet)")
    public static String miscOptions;

  @Option(gloss="Use this to group executions")
    public static String execId;
  @Option(gloss="Directory to put all output files; if blank, use execPoolDir.")
    public static String execDir;

  private static String actualExecDir;

  static OrderedStringMap inputMap = new OrderedStringMap();
  private static OrderedStringMap outputMap = new OrderedStringMap();
  static MonitorThread monitorThread; // Thread for monitoring
  static int exitCode = 0;

  private static void mkdirHard(File f) {
    if(!f.mkdir()) {
      stderr.println("Cannot create directory: " + f);
      System.exit(1);
    }
  }

  public static String getActualExecDir() { return actualExecDir; }

  /**
   * Return an unused directory in the execution pool directory.
   * Set actualExecDir
   */
  public static String createActualExecDir() {
    if(execDir != null) { // Use specified execDir
      mkdirHard(new File(execDir));
      return actualExecDir = execDir;
    }

    // Get a list of files that already exists
    Set<String> files = new HashSet<String>();
    for(String f : new File(execPoolDir).list()) files.add(f);

    // Go through and pick out a file that doesn't exist
    int numFailures = 0;
    for(int i = 0; numFailures < 3; i++) {
      // Possibly a link
      File f = new File(execPoolDir, i+".exec");
      // Actual file
      File g = StrUtils.isEmpty(actualExecPoolDir) ? null : new File(actualExecPoolDir, i+".exec");

      if(!files.contains(i+".exec") && (g == null || !g.exists())) {
        if(g == null || g.equals(f)) {
          mkdirHard(f);
          return actualExecDir = f.toString();
        }
        if(Utils.createSymLink(g.toString(), f.toString())) {
          mkdirHard(g);
          return actualExecDir = f.toString();
        }

        // Probably because someone else already linked to it
        // in the race condition: so try again
        stderr.println("Cannot create symlink from " + f + " to " + g);
        numFailures++;
      }
    }
    throw Exceptions.bad("Failed many times to create execution directory");
  }

  // Get the path of the file (in the execution directory)
  public static String getFile(String file) {
    if(StrUtils.isEmpty(actualExecDir)) return null;
    if(StrUtils.isEmpty(file)) return null;
    return new File(actualExecDir, file).toString();
  }

  public static void linkFileToExec(String realFileName, String file) {
    if(StrUtils.isEmpty(realFileName) || StrUtils.isEmpty(file)) return;
    File f = new File(realFileName);
    Utils.createSymLink(f.getAbsolutePath(), getFile(file));
  }
  public static void linkFileFromExec(String file, String realFileName) {
    if(StrUtils.isEmpty(realFileName) || StrUtils.isEmpty(file)) return;
    File f = new File(realFileName);
    Utils.createSymLink(getFile(file), f.getAbsolutePath());
  }

  // Getting input and writing output
  public static boolean getBooleanInput(String s) {
    String t = inputMap.get(s, "0");
    return t.equals("true") || t.equals("1");
  }
  public static String getInput(String s) { return inputMap.get(s); }
  public synchronized static void putOutput(String s, Object t) { outputMap.put(s, StrUtils.toString(t)); }
  public synchronized static void printOutputMapToStderr() { outputMap.print(stderr); }
  public synchronized static void printOutputMap(String path) {
    if(StrUtils.isEmpty(path)) return;
    // First write to a temporary directory and then rename the file
    String tmpPath = path+".tmp";
    if(outputMap.printEasy(tmpPath))
      new File(tmpPath).renameTo(new File(path));
  }

  public static void setExecStatus(String newStatus, boolean override) {
    String oldStatus = outputMap.get("exec.status");
    if(oldStatus == null || oldStatus.equals("running")) override = true;
    if(override) putOutput("exec.status", newStatus);
  }

  static OrderedStringMap getInfo() {
    OrderedStringMap map = new OrderedStringMap();
    map.put("Date", SysInfoUtils.getCurrentDate());
    map.put("Host", SysInfoUtils.getHostName());
    map.put("CPU speed", SysInfoUtils.getCPUSpeed());
    map.put("Max memory", SysInfoUtils.getMaxMemory());
    return map;
  }

  public static void init(String[] args) {
    OptionsParser.register("log", LogInfo.class);
    OptionsParser.register("exec", Execution.class);
    if(!OptionsParser.parse(args)) System.exit(1);

    for(String jarFile : jarFiles) // Load classes
      ClassInitializer.initializeJar(jarFile);
    if(charEncoding != null) // Set character encoding
      CharEncUtils.setCharEncoding(charEncoding);

    if(printOptionsAndExit) { // Just print options and exit
      OptionsParser.getOptionPairs().print(stdout);
      System.exit(0);
    }

    // Create a new directory
    if(create) {
      createActualExecDir();
      stderr.println(actualExecDir);
      if(!StrUtils.isEmpty(writeExecDirName)) {
        try {
          IOUtils.filePrintln(writeExecDirName, actualExecDir);
        } catch(IOException e) {
          stderr.println("Can't write exec dir: " + writeExecDirName);
          System.exit(1);
        }
      }
      LogInfo.file = getFile("log");
    }
    else {
      LogInfo.file = "";
    }

    LogInfo.init();
    track("main()", true);

    // Output options
    if(actualExecDir != null) logs("Execution directory: " + actualExecDir);
    getInfo().printEasy(getFile("info.map"));
    printOptions();
    putOutput("hotOptions", OptionsParser.theParser.getHotSpec());
    if(!StrUtils.isEmpty(addToView))
      IOUtils.filePrintlnEasy(Execution.getFile("addToView"), addToView);

    // Start monitoring
    if(monitor) {
      monitorThread = new MonitorThread();
      monitorThread.start();
    }

    Record.init(Execution.getFile("record"));
  }

  // Might want to call this again after some command-line options were changed.
  public static void printOptions() {
    OptionsParser.getOptionPairs().printEasy(getFile("options.map"));
    OptionsParser.getOptionStrings().printEasy(getFile("options.help"));
  }

  public static void raiseException(Throwable t) {
    error(t + ":\n" + StrUtils.join(t.getStackTrace(), "\n"));
    t = t.getCause();
    if(t != null)
      error("Caused by " + t + ":\n" + StrUtils.join(t.getStackTrace(), "\n"));
    Execution.putOutput("exec.status", "exception");
    exitCode = 1;
  }

  public static void finish() {
    Record.finish();

    if(monitor) monitorThread.finish();
    setExecStatus("done", false);
    outputMap.printEasy(getFile("output.map"));
    StopWatch.getStats().printEasy(getFile("time.map"));
    if(create) stderr.println(actualExecDir);
    if(LogInfo.getNumErrors() > 0 || LogInfo.getNumWarnings() > 0)
      stderr.printf("%d errors, %d warnings\n",
          LogInfo.getNumErrors(), LogInfo.getNumWarnings());
    end_track();
    System.exit(exitCode);
  }

  // This should be all we need to put in a main function.
  // args are the commandline arguments
  // First object is the Runnable object to call run on.
  // All of them are objects whose options args is to supposed to populate.
  public static void run(String[] args, Object... objects) {
    OptionsParser.registerAll(objects);
    init(args);
    try {
      ((Runnable)objects[0]).run();
    } catch(Throwable t) {
      raiseException(t);
    }
    finish();
  }
}
