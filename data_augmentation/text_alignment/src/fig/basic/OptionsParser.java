package fig.basic;

import static fig.basic.LogInfo.*;
import java.io.*;
import java.util.*;
import java.lang.annotation.*;
import java.lang.reflect.*;

class OptInfo {
  public String group, name, gloss;
  public String condReq;
  public String hotTag;
  public boolean required;
  public boolean specified;
  public Object obj;
  public String stringRepn; // Used when obj is Random or BufferedReader (hard to get string)
  public Field field;

  public String fullName() { return group+"."+name; }

  // Return "" if field is not an enum type
  public String getEnumStr() { return getEnumStr(field.getType()); }
  public static String getEnumStr(Class c) {
    return StrUtils.join(c.getEnumConstants(), "|");
  }

  public Object getValue() {
    try {
      return field.get(obj);
    } catch(IllegalAccessException e) {
      stderr.println("Can't access field: " + e);
      return null;
    }
  }

  // Important to format properly in a way that we can read it and parse it again.
  public String getValueString() {
    if(stringRepn != null) return stringRepn;
    Object o = getValue();
    if(o == null) return "";
    if(o instanceof ArrayList)
      return StrUtils.join((ArrayList)o);
    if(o instanceof Random) // Argh, can't get the seed, just assume it's 1
      return "1";
    return o.toString();
  }

  public String toString() {
    String valueStr = getValueString();
    String s = String.format("%-30s <%5s> : %s [%s]",
      fullName(), typeStr(), gloss, valueStr);
    String t = getEnumStr();
    if(!t.equals("")) s += " " + t;
    return s;
  }
  public void print() { stdout.println("  " + toString()); }

  private String typeStr() {
    return typeStr(field.getGenericType());
  }

  private static boolean isEnum(Type type) {
    return type instanceof Class && ((Class)type).isEnum();
  }

  private static String typeStr(Type type) {
    if(type.equals(boolean.class) || type.equals(Boolean.class)) return "bool";
    if(type.equals(int.class) || type.equals(Integer.class))     return "int";
    if(type.equals(double.class) || type.equals(Double.class))   return "dbl";
    if(type.equals(String.class))         return "str";
    if(type.equals(BufferedReader.class)) return "read";
    if(type.equals(Random.class))         return "rand";
    if(isEnum(type))                      return "enum";
    if(type instanceof ParameterizedType) {
      ParameterizedType ptype = (ParameterizedType)type;
      type = ptype.getRawType();
      Type[] childTypes = ptype.getActualTypeArguments();
      if(type.equals(ArrayList.class))      return typeStr(childTypes[0]) + "*";
      if(type.equals(Pair.class))           return typeStr(childTypes[0]) + "2";
    }
    return "unk";
  }

  private static boolean checkNumArgs(int want, int have, String fullName) {
    if(have != want) {
      stderr.printf(want + " arguments required for " + fullName + ", but got " + have + "\n");
      return false;
    }
    return true;
  }

  // Return errorValue if there's an error (null is a valid value).
  // type: the data type of the variable
  // l: the command line arguments to interpret
  private static String errorValue = "ERROR";
  private static Object interpretValue(Type type, List<String> l, String fullName) {
    int n = l.size();
    String firstArg = n > 0 ? l.get(0) : null;

    if(type.equals(boolean.class) || type.equals(Boolean.class)) {
      boolean x = (n == 0 ? true : Boolean.parseBoolean(firstArg));
      return x;
    }
    if(type.equals(int.class) || type.equals(Integer.class)) {
      if(!checkNumArgs(1, n, fullName)) return errorValue;
      int x;
      if(firstArg.equals("MAX"))      x = Integer.MAX_VALUE;
      else if(firstArg.equals("MIN")) x = Integer.MIN_VALUE;
      else                            x = Integer.parseInt(firstArg);
      return x;
    }
    if(type.equals(double.class) || type.equals(Double.class)) {
      if(!checkNumArgs(1, n, fullName)) return errorValue;
      double x;
      if(firstArg.equals("MAX"))      x = Double.POSITIVE_INFINITY;
      else if(firstArg.equals("MIN")) x = Double.NEGATIVE_INFINITY;
      else                            x = Double.parseDouble(firstArg);
      return x;
    }
    if(type.equals(String.class)) { // Join many arguments using spaces
      String x = StrUtils.join(l);
      return x;
    }
    if(type.equals(BufferedReader.class)) {
      if(!checkNumArgs(1, n, fullName)) return errorValue;
      BufferedReader x = "-".equals(firstArg) ? LogInfo.stdin : IOUtils.openInHard(firstArg);
      return x;
    }
    if(type.equals(Random.class)) {
      if(!checkNumArgs(1, n, fullName)) return errorValue;
      // seed 0 means use the time
      int seed = Integer.parseInt(firstArg);
      Random x = seed == 0 ? new Random() : new Random(seed);
      return x;
    }
    if(type instanceof Class && ((Class)type).isEnum()) {
      if(n == 0) return null;
      if(!checkNumArgs(1, n, fullName)) return errorValue;
      Object x = Utils.parseEnum((Class)type, firstArg);
      if(x == null) {
        stderr.println("Invalid enum: '" + firstArg + "'; valid choices: " + getEnumStr((Class)type));
        return errorValue;
      }
      return x;
    }

    if(type instanceof ParameterizedType) {
      // Types involving generics: pair, arraylist
      ParameterizedType ptype = (ParameterizedType)type;
      type = ptype.getRawType();
      Type[] childTypes = ptype.getActualTypeArguments();

      if(type.equals(Pair.class)) { // Delimited by comma
        if(!checkNumArgs(1, n, fullName)) return errorValue;
        // Put the elements in the array
        String[] tokens = firstArg.split(",", 2);
        if(tokens.length != 2) {
          stderr.println("Invalid pair: '" + firstArg + "'");
          return errorValue;
        }
        Object o1 = interpretValue(childTypes[0], ListUtils.newList(tokens[0]), fullName);
        if(o1 == errorValue) return errorValue;
        Object o2 = interpretValue(childTypes[1], ListUtils.newList(tokens[1]), fullName);
        if(o2 == errorValue) return errorValue;
        return new Pair(o1, o2);
      }
      else if(type.equals(ArrayList.class)) {
        ArrayList x = new ArrayList();
        // Put the elements in the array
        for(String a : l) {
          Object o = interpretValue(childTypes[0], ListUtils.newList(a), fullName);
          if(o == errorValue) return errorValue;
          x.add(o);
        }
        return x;
      }
    }

    stderr.println("Can't handle weird field type: " + type);
    return errorValue;
  }

  public boolean set(List<String> l, boolean append) {
    try {
      Object v = interpretValue(field.getGenericType(), l, fullName());
      if(v == errorValue) return false;
      //System.out.println(name + " " + stringRepn + " " + v);
      if(!append) {
        stringRepn = StrUtils.join(l);
        field.set(obj, v);
      }
      else {
        Object oldv = field.get(obj);
        //System.out.println("append " + l);
        //System.out.println((oldv == null ? "" : (String)oldv + " ") + v);
        stringRepn = (stringRepn == null ? "" : stringRepn + " ") +
          StrUtils.join(l);
        if(oldv instanceof ArrayList)
          ((ArrayList)oldv).addAll((ArrayList)v);
        else if(oldv instanceof String)
          field.set(obj, (oldv == null ? "" : (String)oldv + " ") + v);
      }
    } catch(IllegalAccessException e) {
      stderr.println("Can't access field: " + e);
      return false;
    }

    specified = true;
    return true;
  }
}

/**
 * Due to historical reasons, all the member functions are prefixed with do,
 * and all the static functions (apply to the global theParser instance)
 * are not.
 */
public class OptionsParser {
  public OptionsParser() { }
  public OptionsParser(Object... objects) { doRegisterAll(objects); }

  public static void register(String group, Object o) { theParser.doRegister(group, o); }
  public void doRegister(String group, Object o) {
    objects.put(group, o);
  }
  public static void registerAll(Object[] objects) { theParser.doRegisterAll(objects); }
  public void doRegisterAll(Object[] objects) {
    // Strings are interpreted as the key name for the next object.
    String name = null;
    for(Object o : objects) {
      if(o instanceof String)
        name = (String)o;
      else {
        if(name == null) {
          if(o instanceof Class)
            name = ((Class)o).getSimpleName();
          else
            name = o.getClass().getSimpleName();
        }
        doRegister(name, o);
        name = null;
      }
    }
  }

  public OptionsParser relaxRequired() {
    this.relaxRequired = true;
    return this;
  }

  private static Class classOf(Object o) {
    return (o instanceof Class) ? (Class)o : o.getClass();
  }

  // Return null if not exactly one match
  private static OptInfo matchOpt(ArrayList<OptInfo> options, String s) {
    s = s.toLowerCase();

    ArrayList<OptInfo> matches = new ArrayList<OptInfo>();
    for(OptInfo opt : options) {
      String t;

      t = opt.fullName().toLowerCase();
      if(t.equals(s)) return opt;
      if(t.startsWith(s)) matches.add(opt);

      t = opt.name.toLowerCase();
      if(t.equals(s)) return opt;
      if(t.startsWith(s)) matches.add(opt);
    }

    if(matches.size() == 0) {
      stderr.println("Unknown option: '" + s + "'; -help for usage");
      return null;
    }
    else if(matches.size() == 1) {
      return matches.get(0);
    }
    else {
      stderr.println("Ambiguous option: '" + s + "'; possible matches:");
      for(OptInfo opt : matches) opt.print();
      return null;
    }
  }

  private static void printHelp(List<OptInfo> options) {
    stdout.println("Usage:");
    for(OptInfo opt : options)
      opt.print();
  }

  private ArrayList<OptInfo> getOptInfos() {
    ArrayList<OptInfo> options = new ArrayList<OptInfo>();

    // For each group...
    for(String group : objects.keySet()) {
      Object obj = objects.get(group);

      // For each field that has an option annotation...
      for(Field field : classOf(obj).getFields()) {
      //for(Field field : classOf(obj).getDeclaredFields()) {
        Option ann = (Option)field.getAnnotation(Option.class);
        if(ann == null) continue;

        // Get the option
        OptInfo opt = new OptInfo();
        opt.group = group;
        opt.name = ann.name().equals("") ? field.getName() : ann.name();
        opt.gloss = ann.gloss();
        opt.condReq = ann.condReq();
        opt.hotTag = ann.hotTag();
        opt.required = ann.required();
        opt.obj = obj;
        opt.field = field;
        options.add(opt);
      }
    }

    return options;
  }

  // Options file: one option per line
  // Key and value separated by tab.
  private static boolean readOptionsFile(ArrayList<OptInfo> options, String file) {
    boolean isDirectory = new File(file).isDirectory();
    if(isDirectory)
      file = new File(file, "options.map").toString();

    try {
      OrderedStringMap map = OrderedStringMap.fromFile(file);

      for(String key : map.keys()) {
        if(key.startsWith("#")) continue; // Skip comments
        boolean append = false;

        String val = map.get(key);
        if(val == null) val = "";
        if(key.startsWith("+")) { append = true; key = key.substring(1); }

        if(key.equals("!include")) { // Include other file
          if(!readOptionsFile(options, val)) return false;
        }
        else {
          OptInfo opt = matchOpt(options, key);
          if(opt == null) continue; // Skip fields that don't exist
          if(isDirectory && opt.group.equals("exec") && opt.name.equals("execDir")) continue; // HACK: do not load execution directory
          if(!opt.set(Arrays.asList(StrUtils.split(val)), append)) return false;
        }
      }
    } catch(IOException e) {
      stderr.println(e);
      return false;
    }
    return true;
  }

  public boolean parseOptionsFile(String path) {
    ArrayList<OptInfo> options = getOptInfos();
    return readOptionsFile(options, path);
  }

  // Return true iff x is a strict prefix of
  private static boolean isStrictPrefixOf(String x, String... ys) {
    for(String y : ys)
      if(x.startsWith(y) && x.length() > y.length()) return true;
    return false;
  }

  public static boolean parse(String[] args) { return theParser.doParse(args); }
  public void doParseHard(String[] args) {
    if(!doParse(args))
      throw new RuntimeException("Parsing '" + StrUtils.join(args) + "' failed");
  }
  public boolean doParse(String[] args) {
    if(this.options == null) this.options = getOptInfos();
    StringBuilder hotSB = new StringBuilder();

    // For each command-line argument...
    for(int i = 0; i < args.length;) {
      if(args[i].equals("-help")) { // Get usage help
        printHelp(options);
        return false;
      }
      else if(isStrictPrefixOf(args[i], "++")) {
        if(!readOptionsFile(options, args[i++].substring(2)))
          return false;
      }
      else if(isStrictPrefixOf(args[i], "-", "+")) {
        boolean append = args[i].startsWith("+");
        OptInfo opt = matchOpt(options, args[i++].substring(1));
        if(opt == null) return false;

        // Get the data values of this parameter
        ArrayList<String> l = new ArrayList<String>();
        boolean nextIsVerbatim = false;
        while(i < args.length) {
          if(args[i].equals("--"))
            nextIsVerbatim = true;
          else {
            if(!nextIsVerbatim && (isStrictPrefixOf(args[i], "+", "-", "++")))
              break;
            l.add(args[i]);
            nextIsVerbatim = false;
          }
          i++;
        }
        if(!opt.set(l, append)) return false;
        if(!StrUtils.isEmpty(opt.hotTag))
          hotSB.append(" "+opt.hotTag+"="+StrUtils.join(l));
      }
      else {
        stderr.println("Argument not part of an option: " + args[i]);
        return false;
      }
    }

    // Check that all required options are specified
    if(!relaxRequired) {
      List<String> missingOptMsgs = new ArrayList<String>();
      for(OptInfo o : options) {
        String msg = isMissing(o, options);
        if(msg != null) missingOptMsgs.add(msg);
      }
      if(missingOptMsgs.size() > 0) {
        stderr.println("Missing required option(s):");
        for(String msg : missingOptMsgs)
          stderr.println(msg);
        return false;
      }
    }

    hotSpec = hotSB.toString();
    if(hotSpec.length() > 0) hotSpec = hotSpec.substring(1);

    return true;
  }

  // Return the option info with the given name (which could be full or not).
  // If not, then prepend the given group.
  private OptInfo findOptInfo(List<OptInfo> optInfos, String name, String group) {
    for(OptInfo info : optInfos)
      if(info.fullName().equals(name)) return info;
    name = group + "." + name;
    for(OptInfo info : optInfos)
      if(info.fullName().equals(name)) return info;
    return null;
  }

  // If the option is missing, return the message (to be printed out) of why
  // Otherwise, return null
  private String isMissing(OptInfo o, List<OptInfo> optInfos) {
    if(o.specified) return null; // Specified, we're fine
    if(o.required) return o.toString(); // This option is required
    if(!StrUtils.isEmpty(o.condReq)) {
      // This option is conditionally required
      String[] tokens = o.condReq.split("=", 2);
      String name = tokens[0], value = tokens.length == 2 ? tokens[1] : null;
      OptInfo info = findOptInfo(optInfos, name, o.group);
      boolean missing;
      if(info == null) // Shouldn't happen, but if it does, the user will be notified
        return o.toString() + ", " + name + " not found";
      else if(value == null) { // Just need to be specified
        if(info.specified) return o.toString() + ", " + name + " specified";
      }
      else {
        if(value.equals(info.getValueString()))
          return o.toString() + ", " + o.condReq + " holds";
      }
    }
    return null;
  }

  // Return a list of options (verbose - human-readable)
  public static OrderedStringMap getOptionStrings() { return theParser.doGetOptionStrings(); }
  public OrderedStringMap doGetOptionStrings() {
    if(this.options == null) this.options = getOptInfos();
    OrderedStringMap map = new OrderedStringMap();
    for(OptInfo opt : options)
      map.put(opt.toString());
    return map;
  }

  // Return a list of option pairs (mapping name to value)
  public static OrderedStringMap getOptionPairs() { return theParser.doGetOptionPairs(); }
  public OrderedStringMap doGetOptionPairs() {
    if(this.options == null) this.options = getOptInfos();
    OrderedStringMap map = new OrderedStringMap();
    for(OptInfo opt : options)
      map.put(opt.fullName(), opt.getValueString());
    return map;
  }

  public boolean writeEasy(String path) {
    return doGetOptionPairs().printEasy(path);
  }

  public String getHotSpec() { return hotSpec; }

  // Each object could either be a class or an object.
  private HashMap<String, Object> objects = new HashMap<String, Object>();
  private ArrayList<OptInfo> options;
  private String hotSpec;
  private boolean relaxRequired; // Forget about having to have all options

  public static final OptionsParser theParser = new OptionsParser();
}
