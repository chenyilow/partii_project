package fig.basic;

import java.io.*;
import java.util.*;
import java.net.*;

public class SysInfoUtils {
  public static String getCurrentDate() {
    return new Date().toString();
  }
  public static String getHostName() {
    try {
      return InetAddress.getLocalHost().getHostName();
    }
    catch(UnknownHostException e) {
      return "(unknown)";
    }
  }
  public static String getCPUSpeed() {
    return "fast";
  }
  public static String getMaxMemory() {
    long mem = Runtime.getRuntime().maxMemory();
    return Fmt.bytesToString(mem);
  }
  public static String getUsedMemory() {
    long totalMem = Runtime.getRuntime().totalMemory();
    long freeMem = Runtime.getRuntime().freeMemory();
    return Fmt.bytesToString(totalMem-freeMem);
  }
}
