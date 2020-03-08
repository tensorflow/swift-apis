public enum _RuntimeConfig {
  /// When true, prints various debug messages on the runtime state.
  ///
  /// If the value is true when running tensor computation for the first time in the process, INFO
  /// log from TensorFlow will also get printed.
  static public var printsDebugLog = false
}
