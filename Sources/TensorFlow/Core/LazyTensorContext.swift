/// A class to keep track of runtime information about `LazyTensorOperation`
/// instances created by the program. This will be managed as a thread local
/// state.
class LazyTensorOperationsTracker {
  struct RefCounts {
    let op: LazyTensorOperation
    let liveRefCount: Int
    let allRefCount: Int
  }

  private var refCounts: [ObjectIdentifier: RefCounts] = [:]

  func incrementRefCount(_ op: LazyTensorOperation, isLive: Bool) {
    let opID = ObjectIdentifier(op)
    if let counts = refCounts[opID] {
      refCounts[opID] = RefCounts(
        op: op,
        liveRefCount: counts.liveRefCount + (isLive ? 1 : 0),
        allRefCount: counts.allRefCount + 1)
    } else {
      refCounts[opID] = RefCounts(
        op: op, liveRefCount: isLive ? 1 : 0, allRefCount: 1)
    }
  }

  func decrementRefCount(_ op: LazyTensorOperation, isLive: Bool) {
    let opID = ObjectIdentifier(op)
    if let counts = refCounts[opID] {
      if counts.allRefCount > 1 {
        refCounts[opID] = RefCounts(
          op: op,
          liveRefCount: counts.liveRefCount - (isLive ? 1 : 0),
          allRefCount: counts.allRefCount - 1)
      } else {
        refCounts.removeValue(forKey: opID)
      }
    }
  }

  func isLive(_ op: LazyTensorOperation) -> Bool {
    let opID = ObjectIdentifier(op)
    if let counts = refCounts[opID] {
      return counts.liveRefCount > 0
    }
    return false
  }

  func forEachLiveOperation(
    _ perform: (LazyTensorOperation) throws -> Void
  ) rethrows {
    for (_, counts) in refCounts where counts.liveRefCount > 0 {
      try perform(counts.op)
    }
  }

  func forEachOperation(
    _ perform: (LazyTensorOperation) throws -> Void
  ) rethrows {
    for (_, counts) in refCounts { try perform(counts.op) }
  }
}

struct LazyTensorContext {
  var operationsTracker = LazyTensorOperationsTracker()
  var isShapeTrackingEnabled = true
  /// Should constants in trace be heuristically promoted to inputs automatically?
  /// (See `LazyTensorTraceCache`)
  var shouldPromoteConstants = true

  static var local: LazyTensorContext {
    _read { yield _ThreadLocalState.local.lazyTensorContext }
    _modify { yield &_ThreadLocalState.local.lazyTensorContext }
  }
}
