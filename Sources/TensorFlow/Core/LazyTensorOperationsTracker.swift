/// A class to keep track of runtime information about `LazyTensorOperation`
/// instances created by the program. This will be manaaged as a thread local
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
    ) rethrows -> Void {
        for (_, counts) in refCounts where counts.liveRefCount > 0 {
            try perform(counts.op)
        }
    }

    func forEachOperation(
        _ perform: (LazyTensorOperation) throws -> Void
    ) rethrows -> Void {
        for (_, counts) in refCounts { try perform(counts.op) }
    }
}
