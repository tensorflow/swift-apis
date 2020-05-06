// Copyright 2020 TensorFlow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import TensorFlow

typealias GradMapFn = (inout Tensor<Float>, Tensor<Float>, Int) -> Void
// A partially type erased base-class that essentially provides a
// WritableKeypath<Base, Child> + a TensorVisitorPlan<Child> and is used for
// doing the recursive walk over the structure tree.
class TensorVisitorPlanWrapperBase<Base> {
  func appendKeyPaths<TrueBase>(
    _ rootKeyPath: WritableKeyPath<TrueBase, Base>,
    _ kps: inout [WritableKeyPath<TrueBase, Tensor<Float>>]
  ) {
    fatalError("calling abstract function")
  }

  func findFirstIndex<TrueBase, T>(
    _ rootKeyPath: WritableKeyPath<TrueBase, Base>,
    _ prefix: WritableKeyPath<TrueBase, T>, _ i: inout Int
  ) -> Bool {
    fatalError("calling abstract function")
  }

  func populateTensors(_ v: Base, _ tensors: inout [Tensor<Float>]) {
    fatalError("calling abstract function")
  }

  func mapTensors(_ v1: inout Base, _ v2: Base, _ i: inout Int, _ fn: GradMapFn) {
    fatalError("calling abstract function")
  }

  func populateMask<Base>(_ mask: inout [Bool], _ kp: WritableKeyPath<Base, Tensor<Float>>) {
    fatalError("calling abstract function")
  }
}

// The basic implementation of TensorVisitorPlanWrapperBase for normally extending
// keypaths.
final class TensorVisitorPlanWrapper<Base, Child>: TensorVisitorPlanWrapperBase<Base> {
  var child: WritableKeyPath<Base, Child>
  var childPlan: TensorVisitorPlan<Child>
  init(child: WritableKeyPath<Base, Child>, childPlan: TensorVisitorPlan<Child>) {
    self.child = child
    self.childPlan = childPlan
  }

  override func appendKeyPaths<TrueBase>(
    _ rootKeyPath: WritableKeyPath<TrueBase, Base>,
    _ kps: inout [WritableKeyPath<TrueBase, Tensor<Float>>]
  ) {
    childPlan.appendKeyPaths(rootKeyPath.appending(path: child), &kps)
  }

  override func populateTensors(_ v: Base, _ tensors: inout [Tensor<Float>]) {
    childPlan.populateTensors(v[keyPath: child], &tensors)
  }

  override func mapTensors(_ v1: inout Base, _ v2: Base, _ i: inout Int, _ fn: GradMapFn) {
    childPlan.mapTensors(&v1[keyPath: child], v2[keyPath: child], &i, fn)
  }

  override func populateMask<Base>(_ mask: inout [Bool], _ kp: WritableKeyPath<Base, Tensor<Float>>)
  {
    childPlan.populateMask(&mask, kp)
  }

  override func findFirstIndex<TrueBase, T>(
    _ rootKeyPath: WritableKeyPath<TrueBase, Base>,
    _ prefix: WritableKeyPath<TrueBase, T>, _ i: inout Int
  ) -> Bool {
    childPlan.findFirstIndex(rootKeyPath.appending(path: child), prefix, &i)
  }
}

// Faster implementation for iterating over nested arrays.
final class ArrayTensorVisitorPlanWrapper<Base, Child>: TensorVisitorPlanWrapperBase<Base> {
  var child: WritableKeyPath<Base, [Child]>
  var childPlans: [TensorVisitorPlan<Child>]
  init(child: WritableKeyPath<Base, [Child]>, childPlans: [TensorVisitorPlan<Child>]) {
    self.child = child
    self.childPlans = childPlans
  }

  override func appendKeyPaths<TrueBase>(
    _ rootKeyPath: WritableKeyPath<TrueBase, Base>,
    _ kps: inout [WritableKeyPath<TrueBase, Tensor<Float>>]
  ) {
    let childKp = rootKeyPath.appending(path: child)
    for (i, childPlan) in childPlans.enumerated() {
      childPlan.appendKeyPaths(childKp.appending(path: \[Child][i]), &kps)
    }
  }

  override func populateTensors(_ v: Base, _ tensors: inout [Tensor<Float>]) {
    let arr = v[keyPath: child]
    for (i, childPlan) in childPlans.enumerated() {
      childPlan.populateTensors(arr[i], &tensors)
    }
  }

  override func mapTensors(_ v1: inout Base, _ v2: Base, _ i: inout Int, _ fn: GradMapFn) {
    { (arr1: inout [Child], arr2: [Child]) in
      for (j, childPlan) in childPlans.enumerated() {
        childPlan.mapTensors(&arr1[j], arr2[j], &i, fn)
      }
    }(&v1[keyPath: child], v2[keyPath: child])
  }

  override func populateMask<Base>(_ mask: inout [Bool], _ kp: WritableKeyPath<Base, Tensor<Float>>)
  {
    for childPlan in childPlans { childPlan.populateMask(&mask, kp) }
  }

  override func findFirstIndex<TrueBase, T>(
    _ rootKeyPath: WritableKeyPath<TrueBase, Base>,
    _ prefix: WritableKeyPath<TrueBase, T>, _ i: inout Int
  ) -> Bool {
    let childKp = rootKeyPath.appending(path: child)
    if childKp == prefix { return true }
    for (j, childPlan) in childPlans.enumerated() {
      let elementKp = childKp.appending(path: \[Child][j])
      if elementKp == prefix || childPlan.findFirstIndex(elementKp, prefix, &i) {
        return true
      }
    }
    return childKp.appending(path: \[Child][childPlans.count]) == prefix
  }
}

// Faster implementation for iterating over nested DifferentiableViews.
// Also improves the firstIndex() ui because \DifferentiableView.base[i] != \DifferentiableView[i]
final class ArrayDifferentiableTensorVisitorPlanWrapper<Base, Child>: TensorVisitorPlanWrapperBase<
  Base
>
where Child: Differentiable {
  var child: WritableKeyPath<Base, Array<Child>.DifferentiableView>
  var childPlans: [TensorVisitorPlan<Child>]
  init(
    child: WritableKeyPath<Base, Array<Child>.DifferentiableView>,
    childPlans: [TensorVisitorPlan<Child>]
  ) {
    self.child = child
    self.childPlans = childPlans
  }

  override func appendKeyPaths<TrueBase>(
    _ rootKeyPath: WritableKeyPath<TrueBase, Base>,
    _ kps: inout [WritableKeyPath<TrueBase, Tensor<Float>>]
  ) {
    let childKp = rootKeyPath.appending(path: child)
    for (i, childPlan) in childPlans.enumerated() {
      childPlan.appendKeyPaths(
        childKp.appending(path: \Array<Child>.DifferentiableView.base[i]), &kps)
    }
  }

  override func populateTensors(_ v: Base, _ tensors: inout [Tensor<Float>]) {
    let arr = v[keyPath: child]
    for (i, childPlan) in childPlans.enumerated() {
      childPlan.populateTensors(arr[i], &tensors)
    }
  }

  override func mapTensors(_ v1: inout Base, _ v2: Base, _ i: inout Int, _ fn: GradMapFn) {
    { (arr1: inout Array<Child>.DifferentiableView, arr2: Array<Child>.DifferentiableView) in
      for (j, childPlan) in childPlans.enumerated() {
        childPlan.mapTensors(&arr1[j], arr2[j], &i, fn)
      }
    }(&v1[keyPath: child], v2[keyPath: child])
  }

  override func populateMask<Base>(_ mask: inout [Bool], _ kp: WritableKeyPath<Base, Tensor<Float>>)
  {
    for childPlan in childPlans { childPlan.populateMask(&mask, kp) }
  }

  override func findFirstIndex<TrueBase, T>(
    _ rootKeyPath: WritableKeyPath<TrueBase, Base>,
    _ prefix: WritableKeyPath<TrueBase, T>, _ i: inout Int
  ) -> Bool {
    let childKp = rootKeyPath.appending(path: child)
    if childKp == prefix { return true }
    for (j, childPlan) in childPlans.enumerated() {
      let elementKp = childKp.appending(path: \Array<Child>.DifferentiableView[j])
      if elementKp == prefix || childPlan.findFirstIndex(elementKp, prefix, &i) {
        return true
      }
    }
    return childKp.appending(path: \Array<Child>.DifferentiableView[childPlans.count]) == prefix
  }
}

/// TensorVisitorPlan approximates `[WritableKeyPath<Base, Tensor<Float>]` but
/// is more efficient. This is useful for writing generic optimizers which want
/// to map over the gradients, the existing weights, and an index which can be
/// used to find auxiliarily stored weights. This is slightly more efficient (~2x) but it could
/// be better because it trades off slightly higher overheads (extra pointer dereference)
/// for not having to do O(depth_of_tree) work that is required with a plain list to track
/// down each individual KeyPath.
public struct TensorVisitorPlan<Base> {
  enum Impl {
    case leaf(WritableKeyPath<Base, Tensor<Float>>)
    case node(TensorVisitorPlanWrapperBase<Base>)
  }
  var elements: [Impl] = []

  func appendKeyPaths<TrueBase>(
    _ rootKeyPath: WritableKeyPath<TrueBase, Base>,
    _ kps: inout [WritableKeyPath<TrueBase, Tensor<Float>>]
  ) {
    for item in elements {
      switch item {
      case .leaf(let kp):
        kps.append(rootKeyPath.appending(path: kp))
      case .node(let plan):
        plan.appendKeyPaths(rootKeyPath, &kps)
      }
    }
  }

  /// Flatten out the plan as a single `[WritableKeyPath<Base, Tensor<Float>]`.
  public var allTensorKeyPaths: [WritableKeyPath<Base, Tensor<Float>>] {
    var kps = [WritableKeyPath<Base, Tensor<Float>>]()
    appendKeyPaths(\Base.self, &kps)
    return kps
  }

  func populateTensors(_ v: Base, _ tensors: inout [Tensor<Float>]) {
    for item in elements {
      switch item {
      case .leaf(let kp):
        tensors.append(v[keyPath: kp])
      case .node(let plan):
        plan.populateTensors(v, &tensors)
      }
    }
  }

  /// Efficiently collect all the tensors.
  public func allTensors(_ v: Base) -> [Tensor<Float>] {
    var tensors = [Tensor<Float>]()
    populateTensors(v, &tensors)
    return tensors
  }

  /// Efficiently map over two values of type `Base` and apply a mapping function.
  /// Returns the number of tensors. The extra `Int` argument is provided to allow indexing
  /// into an auxiliary list of Tensors with the same Tensor count as the plan.
  @discardableResult
  public func mapTensors(
    _ v1: inout Base, _ v2: Base, _ fn: (inout Tensor<Float>, Tensor<Float>, Int) -> Void
  ) -> Int {
    var i = 0
    mapTensors(&v1, v2, &i, fn)
    return i
  }

  func mapTensors(_ v1: inout Base, _ v2: Base, _ i: inout Int, _ fn: GradMapFn) {
    for item in elements {
      switch item {
      case .leaf(let kp):
        let _ = fn(&v1[keyPath: kp], v2[keyPath: kp], i)
        i += 1
      case .node(let plan):
        plan.mapTensors(&v1, v2, &i, fn)
      }
    }
  }
}

protocol VisitorPlanBuilder {
  func _buildWrappedVisitorPlan<Base>(_ rootKeyPath: PartialKeyPath<Base>)
    -> TensorVisitorPlanWrapperBase<Base>?
}

extension Array: VisitorPlanBuilder {
  func _buildWrappedVisitorPlan<Base>(_ rootKeyPath: PartialKeyPath<Base>)
    -> TensorVisitorPlanWrapperBase<Base>?
  {
    if let kp = rootKeyPath as? WritableKeyPath<Base, Self> {
      var nonEmpty = false
      var plans = [TensorVisitorPlan<Element>]()
      for element in self {
        guard let element = element as? _KeyPathIterableBase else { return nil }
        var plan = TensorVisitorPlan<Element>()
        element._populateTensorVisitorPlan(&plan)
        nonEmpty = nonEmpty || !plan.elements.isEmpty
        plans.append(plan)
      }
      if nonEmpty {
        return ArrayTensorVisitorPlanWrapper(child: kp, childPlans: plans)
      }
    }
    return nil
  }
}

extension Array.DifferentiableView: VisitorPlanBuilder where Element: Differentiable {
  func _buildWrappedVisitorPlan<Base>(_ rootKeyPath: PartialKeyPath<Base>)
    -> TensorVisitorPlanWrapperBase<Base>?
  {
    if let kp = rootKeyPath as? WritableKeyPath<Base, Self> {
      var nonEmpty = false
      var plans = [TensorVisitorPlan<Element>]()
      for element in self {
        guard let element = element as? _KeyPathIterableBase else { return nil }
        var plan = TensorVisitorPlan<Element>()
        element._populateTensorVisitorPlan(&plan)
        nonEmpty = nonEmpty || !plan.elements.isEmpty
        plans.append(plan)
      }
      if nonEmpty {
        return ArrayDifferentiableTensorVisitorPlanWrapper(child: kp, childPlans: plans)
      }
    }
    return nil
  }
}

extension _KeyPathIterableBase {
  func _buildWrappedVisitorPlan<Base>(
    _ rootKeyPath: PartialKeyPath<Base>
  ) -> TensorVisitorPlanWrapperBase<Base>? {
    if let kp = rootKeyPath as? WritableKeyPath<Base, Self> {
      var plan = TensorVisitorPlan<Self>()
      _populateTensorVisitorPlan(&plan)
      if !plan.elements.isEmpty {
        return TensorVisitorPlanWrapper(child: kp, childPlan: plan)
      }
    }
    return nil
  }

  func _populateTensorVisitorPlan<Base>(_ plan: inout TensorVisitorPlan<Base>) {
    for kp in _allKeyPathsTypeErased {
      if let kp = kp as? WritableKeyPath<Base, Tensor<Float>> {
        plan.elements.append(.leaf(kp))
      } else if let nested = self[keyPath: kp] as? VisitorPlanBuilder {
        if let child = nested._buildWrappedVisitorPlan(kp as! PartialKeyPath<Base>) {
          plan.elements.append(.node(child))
        }
      } else if let nested = self[keyPath: kp] as? _KeyPathIterableBase {
        if let child = nested._buildWrappedVisitorPlan(kp as! PartialKeyPath<Base>) {
          plan.elements.append(.node(child))
        }
      }
    }
  }
}

extension TensorVisitorPlan where Base: KeyPathIterable {
  /// Creates a plan to visit all the tensors in a particular instance of `Base`.
  /// This plan is transferable to structurally equivalent versions of Base.
  public init(_ obj: Base) {
    obj._populateTensorVisitorPlan(&self)
  }
}

extension TensorVisitorPlan {
  func populateMask<Base>(_ mask: inout [Bool], _ kp: WritableKeyPath<Base, Tensor<Float>>) {
    for item in elements {
      switch item {
      case .leaf(let okp):
        mask.append(kp == okp)
      case .node(let plan):
        plan.populateMask(&mask, kp)
      }
    }
  }

  /// Find all keys ending with a particular key-path.
  public func keysEnding<Base>(with kp: WritableKeyPath<Base, Tensor<Float>>) -> [Bool] {
    var mask = [Bool]()
    populateMask(&mask, kp)
    return mask
  }

  func findFirstIndex<TrueBase, T>(
    _ rootKeyPath: WritableKeyPath<TrueBase, Base>,
    _ prefix: WritableKeyPath<TrueBase, T>, _ i: inout Int
  ) -> Bool {
    if rootKeyPath == prefix { return true }
    for item in elements {
      switch item {
      case .leaf(let kp):
        if rootKeyPath.appending(path: kp) == prefix { return true }
        i += 1
      case .node(let plan):
        if plan.findFirstIndex(rootKeyPath, prefix, &i) { return true }
      }
    }
    return false
  }

  /// Find the index of the first keypath starting with a particular prefix.
  /// Note: All array layers support 1-past-the-end indexing.
  func firstIndex<T>(withPrefix prefix: WritableKeyPath<Base, T>) -> Int {
    var i = 0
    let _ = findFirstIndex(\Base.self, prefix, &i)
    return i
  }

  /// Find all keys indices in a range defined by two KeyPath prefixes: [lower, upper)
  public func allKeysBetween<T, U>(lower: WritableKeyPath<Base, T>, upper: WritableKeyPath<Base, U>)
    -> [Bool]
  {
    let range = firstIndex(withPrefix: lower)..<firstIndex(withPrefix: upper)
    return allTensorKeyPaths.indices.map { range.contains($0) }
  }
}

extension Array where Element == Bool {
  /// Computes `a || b` elementwise as though we were or-ing together
  /// two masks.
  public func mergingMask(with other: [Bool]) -> [Bool] {
    precondition(count == other.count)
    return indices.map { i in self[i] || other[i] }
  }
}
