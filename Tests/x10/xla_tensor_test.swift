import TensorFlow
import XCTest

/// Direct tests of xla tensor.
final class XLATensorTests: XCTestCase {
  func testLazyTensorBarrier() throws {
    let x = Tensor<Float>(20, on: Device.defaultXLA) * Tensor<Float>(30, on: Device.defaultXLA)
    LazyTensorBarrier()
    XCTAssertEqual(x.scalarized(), 20 * 30)
  }

  func testAnnotationsTFEager() throws {
    let tensor = Tensor<Float>(repeating: 0, shape: [1, 2, 3], on: Device.defaultTFEager)
    XCTAssertEqual(tensor.annotations, "Annotations not available in TF_EAGER.")
    let annotated = tensor.annotate("type=Tensor<Float>")
    XCTAssertEqual(annotated.annotations, "Annotations not available in TF_EAGER.")
  }

  func testAnnotationsXLA() throws {
    let tensor = Tensor<Float>(repeating: 0, shape: [1, 2, 3], on: Device.defaultXLA)
    XCTAssertEqual(tensor.annotations, "{\n}")
    let annotated = tensor.annotate("type=Tensor<Float>")
    XCTAssertEqual(annotated.annotations, "{\n  shape=[1, 2, 3] type=Tensor<Float>\n}")
  }
}

extension XLATensorTests {
  static var allTests = [
    ("testLazyTensorBarrier", testLazyTensorBarrier),
    ("testAnnotationsTFEager", testAnnotationsTFEager),
    ("testAnnotationsXLA", testAnnotationsXLA),
  ]
}

final class MultiDeviceAPITests: XCTestCase {
  func testGetAllDevices() {
    XCTAssertFalse(Device.allDevices.isEmpty)
  }

  func testTensorDevice() {
    let allDevices = Device.allDevices
    let tpuDevices = allDevices.filter { $0.kind == .TPU }
    let dims = [2]
    let seed = 47
    let content = _Raw.rand(dims, seed)
    if tpuDevices.isEmpty {
      let cpuDevice = Device(kind: .CPU, ordinal: 0, backend: .XLA)
      XCTAssertEqual(content.device, cpuDevice)
    }
    let tpuTensors = tpuDevices.map { _Raw.toDevice(content, $0) }
    for (tpuTensor, tpuDevice) in zip(tpuTensors, tpuDevices) {
      XCTAssertEqual(tpuTensor.device, tpuDevice)
    }
  }

  func testSetGetReplication() {
    let allDevices = Device.allDevices
    let tpuDevices = allDevices.filter { $0.kind == .TPU }
    Device.setReplicationDevices(tpuDevices)
    XCTAssertEqual(Device.getReplicationDevices(), tpuDevices)
    Device.setReplicationDevices([])
  }

  func testSyncLiveTensors() {
    let allDevices = Device.allDevices
    let tpuDevices = allDevices.filter { $0.kind == .TPU }
    Device.syncLiveTensorsForDevices(tpuDevices)
  }

  func testCrossReplicaSum() {
    let allDevices = Device.allDevices
    let tpuDevices = allDevices.filter { $0.kind == .TPU }
    let dims = [2]
    let seed = 47
    let content = [_Raw.rand(dims, seed), _Raw.rand(dims, seed)]
    let tpuTensors = tpuDevices.map { tpuDevice in content.map { _Raw.toDevice($0, tpuDevice) } }
    let results = tpuTensors.map { _Raw.crossReplicaSum($0, 1.0) }
    Device.syncLiveTensorsForDevices(tpuDevices)
    let count = Tensor(Float(tpuDevices.count), on: Device.defaultXLA)
    let axes = Tensor<Int32>(0..<Int32(content[0].rank), on: Device.defaultXLA)
    for (result, tpuDev) in zip(results, tpuDevices) {
      for i in 0..<content.count {
        let countDev = _Raw.toDevice(count, tpuDev)
        let contentDev = _Raw.toDevice(content[i], tpuDev)
        XCTAssertTrue(
          _Raw.all(_Raw.approximateEqual(result[i], contentDev * countDev), reductionIndices: axes)
            .scalarized())
      }
    }
  }

  func testFunctionalWhile() {
    let res = (_RawXLA.functionalWhile(n: Tensor<Int32>(4, on: .defaultXLA),
               initial: [Tensor<Float>(3.0, on: .defaultXLA)]) { args, i in
      var a = args[0] as! Tensor<Float>
      a = a * 2 + 1;
      return [a]
    })[0] as! Tensor<Float>
    XCTAssertEqual(res.scalarized(), 63)
  }
}

extension MultiDeviceAPITests {
  static var allTests = [
    ("testGetAllDevices", testGetAllDevices),
    ("testTensorDevice", testTensorDevice),
    ("testSetGetReplication", testSetGetReplication),
    ("testSyncLiveTensors", testSyncLiveTensors),
    ("testCrossReplicaSum", testCrossReplicaSum),
    ("testFunctionalWhile", testFunctionalWhile),
  ]
}

XCTMain([
  testCase(XLATensorTests.allTests),
  testCase(MultiDeviceAPITests.allTests),
])
