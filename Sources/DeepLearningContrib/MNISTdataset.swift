import Foundation
import TensorFlow

/// MNIST constants
fileprivate enum constants {
    static let URL = "http://yann.lecun.com/exdb/mnist/"
    static let trainImagesFile = "train-images-idx3-ubyte.gz"
    static let trainLabelsFile = "train-labels-idx1-ubyte.gz"
    static let testImagesFile = "t10k-images-idx3-ubyte.gz"
    static let testLabelsFile = "t10k-labels-idx1-ubyte.gz"
    static let trainImages = "train-images-idx3-ubyte"
    static let trainLabels = "train-labels-idx1-ubyte"
    static let testImages = "t10k-images-idx3-ubyte"
    static let testLabels = "t10k-label-idx1-ubyte" 
}

/// To download and extract files from the URL
fileprivate func downloadAndExtract() {
    let helperFiles = [constants.trainImagesFile,
                       constants.trainLabelsFile,
                       constants.testImagesFile,
                       constants.testLabelsFile]
    let extractedFiles = [constants.trainImages,
                          constants.trainLabels,
                          constants.testImages,
                          constants.testLabels]
    var counter = 0
    let path = FileManager.default.currentDirectoryPath
    for helperFile in helperFiles {
        let url = NSURL(fileURLWithPath: path)
        if let pathComponent = url.appendingPathComponent(helperFile) {
            let filePath = pathComponent.path
            let fileManager = FileManager.default
            if fileManager.fileExists(atPath: filePath) {
                print("Not downloading \(helperFile): already exists")
            } else {
                print("Downloading \(helperFile)")
                /// TODO: Add Swift Implementation for downloading files.
                print("Downloaded \(helperFile)")
            }
        }    
    }
    for extractedFile in extractedFiles {
        let url = NSURL(fileURLWithPath: path)
        if let pathComponent = url.appendingPathComponent(extractedFile) {
            let filePath = pathComponent.path
            let fileManager = FileManager.default
            if fileManager.fileExists(atPath: filePath) {
                print("File \(extractedFile): already exists")
            } else {
                print("Extracting \(extractedFile)")
                /// TODO: Add Swift Implementation for extracting downloaded files.
                print("Extraction Completed: \(extractedFile)")
            }
            counter = counter + 1
        }    
    }
}

/// Function to read MNIST Dataset.
///
/// - Returns: A Tuple of 4 Tensors for training images, training labels, test images,
/// test labels in that order.
func readMNIST() -> (
    trainImages: Tensor<Float>,
    trainLabels: Tensor<Int32>,
    testImages: Tensor<Float>,
    testLabels: Tensor<Int32>
) {
    downloadAndExtract()
    print("Reading data.")
    let trainImages = try! Data(contentsOf: URL(fileURLWithPath: constants.trainImages)).dropFirst(16).map { Float($0) }
    let trainLabels = try! Data(contentsOf: URL(fileURLWithPath: constants.trainLabels)).dropFirst(8).map { Int32($0) }
    let testImages = try! Data(contentsOf: URL(fileURLWithPath: constants.testImages)).dropFirst(16).map { Float($0) }
    let testLabels = try! Data(contentsOf: URL(fileURLWithPath: constants.testLabels)).dropFirst(8).map { Int32($0) }
    let trainrowCount = Int32(trainLabels.count)
    let testrowCount = Int32(testLabels.count)
    let columnCount = Int32(testImages.count) / testrowCount
    print("Constructing data tensors.")
    return (
        trainImages: Tensor(shape: [trainrowCount, columnCount], scalars: trainImages) / 255,
        trainLabels: Tensor(trainLabels),
        testImages: Tensor(shape: [testrowCount, columnCount], scalars: testImages) / 255,
        testLabels: Tensor(testLabels)
    )
}
