import Python
import TensorFlow

fileprivate let np = Python.import("numpy")
fileprivate let path = Python.import("os.path")
fileprivate let urllib = Python.import("urllib.request")
fileprivate let gzip = Python.import("gzip")
fileprivate let sh = Python.import("shutil")

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
    var fin: PythonObject
    var fout: PythonObject
    for helperFile in helperFiles {
        if !Bool(path.isfile(helperFile))! {
            print("Downloading \(helperFile)")
            urllib.urlretrieve(constants.URL + helperFile, filename: helperFile)
            print("Downloaded \(helperFile)")
        } else {
            print("Not downloading \(helperFile): already exists")
        }
    }
    for extractedFile in extractedFiles {
        if !Bool(path.isfile(extractedFile))! {
            print("Extracting \(extractedFile)")
            fin = gzip.open(helperFiles[counter], "rb")
            fout = Python.open(extractedFiles[counter], "wb")
            sh.copyfileobj(fin, fout)
            print("Extraction Completed: \(extractedFile)")
        } else {
            print("File \(extractedFile): already exists")
        }
        counter = counter + 1
    }

}

/// Reads a file into an array of bytes.
///
/// - Parameters:
///   - filename: A String with file name that is to be converted to array of bytes. 
/// - Returns: An Array of type UInt8.
fileprivate func readFile(_ filename: String) -> [UInt8] {
    let buffer = Python.open(filename, "rb").read()
    return Array(numpy: np.frombuffer(buffer, dtype: np.uint8))!
}

/// Used to read MNIST Dataset.
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
    let trainImages = readFile(constants.trainImages).dropFirst(16).map { Float($0) }
    let trainLabels = readFile(constants.trainLabels).dropFirst(8).map { Int32($0) }
    let testImages = readFile(constants.testImages).dropFirst(16).map { Float($0) }
    let testLabels = readFile(constants.testLabels).dropFirst(8).map { Int32($0) }
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

let (trainX, trainY, testX, testY) = readMNIST()
