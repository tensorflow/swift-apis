import Python
import TensorFlow

let np = Python.import("numpy")
let path = Python.import("os.path")
let urllib = Python.import("urllib.request")
let gzip = Python.import("gzip")
let sh = Python.import("shutil")

let URL = "http://yann.lecun.com/exdb/mnist/"
let trainImagesFile = "train-images-idx3-ubyte.gz"
let trainLabelsFile = "train-labels-idx1-ubyte.gz"
let testImagesFile = "t10k-images-idx3-ubyte.gz"
let testLabelsFile = "t10k-labels-idx1-ubyte.gz"
let trainImages = "train-images-idx3-ubyte"
let trainLabels = "train-labels-idx1-ubyte"
let testImages = "t10k-images-idx3-ubyte"
let testLabels = "t10k-label-idx1-ubyte"
let imageSize = 28
let trainCount = 60000
let testCount = 10000

// Function to download and extract from online
func download_and_extract(){
    
    let helperFiles = [trainImagesFile, trainLabelsFile, testImagesFile, testLabelsFile]
    let extractedFiles = [trainImages, trainLabels, testImages, testLabels]
    var counter = 0
    var f_in: PythonObject
    var f_out: PythonObject
    
    for helperFile in helperFiles {
        if !Bool(path.isfile(helperFile))! {
            print("Downloading \(helperFile)")
            urllib.urlretrieve(URL + helperFile, filename: helperFile)
            print("Downloaded \(helperFile)")
        } else {
            print("Not downloading \(helperFile): already exists")
        }
    }
    
    for extractedFile in extractedFiles {
        
        if !Bool(path.isfile(extractedFile))! {
            print("Extracting \(extractedFile)")
            f_in = gzip.open(helperFiles[counter], "rb")
            f_out = Python.open(extractedFiles[counter], "wb")
            sh.copyfileobj(f_in, f_out)
            print("EXtraction Completed: \(extractedFile)")
        } else {
            print("File \(extractedFile): already exists")
        }
        counter = counter + 1
    }
    
}

// Reads a file into an array of bytes.
func readFile(_ filename: String) -> [UInt8] {
    let d = Python.open(filename, "rb").read()
    return Array(numpy: np.frombuffer(d, dtype: np.uint8))!
}

// Reads MNIST images and labels from specified file paths.
func readMNIST(trainImages: String, trainLabels: String, testImages: String, testLabels: String) -> (trainImages: Tensor<Float>, trainLabels: Tensor<Int32>, testImages: Tensor<Float>, testLabels: Tensor<Int32>) {
    
    print("Reading data.")
    let trainImages = readFile(trainImages).dropFirst(16).map { Float($0) }
    let trainLabels = readFile(trainLabels).dropFirst(8).map { Int32($0) }
    let testImages = readFile(testImages).dropFirst(16).map { Float($0) }
    let testLabels = readFile(testLabels).dropFirst(8).map { Int32($0) }
        
    let trainrowCount = Int32(trainLabels.count)
    let testrowCount = Int32(testLabels.count)
    let columnCount = Int32(testImages.count) / testrowCount
    print(trainImages.count)
    print(testImages.count)
    print(trainrowCount)
    print(columnCount)
    print(testrowCount)
    print("Constructing data tensors.")
    return (
        trainImages: Tensor(shape: [trainrowCount, columnCount], scalars: trainImages) / 255,
        trainLabels: Tensor(trainLabels),
        testImages: Tensor(shape: [testrowCount, columnCount], scalars: testImages) / 255,
        testLabels: Tensor(testLabels)
    )
}

download_and_extract()
let (trainx, trainy, testx, testy) = readMNIST(trainImages: trainImages, trainLabels: trainLabels, testImages: testImages, testLabels: testLabels)