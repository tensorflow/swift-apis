/*
Usage:

downloadAndExtract()
let (trainX, trainY, testX, testY) = readMNIST(trainImages: constants["trainImages"]!, 
                                                   trainLabels: constants["trainLabels"]!, 
                                                   testImages: constants["testImages"]!, 
                                                   testLabels: constants["testLabels"]!)
*/

import Python
import TensorFlow

let np = Python.import("numpy")
let path = Python.import("os.path")
let urllib = Python.import("urllib.request")
let gzip = Python.import("gzip")
let sh = Python.import("shutil")

// MNIST constants
let constants: [String: String] = [
    "URL": "http://yann.lecun.com/exdb/mnist/",
    "trainImagesFile": "train-images-idx3-ubyte.gz",
    "trainLabelsFile": "train-labels-idx1-ubyte.gz",
    "testImagesFile": "t10k-images-idx3-ubyte.gz",
    "testLabelsFile": "t10k-labels-idx1-ubyte.gz",
    "trainImages": "train-images-idx3-ubyte",
    "trainLabels": "train-labels-idx1-ubyte",
    "testImages": "t10k-images-idx3-ubyte",
    "testLabels": "t10k-label-idx1-ubyte",
]

// Function to download and extract files from online
func downloadAndExtract(){
    
    let helperFiles = [constants["trainImagesFile"]!, constants["trainLabelsFile"]!,
                       constants["testImagesFile"]!, constants["testLabelsFile"]!]
    let extractedFiles = [constants["trainImages"]!, constants["trainLabels"]!,
                          constants["testImages"]!, constants["testLabels"]!]
    var counter = 0
    var fin: PythonObject
    var fout: PythonObject
    
    for helperFile in helperFiles {
        if !Bool(path.isfile(helperFile))! {
            print("Downloading \(helperFile)")
            urllib.urlretrieve(constants["URL"]! + helperFile, filename: helperFile)
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

// Reads a file into an array of bytes.
func readFile(_ filename: String) -> [UInt8] {
    let buffer = Python.open(filename, "rb").read()
    return Array(numpy: np.frombuffer(buffer, dtype: np.uint8))!
}

// Reads MNIST images and labels from specified file paths.
func readMNIST(trainImages: String, trainLabels: String, 
               testImages: String, testLabels: String) -> 
               (trainImages: Tensor<Float>, trainLabels: Tensor<Int32>,
               testImages: Tensor<Float>, testLabels: Tensor<Int32>)
{
    print("Reading data.")
    let trainImages = readFile(trainImages).dropFirst(16).map { Float($0) }
    let trainLabels = readFile(trainLabels).dropFirst(8).map { Int32($0) }
    let testImages = readFile(testImages).dropFirst(16).map { Float($0) }
    let testLabels = readFile(testLabels).dropFirst(8).map { Int32($0) }
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
