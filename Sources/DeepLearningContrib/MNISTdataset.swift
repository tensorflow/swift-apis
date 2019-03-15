import Foundation
import TensorFlow
import Gzip

/// MNIST Constants
fileprivate enum Constants {
    static let URL = "http://yann.lecun.com/exdb/mnist/"
    static let trainImagesFile = "train-images-idx3-ubyte.gz"
    static let trainLabelsFile = "train-labels-idx1-ubyte.gz"
    static let testImagesFile = "t10k-images-idx3-ubyte.gz"
    static let testLabelsFile = "t10k-labels-idx1-ubyte.gz"
    static let trainImages = "train-images-idx3-ubyte"
    static let trainLabels = "train-labels-idx1-ubyte"
    static let testImages = "t10k-images-idx3-ubyte"
    static let testLabels = "t10k-labels-idx1-ubyte"
}

/// To download and extract files from the URL
fileprivate func downloadAndExtract() {
    let helperFiles = [
        Constants.trainImagesFile,
        Constants.trainLabelsFile,
        Constants.testImagesFile,
        Constants.testLabelsFile
    ]
    let extractedFiles = [
        Constants.trainImages,
        Constants.trainLabels,
        Constants.testImages,
        Constants.testLabels
    ]
    let localDownloadsUrl = FileManager.default.urls(
            for: .downloadsDirectory, in: .userDomainMask
        ).first as URL?
    let sessionConfig = URLSessionConfiguration.default
    let session = URLSession(configuration: sessionConfig)

    for (index, helperFile) in helperFiles.enumerated() {
        let fileURL = URL(string: Constants.URL+helperFile)
        let request = URLRequest(url:fileURL!)
        var destinationFileUrl = localDownloadsUrl!.appendingPathComponent(helperFile)
        var filePath = destinationFileUrl.path
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: filePath) {
            print("Not downloading \(helperFile): already exists")
            var extractedFileUrl = localDownloadsUrl!.appendingPathComponent(extractedFiles[index])
            filePath = extractedFileUrl.path
            if fileManager.fileExists(atPath: filePath){
                print("File \(extractedFiles[index]): already exists")
            } else {
                do{
                    try Data(contentsOf: extractedFileUrl).gunzipped().write(to: destinationFileUrl)
                } catch {
                    print("Error Extracting File: \(extractedFiles[index]) ")
                }
                print(extractedFiles[index])
            }
        } else {
            print("Downloading \(helperFile)")
            let task = session.downloadTask(with: request) { (tempLocalUrl, response, error) in
                if let tempLocalUrl = tempLocalUrl, error == nil {
                    if let statusCode = (response as? HTTPURLResponse)?.statusCode {
                        print("Successfully downloaded. Status code: \(statusCode)")
                    }
                    do {
                        try FileManager.default.copyItem(at: tempLocalUrl, to: destinationFileUrl)
                    } catch (let writeError) {
                        print("Error creating a file \(destinationFileUrl) : \(writeError)")
                    }
                } else {
                    print("Error took place while downloading a file.");
                }
            }
            task.resume()
            print("Downloaded \(helperFile)")
            print("Extracting \(extractedFiles[index])")
            var extractedFileUrl = localDownloadsUrl!.appendingPathComponent(extractedFiles[index])
            filePath = extractedFileUrl.path
            do{
                try Data(contentsOf: extractedFileUrl).gunzipped().write(to: destinationFileUrl)
            } catch {
                print("Error Extracting File: \(extractedFiles[index]) ")
            }
            print("Extraction Completed: \(extractedFiles[index])")
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
        let localDownloadsUrl = FileManager.default.urls(
            for: .downloadsDirectory, in: .userDomainMask).first as URL?
        let trainImages = try! Data(
            contentsOf: URL(fileURLWithPath: localDownloadsUrl!.path + "/" + Constants.trainImages)
        ).dropFirst(16).map { Float($0) }
        let trainLabels = try! Data(
            contentsOf: URL(fileURLWithPath: localDownloadsUrl!.path + "/" + Constants.trainLabels)
        ).dropFirst(8).map { Int32($0) }
        let testImages = try! Data(
            contentsOf: URL(fileURLWithPath: localDownloadsUrl!.path + "/" + Constants.testImages)
        ).dropFirst(16).map { Float($0) }
        let testLabels = try! Data(
            contentsOf: URL(fileURLWithPath: localDownloadsUrl!.path + "/" + Constants.testLabels)
        ).dropFirst(8).map { Int32($0) }
        let trainRowCount = Int32(trainLabels.count)
        let testRowCount = Int32(testLabels.count)
        let columnCount = Int32(testImages.count) / testRowCount
        print("Constructing data tensors.")
        return (
            trainImages: Tensor(shape: [trainRowCount, columnCount], scalars: trainImages) / 255,
            trainLabels: Tensor(trainLabels),
            testImages: Tensor(shape: [testRowCount, columnCount], scalars: testImages) / 255,
            testLabels: Tensor(testLabels)
        )
}
