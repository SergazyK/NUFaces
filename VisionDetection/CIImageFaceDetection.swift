
import Foundation
import Vision
import ARKit
import Accelerate
import CoreImage
public extension CIImage {
    
    var rotate: CIImage {
        get {
            return self.oriented(UIDevice.current.orientation.cameraOrientation())
        }
    }
    
    /// Cropping the image containing the face.
    ///
    /// - Parameter toFace: the face to extract
    /// - Returns: the cropped image
    func cropImage(toFace face: VNFaceObservation) -> CVPixelBuffer {
        let width = face.boundingBox.width * CGFloat(extent.size.width)
        let height = face.boundingBox.height * CGFloat(extent.size.height)
        let x = face.boundingBox.origin.x * CGFloat(extent.size.width)
        let y =  face.boundingBox.origin.y * CGFloat(extent.size.height)
        
        let leftEye = face.landmarks?.leftEye
        let rightEye = face.landmarks?.rightEye

        var leftEyeCenterX: CGFloat = 0.0
        var leftEyeCenterY: CGFloat = 0.0
        let leftPointCount = leftEye!.pointCount
        for i in 0..<leftPointCount {
            let point = leftEye!.normalizedPoints[i]
            leftEyeCenterX += point.x
            leftEyeCenterY += point.y
        }
        leftEyeCenterX = leftEyeCenterX/CGFloat(leftPointCount)
        leftEyeCenterY = leftEyeCenterY/CGFloat(leftPointCount)
        
        
        let rightPointCount = rightEye!.pointCount
        var rightEyeCenterX: CGFloat = 0.0
        var rightEyeCenterY: CGFloat = 0.0
        for i in 0..<rightPointCount {
            let point = rightEye!.normalizedPoints[i]
            rightEyeCenterX += point.x
            rightEyeCenterY += point.y
        }
        rightEyeCenterX = rightEyeCenterX / CGFloat(rightPointCount)
        rightEyeCenterY = rightEyeCenterY / CGFloat(rightPointCount)
        
        rightEyeCenterX = rightEyeCenterX * width + x
        rightEyeCenterY = rightEyeCenterY * height + y
        
        leftEyeCenterX = leftEyeCenterX * width + x
        leftEyeCenterY = leftEyeCenterY * height + y
        
        let diffX = rightEyeCenterX - leftEyeCenterX
        
        let xShift = diffX * 38.5/35.0
        let yShift = diffX * 60.4/35.0
        let cropSize:CGFloat = diffX * 112.0/35.0
        let faceCrop = CGRect.init(x: leftEyeCenterX - xShift, y: leftEyeCenterY - yShift, width: cropSize, height: cropSize)
//        let dY = rightEyeCenterY - leftEyeCenterY
//        let dX = rightEyeCenterX - leftEyeCenterX
//        let angle = atan2(dY, dX)
//        var transform : CGAffineTransform = CGAffineTransform.init(rotationAngle: -angle)
 
        let cropppedFace = self.cropped(to: faceCrop)
        let context = CIContext() // Prepare for create CGImage
        let cgimg = context.createCGImage(cropppedFace, from: cropppedFace.extent)
        let output = UIImage(cgImage: cgimg!).resize(to: CGSize.init(width: 112, height: 112))
        return output.pixelBuffer()!
    }
}

private extension UIDeviceOrientation {
    func cameraOrientation() -> CGImagePropertyOrientation {
        switch self {
        case .landscapeLeft: return .down
        case .landscapeRight: return .up
        case .portraitUpsideDown: return .left
        default: return .right
        }
    }
}



public func resizePixelBuffer(_ srcPixelBuffer: CVPixelBuffer,
                              cropX: Int,
                              cropY: Int,
                              cropWidth: Int,
                              cropHeight: Int,
                              scaleWidth: Int,
                              scaleHeight: Int) -> CVPixelBuffer? {
    
    CVPixelBufferLockBaseAddress(srcPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
    guard let srcData = CVPixelBufferGetBaseAddress(srcPixelBuffer) else {
        print("Error: could not get pixel buffer base address")
        return nil
    }
    let srcBytesPerRow = CVPixelBufferGetBytesPerRow(srcPixelBuffer)
    let offset = cropY*srcBytesPerRow + cropX*4
    var srcBuffer = vImage_Buffer(data: srcData.advanced(by: offset),
                                  height: vImagePixelCount(cropHeight),
                                  width: vImagePixelCount(cropWidth),
                                  rowBytes: srcBytesPerRow)
    
    let destBytesPerRow = scaleWidth*4
    guard let destData = malloc(scaleHeight*destBytesPerRow) else {
        print("Error: out of memory")
        return nil
    }
    var destBuffer = vImage_Buffer(data: destData,
                                   height: vImagePixelCount(scaleHeight),
                                   width: vImagePixelCount(scaleWidth),
                                   rowBytes: destBytesPerRow)
    
    let error = vImageScale_ARGB8888(&srcBuffer, &destBuffer, nil, vImage_Flags(0))
    CVPixelBufferUnlockBaseAddress(srcPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
    if error != kvImageNoError {
        print("Error:", error)
        free(destData)
        return nil
    }
    
    let releaseCallback: CVPixelBufferReleaseBytesCallback = { _, ptr in
        if let ptr = ptr {
            free(UnsafeMutableRawPointer(mutating: ptr))
        }
    }
    
    let pixelFormat = CVPixelBufferGetPixelFormatType(srcPixelBuffer)
    var dstPixelBuffer: CVPixelBuffer?
    let status = CVPixelBufferCreateWithBytes(nil, scaleWidth, scaleHeight,
                                              pixelFormat, destData,
                                              destBytesPerRow, releaseCallback,
                                              nil, nil, &dstPixelBuffer)
    if status != kCVReturnSuccess {
        print("Error: could not create new pixel buffer")
        free(destData)
        return nil
    }
    return dstPixelBuffer
}


extension UIImage {
    
    func resize(to newSize: CGSize) -> UIImage {
        UIGraphicsBeginImageContextWithOptions(CGSize(width: newSize.width, height: newSize.height), true, 1.0)
        self.draw(in: CGRect(x: 0, y: 0, width: newSize.width, height: newSize.height))
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()
        
        return resizedImage
    }
    
    func cropToSquare() -> UIImage? {
        guard let cgImage = self.cgImage else {
            return nil
        }
        var imageHeight = self.size.height
        var imageWidth = self.size.width
        
        if imageHeight > imageWidth {
            imageHeight = imageWidth
        }
        else {
            imageWidth = imageHeight
        }
        
        let size = CGSize(width: imageWidth, height: imageHeight)
        
        let x = ((CGFloat(cgImage.width) - size.width) / 2).rounded()
        let y = ((CGFloat(cgImage.height) - size.height) / 2).rounded()
        
        let cropRect = CGRect(x: x, y: y, width: size.height, height: size.width)
        if let croppedCgImage = cgImage.cropping(to: cropRect) {
            return UIImage(cgImage: croppedCgImage, scale: 0, orientation: self.imageOrientation)
        }
        
        return nil
    }
    
    func pixelBuffer() -> CVPixelBuffer? {
        let width = self.size.width
        let height = self.size.height
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                         Int(width),
                                         Int(height),
                                         kCVPixelFormatType_32ARGB,
                                         attrs,
                                         &pixelBuffer)
        
        guard let resultPixelBuffer = pixelBuffer, status == kCVReturnSuccess else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(resultPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(resultPixelBuffer)
        
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(data: pixelData,
                                      width: Int(width),
                                      height: Int(height),
                                      bitsPerComponent: 8,
                                      bytesPerRow: CVPixelBufferGetBytesPerRow(resultPixelBuffer),
                                      space: rgbColorSpace,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue) else {
                                        return nil
        }
        
        context.translateBy(x: 0, y: height)
        context.scaleBy(x: 1.0, y: -1.0)
        
        UIGraphicsPushContext(context)
        self.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
        UIGraphicsPopContext()
        CVPixelBufferUnlockBaseAddress(resultPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        
        return resultPixelBuffer
    }
}

