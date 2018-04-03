/*
 Adapted from WeiJay
 */
import UIKit
import Vision
import AVFoundation

class PreviewView: UIView {
 
    public var cameraType = true
    public var maskLayer = [CAShapeLayer]()
    public var textMaskLayer = [CALayer]()
    // MARK: AV capture properties
    var videoPreviewLayer: AVCaptureVideoPreviewLayer {
        return layer as! AVCaptureVideoPreviewLayer
    }
    
    var session: AVCaptureSession? {
        get {
            return videoPreviewLayer.session
        }
        
        set {
            videoPreviewLayer.session = newValue
        }
    }
    
    override class var layerClass: AnyClass {
        return AVCaptureVideoPreviewLayer.self
    }
    
    // Create a new layer drawing the bounding box
    private func createLayer(in rect: CGRect, name: String) -> CAShapeLayer{
        
        let mask = CAShapeLayer()
        mask.frame = rect
        mask.cornerRadius = 10
        mask.opacity = 0.75
        mask.borderColor = UIColor.yellow.cgColor
        mask.borderWidth = 1.0
        
        let textLayer = CATextLayer()
        textLayer.font = UIFont.systemFont(ofSize: 20.0, weight: UIFont.Weight.light) //UIFont(name: "System", size: 20)
        textLayer.fontSize = 20.0
        textLayer.frame = CGRect.init(x: rect.origin.x, y: rect.origin.y + rect.height, width: 300, height: 60)
        textLayer.string = name
        textLayer.isHidden = false
        textLayer.foregroundColor = UIColor.red.cgColor
        
        textMaskLayer.append(textLayer)
        maskLayer.append(mask)
        
        layer.insertSublayer(mask, at: 1)
        layer.insertSublayer(textLayer, at: 1)
        
        return mask
    }
    
    func drawFaceboundingBox(faceBoundixBox : CGRect, name: String) {
        
        let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: -frame.height)

        let translate = CGAffineTransform.identity.scaledBy(x: frame.width, y: frame.height)
        
        // The coordinates are normalized to the dimensions of the processed image, with the origin at the image's lower-left corner.

        var facebounds = faceBoundixBox.applying(translate).applying(transform)
        
        let p: CGFloat = 0.1
        let width = facebounds.width
        let height = facebounds.height
        let x = facebounds.origin.x
        if (cameraType == false){
            facebounds.origin.x = 2 * center.x - facebounds.origin.x - facebounds.width
        }
        let y = facebounds.origin.y
        //CGRectMake(CGFloat x, CGFloat y, CGFloat width, CGFloat height);
        let faceRect = CGRect(x: x, y: y - (height * p), width: width*(1+p), height: height*(1+p))
        
        _ = createLayer(in: facebounds, name: name)
        
    }
    
    
    

    func drawFaceWithLandmarks(face: VNFaceObservation, name: String) {
        
        let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: -frame.height)

        let translate = CGAffineTransform.identity.scaledBy(x: frame.width, y: frame.height)
        
        // The coordinates are normalized to the dimensions of the processed image, with the origin at the image's lower-left corner.
        let facebounds = face.boundingBox.applying(translate).applying(transform)
        
        // Draw the bounding rect
        
        let faceLayer = createLayer(in: facebounds, name: name)
//        faceLayer.isGeometryFlipped = true
        // Draw the landmarks
        drawLandmarks(on: faceLayer, faceLandmarkRegion: (face.landmarks?.nose)!, isClosed:false)
        drawLandmarks(on: faceLayer, faceLandmarkRegion: (face.landmarks?.noseCrest)!, isClosed:false)
        drawLandmarks(on: faceLayer, faceLandmarkRegion: (face.landmarks?.medianLine)!, isClosed:false)
        drawLandmarks(on: faceLayer, faceLandmarkRegion: (face.landmarks?.leftEye)!)
        drawLandmarks(on: faceLayer, faceLandmarkRegion: (face.landmarks?.leftPupil)!)
        drawLandmarks(on: faceLayer, faceLandmarkRegion: (face.landmarks?.leftEyebrow)!, isClosed:false)
        drawLandmarks(on: faceLayer, faceLandmarkRegion: (face.landmarks?.rightEye)!)
        drawLandmarks(on: faceLayer, faceLandmarkRegion: (face.landmarks?.rightPupil)!)
        drawLandmarks(on: faceLayer, faceLandmarkRegion: (face.landmarks?.rightEye)!)
        drawLandmarks(on: faceLayer, faceLandmarkRegion: (face.landmarks?.rightEyebrow)!, isClosed:false)
        drawLandmarks(on: faceLayer, faceLandmarkRegion: (face.landmarks?.innerLips)!)
        drawLandmarks(on: faceLayer, faceLandmarkRegion: (face.landmarks?.outerLips)!)
        drawLandmarks(on: faceLayer, faceLandmarkRegion: (face.landmarks?.faceContour)!, isClosed: false)
    }
    

    
    func drawLandmarks(on targetLayer: CALayer, faceLandmarkRegion: VNFaceLandmarkRegion2D, isClosed: Bool = true) {
        let rect: CGRect = targetLayer.frame
        var points: [CGPoint] = []
        
        for i in 0..<faceLandmarkRegion.pointCount {
            let point = faceLandmarkRegion.normalizedPoints[i]
            points.append(point)
        }
        
        let landmarkLayer = drawPointsOnLayer(rect: rect, landmarkPoints: points, isClosed: isClosed)
        
        // Change scale, coordinate systems, and mirroring
        landmarkLayer.transform = CATransform3DMakeAffineTransform(
            CGAffineTransform.identity
                .scaledBy(x: rect.width, y: -rect.height)
                .translatedBy(x: 0, y: -1)
        )

        targetLayer.insertSublayer(landmarkLayer, at: 1)
    }
    
    func drawPointsOnLayer(rect:CGRect, landmarkPoints: [CGPoint], isClosed: Bool = true) -> CALayer {
        let linePath = UIBezierPath()
        linePath.move(to: landmarkPoints.first!)
        
        for point in landmarkPoints.dropFirst() {
            linePath.addLine(to: point)
        }
        
        if isClosed {
            linePath.addLine(to: landmarkPoints.first!)
        }
        
        let lineLayer = CAShapeLayer()
        lineLayer.path = linePath.cgPath
        lineLayer.fillColor = nil
        lineLayer.opacity = 1.0
        lineLayer.strokeColor = UIColor.green.cgColor
        lineLayer.lineWidth = 0.02
        
        return lineLayer
    }
    
    func removeMask() {
        for mask in maskLayer {
            mask.removeFromSuperlayer()
        }
        for mask in textMaskLayer{
            mask.removeFromSuperlayer()
        }
        maskLayer.removeAll()
        textMaskLayer.removeAll()
    }
    
}
