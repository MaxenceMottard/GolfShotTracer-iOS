//
//  ContentView.swift
//  Golf Shot Tracer
//
//  Created by Maxence Mottard on 20/07/2023.
//

import SwiftUI
import Vision
import PhotosUI

struct ContentView: View {
    static let imageName = "Le-swing-de-golf-de-Rory-Mcillroy-en-slow-motion"

    @State var observations: [VNDetectedObjectObservation] = []
    @State private var pickedImage: PhotosPickerItem?
    @State var image = Image(ContentView.imageName)
    @State var ciImage = UIImage(named: Self.imageName)?.ciImage

    @State var confidenceThreshold: Double = 0.15
    @State var iouThreshold: Double = 0.1

    var body: some View {
        VStack {
            image
                .resizable()
                .aspectRatio(contentMode: .fit)
                .overlay {
                    VNResultShape(observations: observations)
                        .stroke()
                        .fill(.red)
                }
                .frame(height: 500)

            PhotosPicker(
                selection: $pickedImage,
                matching: .any(of: [.images, .livePhotos, .screenshots])
            ) {
                Text("Choose photo")
            }

            HStack {
                Text("Confidence threshold: \(String(format: "%.2f", confidenceThreshold))")

                Button(action: { confidenceThreshold -= 0.05 }) {
                    Image(systemName: "minus").padding()
                }

                Slider(value: $confidenceThreshold, in: 0 ... 1, step: 0.05)

                Button(action: { confidenceThreshold += 0.05 }) {
                    Image(systemName: "plus").padding()
                }
            }
            .hidden()

            HStack {
                Text("Overlap threshold: \(String(format: "%.2f", iouThreshold))")

                Button(action: { iouThreshold -= 0.05 }) {
                    Image(systemName: "minus").padding()
                }

                Slider(value: $iouThreshold, in: 0 ... 1, step: 0.05)

                Button(action: { iouThreshold += 0.05 }) {
                    Image(systemName: "plus").padding()
                }
            }
            .hidden()
        }
        .padding()
        .onAppear {
            authoriation()
            processImage(image: UIImage(named: Self.imageName)?.ciImage)
        }
        .onChange(of: ciImage) { processImage(image: $0) }
        .onChange(of: iouThreshold) { _ in processImage(image: ciImage) }
        .onChange(of: confidenceThreshold) { _ in processImage(image: ciImage) }
        .onChange(of: pickedImage) { _ in
            Task { await getImage()}
        }
    }

    func authoriation() {
        PHPhotoLibrary.requestAuthorization{ (newStatus) in
            print("status is \(newStatus)")
        }
    }

    func processImage(image ciImage: CIImage?) {
        let configuration = MLModelConfiguration()

        guard let ciImage,
              let golfTracking = try? GolfBallTracking(configuration: configuration),
              let model = try? VNCoreMLModel(for: golfTracking.model) else {
            return
        }

        model.featureProvider = Inputs(
            iouThreshold: iouThreshold,
            confidenceThreshold: confidenceThreshold
        )

        let request = VNCoreMLRequest(model: model) { (request, error) in
            requestHandler(request: request, error: error)
        }

        let requestOptions: [VNImageOption: Any] = [
            VNImageOption(rawValue: "confidenceThreshold"): confidenceThreshold,
            VNImageOption(rawValue: "iouThreshold"): iouThreshold
        ]

        DispatchQueue.global(qos: .userInitiated).async {
            let handler = VNImageRequestHandler(
                ciImage: ciImage,
                orientation: .up,
                options: requestOptions
            )

            try? handler.perform([request])
        }
    }

    func getImage() async {
        guard let pickedImage else { return }

        if let image = try? await pickedImage.loadTransferable(type: Image.self),
           let imageData = try? await pickedImage.loadTransferable(type: Data.self) {
            self.image = image
            self.ciImage = UIImage(data: imageData)?.ciImage
        }
    }

    func requestHandler(request: VNRequest, error: Error?) {
        guard let results = request.results else { return }

        observations = results.compactMap { $0 as? VNDetectedObjectObservation }
    }
}

class Inputs: MLFeatureProvider {
    let iouThreshold: Double
    let confidenceThreshold: Double

    init(iouThreshold: Double, confidenceThreshold: Double) {
        self.iouThreshold = iouThreshold
        self.confidenceThreshold = confidenceThreshold
    }

    private var values: [String: MLFeatureValue] {
        [
            "iouThreshold": MLFeatureValue(double: iouThreshold),
            "confidenceThreshold": MLFeatureValue(double: confidenceThreshold)
        ]
    }

    var featureNames: Set<String> {
        return Set(values.keys)
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        return values[featureName]
    }
}

extension UIImage {
    var ciImage: CIImage? {
        CIImage(image: self)
    }
}

struct VNResultShape: Shape {
    let observations: [VNDetectedObjectObservation]

    func path(in rect: CGRect) -> Path {
        var path = Path()

        observations.forEach { observationRect in
            let rectToDraw = VNImageRectForNormalizedRect(
                observationRect.boundingBox, Int(rect.width), Int(rect.height)
            )

            path.addRect(.init(
                x: rectToDraw.origin.x,
                y: rect.height - rectToDraw.origin.y - rectToDraw.height,
                width: rectToDraw.width,
                height: rectToDraw.height)
            )
        }

        return path
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
