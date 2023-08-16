//
//  ContentView.swift
//  Golf Shot Tracer
//
//  Created by Maxence Mottard on 20/07/2023.
//

import SwiftUI
import Vision
import PhotosUI
import AVKit

struct ContentView: View {
    @State private var pickedItem: PhotosPickerItem?
    @State private var movie: TransferableVideo?

    @State private var player: AVPlayer?

    @State var confidenceThreshold: Double = 0.15
    @State var iouThreshold: Double = 0.1

    @State var images: [(CMTime, UIImage)] = []
    @State var observations: [(CMTime, [VNDetectedObjectObservation])] = []

    var body: some View {
        ScrollView {
            VStack {
                PhotosPicker(
                    selection: $pickedItem,
                    matching: .any(of: [.videos])
                ) {
                    Text("Choose vidéo")
                }

                if let player {
                    VideoPlayer(player: player)
                        .frame(width: 300, height: 500)
                }

                ScrollView(.horizontal) {
                    LazyHStack {
                        ForEach(images, id: \.1.self) { (time, image) in
                            getImage(time: time, image: image)
                                .frame(width: 200, height: 350)
                        }
                    }
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
        }
        .onAppear {
            authorisation()
        }
        .onChange(of: pickedItem) { newValue in
            Task {
                movie = try? await newValue?.loadTransferable(type: TransferableVideo.self)

                if let movie {
                    player = AVPlayer(url: movie.url)
                    player?.play()

                    Task {
                        await processingVideo(videoURL: movie.url)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func getImage(time: CMTime, image: UIImage) -> some View {
        if let observations = observations.first(where: { $0.0 == time })?.1 {
            Image(uiImage: image)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .overlay {
                    VNResultShape(observations: observations)
                        .stroke()
                        .fill(.red)
                }
        } else {
            Image(uiImage: image)
                .resizable()
                .aspectRatio(contentMode: .fit)
        }
    }

    private func authorisation() {
        PHPhotoLibrary.requestAuthorization{ (newStatus) in
            print("status is \(newStatus)")
        }
    }

    private func processingVideo(videoURL url: URL) async {
        let videoAsset = AVAsset(url: url)

        guard let numberOfFrames = await videoAsset.numberOfFrames,
              let nominalFrameRate = await videoAsset.nominalFrameRate else {
            return
        }

        let timesArray = (0..<numberOfFrames)
            .map { CMTime(value: Int64($0), timescale: CMTimeScale(nominalFrameRate)) }
            .map { NSValue(time: $0) }

        let generator = AVAssetImageGenerator(asset: videoAsset)
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero

        let images = await generator.generateCGImages(forTimes: timesArray)
        print("Frames", numberOfFrames, "|", "Count", images.count)

        self.images = images.map { (time, image) in
            (time, UIImage(cgImage: image))
        }

        self.images.forEach { (time, image) in
            processImage(image: image.ciImage, time: time)
        }
    }

    func processImage(image ciImage: CIImage?, time: CMTime) {
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
            requestHandler(request: request, error: error, time: time)
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

    func requestHandler(request: VNRequest, error: Error?, time: CMTime) {
        guard let results = request.results else { return }

        let vnResults = results.compactMap { $0 as? VNDetectedObjectObservation }
        observations.append((time, vnResults))
    }
}

extension AVAssetImageGenerator {
    func generateCGImages(forTimes requestedTimes: [NSValue]) async -> [(CMTime, CGImage)] {
        await withUnsafeContinuation { [weak self] continuation in
            var cgImages: [(CMTime, CGImage?)] = []

            self?.generateCGImagesAsynchronously(forTimes: requestedTimes) { t1, image, _, _, _ in
                cgImages.append((t1, image))

                if cgImages.count == requestedTimes.count {
                    let images = cgImages.compactMap { $0 as? (CMTime, CGImage) }
                    continuation.resume(returning: images)
                }
            }
        }
    }
}

extension AVAsset {
    var nominalFrameRate: Float? {
        get async {
            let videoTrack = try? await loadTracks(withMediaType: .video).first

            return try? await videoTrack?.load(.nominalFrameRate)
        }
    }

    var numberOfFrames: Int? {
        get async {
            guard let duration = try? await load(.duration),
                  let nominalFrameRate = await nominalFrameRate else {
                return nil
            }

            let durationInSeconds = CMTimeGetSeconds(duration)
            let framesPerSecond = Float64(nominalFrameRate)
            let totalFrames = Int(framesPerSecond * durationInSeconds)

            return totalFrames
        }
    }
}

extension UIImage {
    var ciImage: CIImage? {
        CIImage(image: self)
    }
}

struct TransferableVideo: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { movie in
            SentTransferredFile(movie.url)
        } importing: { received in
            let copy = URL.documentsDirectory.appending(path: "movie.mp4")

            if FileManager.default.fileExists(atPath: copy.path()) {
                try FileManager.default.removeItem(at: copy)
            }

            try FileManager.default.copyItem(at: received.file, to: copy)
            return Self.init(url: copy)
        }
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
