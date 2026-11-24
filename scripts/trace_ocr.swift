import Foundation
import Vision
import AppKit

let args = CommandLine.arguments
guard args.count >= 3 else { print("usage: ocr.swift image.json out.json"); exit(1) }
let url = URL(fileURLWithPath: args[1])
guard let img = NSImage(contentsOf: url), let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else { print("cannot load"); exit(1) }
let req = VNRecognizeTextRequest()
req.recognitionLevel = .accurate
req.usesLanguageCorrection = false
req.recognitionLanguages = ["fr-FR", "en-US"]
let handler = VNImageRequestHandler(cgImage: cg, options: [:])
try handler.perform([req])
var out: [[String: Any]] = []
for obs in (req.results ?? []) {
    guard let c = obs.topCandidates(1).first else { continue }
    let b = obs.boundingBox
    out.append(["t": c.string, "x": b.origin.x, "y": b.origin.y, "w": b.width, "h": b.height, "c": c.confidence])
}
let data = try JSONSerialization.data(withJSONObject: out)
try data.write(to: URL(fileURLWithPath: args[2]))
print("observations:", out.count)
