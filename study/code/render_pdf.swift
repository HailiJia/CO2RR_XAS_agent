import AppKit
import PDFKit
import Foundation
let source = URL(fileURLWithPath: CommandLine.arguments[1])
let output = URL(fileURLWithPath: CommandLine.arguments[2])
try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
guard let document = PDFDocument(url: source) else { fatalError("Cannot read PDF") }
for index in 0..<document.pageCount {
    guard let page = document.page(at: index) else { continue }
    let image = page.thumbnail(of: NSSize(width: 1400, height: 1800), for: .mediaBox)
    guard let tiff = image.tiffRepresentation, let bitmap = NSBitmapImageRep(data: tiff), let png = bitmap.representation(using: .png, properties: [:]) else { fatalError("Cannot render page \(index+1)") }
    try png.write(to: output.appendingPathComponent(String(format: "page_%02d.png", index+1)))
}
print("pages=\(document.pageCount)")
