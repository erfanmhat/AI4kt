import kotlinx.io.files.SystemFileSystem
import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.IOException

/**
 * Reads an IDX1 file and returns the labels as a List<Int>.
 *
 * @param filePath The path to the IDX1 file.
 * @return A list of labels (integers).
 * @throws IOException If an I/O error occurs.
 */
fun readIDX1Labels(filePath: String): List<Int> {
    val path = Path(filePath)
    if (!SystemFileSystem.exists(path)) {
        throw IOException("File not found: $filePath")
    }

    return SystemFileSystem.source(path).buffered().use { source ->
        // Read the magic number (first 4 bytes)
        val magicNumber = source.readInt()
        if (magicNumber != 2049) {
            throw IOException("Invalid IDX1 file: Magic number mismatch.")
        }

        // Read the number of labels (next 4 bytes)
        val numLabels = source.readInt()

        // Read the labels (each label is 1 byte)
        List(numLabels) { source.readByte().toInt() and 0xFF } // Convert to unsigned byte
    }
}

/**
 * Reads an IDX3 file and returns the images as a List of 2D arrays (List<Array<ByteArray>>).
 *
 * @param filePath The path to the IDX3 file.
 * @return A list of images, where each image is represented as a 2D array of bytes.
 * @throws IOException If an I/O error occurs.
 */
fun readIDX3Images(filePath: String): List<List<List<Double>>> {
    val path = Path(filePath)
    if (!SystemFileSystem.exists(path)) {
        throw IOException("File not found: $filePath")
    }

    return SystemFileSystem.source(path).buffered().use { source ->
        // Read the magic number (first 4 bytes)
        val magicNumber = source.readInt()
        if (magicNumber != 2051) {
            throw IOException("Invalid IDX3 file: Magic number mismatch.")
        }

        // Read the number of images (next 4 bytes)
        val numImages = source.readInt()

        // Read the number of rows (next 4 bytes)
        val numRows = source.readInt()

        // Read the number of columns (next 4 bytes)
        val numCols = source.readInt()

        // Read the image data
        List(numImages) {
            List(numRows) { List(numCols) { source.readByte().toDouble() } }
        }
    }
}