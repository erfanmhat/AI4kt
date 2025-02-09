import kotlinx.io.files.SystemFileSystem
import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.IOException
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set

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

fun zeros(shape: IntArray): NDArray<Double, *> {
    return when (shape.size) {
        1 -> mk.zeros<Double>(shape[0])
        2 -> mk.zeros<Double>(shape[0], shape[1])
        3 -> mk.zeros<Double>(shape[0], shape[1], shape[2])
        4 -> mk.zeros<Double>(shape[0], shape[1], shape[2], shape[3])
        else -> throw IOException("Invalid shape $shape")
    }
}

/**
 * Concatenates two NDArray objects along the specified axis.
 *
 * @param a The first NDArray.
 * @param b The second NDArray.
 * @param axis The axis along which to concatenate.
 * @return The concatenated NDArray.
 * @throws IllegalArgumentException If the shapes of the arrays are incompatible for concatenation.
 */
fun concat(a: NDArray<Double, *>, b: NDArray<Double, *>, axis: Int): NDArray<Double, *> {
    // Check if the arrays have the same number of dimensions
    if (a.shape.size != b.shape.size) {
        throw IllegalArgumentException("Arrays must have the same number of dimensions.")
    }

    // Check if the shapes are compatible for concatenation along the specified axis
    for (i in a.shape.indices) {
        if (i != axis && a.shape[i] != b.shape[i]) {
            throw IllegalArgumentException("Shapes are incompatible for concatenation along axis $axis.")
        }
    }

    // Create the shape of the resulting array
    val newShape = a.shape.toMutableList()
    newShape[axis] = a.shape[axis] + b.shape[axis]

    // Create a new array to hold the concatenated result
    val result = zeros(newShape.toIntArray())

    // Copy elements from the first array
    val aIndices = Array(a.shape.size) { 0 }
    for (i in 0 until a.size) {
        result[aIndices.copyOf().toIntArray()] = a[aIndices.copyOf().toIntArray()]
        incrementIndices(aIndices, a.shape)
    }

    // Copy elements from the second array
    val bIndices = Array(b.shape.size) { 0 }
    for (i in 0 until b.size) {
        // Calculate the result indices by shifting along the concatenation axis
        val resultIndices = bIndices.copyOf().toIntArray().apply {
            this[axis] += a.shape[axis]
        }
        result[resultIndices] = b[bIndices.copyOf().toIntArray()]
        incrementIndices(bIndices, b.shape)
    }

    return result
}

/**
 * Helper function to increment indices for array traversal.
 *
 * @param indices The current indices.
 * @param shape The shape of the array.
 */
private fun incrementIndices(indices: Array<Int>, shape: IntArray) {
    for (i in indices.indices.reversed()) {
        indices[i]++
        if (indices[i] < shape[i]) {
            break
        }
        indices[i] = 0
    }
}