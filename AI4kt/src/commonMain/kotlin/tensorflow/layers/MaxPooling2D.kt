package tensorflow.layers

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import tensorflow.findMaxIndex
import zeros

class MaxPooling2D(
    val poolSize: Pair<Int, Int>, // (height, width)
    val strides: Pair<Int, Int>,  // (height, width)
    val padding: String           // "valid" or "same"
) : Layer {
    private lateinit var inputCache: NDArray<Double, D4>
    private var padTop: Int = 0
    private var padLeft: Int = 0

    override fun forward(inputs: NDArray<Double, *>): NDArray<Double, *> {
        inputCache = inputs.asDNArray().toD4()

        val batchSize = inputCache.shape[0]
        val inputHeight = inputCache.shape[1]
        val inputWidth = inputCache.shape[2]
        val channels = inputCache.shape[3]
        val (poolHeight, poolWidth) = poolSize
        val (strideHeight, strideWidth) = strides

        // Dynamically compute output shape
        val (outputHeight, outputWidth) = when (padding) {
            "valid" -> Pair(
                (inputHeight - poolHeight) / strideHeight + 1,
                (inputWidth - poolWidth) / strideWidth + 1
            )

            "same" -> Pair(
                (inputHeight + strideHeight - 1) / strideHeight,
                (inputWidth + strideWidth - 1) / strideWidth
            )

            else -> throw IllegalArgumentException("Invalid padding mode: $padding")
        }

        // Apply padding if needed
        val paddedInput = when (padding) {
            "same" -> {
                val totalPadHeight = (outputHeight - 1) * strideHeight + poolHeight - inputHeight
                val totalPadWidth = (outputWidth - 1) * strideWidth + poolWidth - inputWidth
                padTop = totalPadHeight / 2
                val padBottom = totalPadHeight - padTop
                padLeft = totalPadWidth / 2
                val padRight = totalPadWidth - padLeft

                val padded = mk.zeros<Double>(
                    batchSize,
                    inputHeight + totalPadHeight,
                    inputWidth + totalPadWidth,
                    channels
                )

                // Copy original input into padded tensor using loops
                for (b in 0 until batchSize) {
                    for (h in 0 until inputHeight) {
                        for (w in 0 until inputWidth) {
                            for (c in 0 until channels) {
                                padded[b, padTop + h, padLeft + w, c] = inputCache[b, h, w, c]
                            }
                        }
                    }
                }
                padded
            }

            else -> inputCache // No padding for "valid"
        }

        val output = mk.zeros<Double>(batchSize, outputHeight, outputWidth, channels)

        for (b in 0 until batchSize) {
            for (c in 0 until channels) {
                for (i in 0 until outputHeight) {
                    for (j in 0 until outputWidth) {
                        val startH = i * strideHeight
                        val endH = startH + poolHeight
                        val startW = j * strideWidth
                        val endW = startW + poolWidth

                        // Extract window from padded input using loops
                        var maxVal = Double.NEGATIVE_INFINITY
                        for (h in startH until endH) {
                            for (w in startW until endW) {
                                val value = paddedInput[b, h, w, c]
                                if (value > maxVal) {
                                    maxVal = value
                                }
                            }
                        }
                        output[b, i, j, c] = maxVal
                    }
                }
            }
        }

        return output
    }

    override fun backward(dvalues: NDArray<Double, *>): NDArray<Double, *> {
        val gradients = zeros(inputCache.shape)
        val batchSize = inputCache.shape[0]
        val inputHeight = inputCache.shape[1]
        val inputWidth = inputCache.shape[2]
        val channels = inputCache.shape[3]
        val (poolHeight, poolWidth) = poolSize
        val (strideHeight, strideWidth) = strides

        // Dynamically compute output shape again
        val (outputHeight, outputWidth) = when (padding) {
            "valid" -> Pair(
                (inputHeight - poolHeight) / strideHeight + 1,
                (inputWidth - poolWidth) / strideWidth + 1
            )

            "same" -> Pair(
                (inputHeight + strideHeight - 1) / strideHeight,
                (inputWidth + strideWidth - 1) / strideWidth
            )

            else -> throw IllegalArgumentException("Invalid padding mode: $padding")
        }

        for (b in 0 until batchSize) {
            for (c in 0 until channels) {
                for (i in 0 until outputHeight) {
                    for (j in 0 until outputWidth) {
                        val startH = i * strideHeight
                        val endH = startH + poolHeight
                        val startW = j * strideWidth
                        val endW = startW + poolWidth

                        // Find the max value and its position in the window
                        var maxVal = Double.NEGATIVE_INFINITY
                        var maxH = startH
                        var maxW = startW
                        for (h in startH until endH) {
                            for (w in startW until endW) {
                                val value = inputCache[b, h, w, c]
                                if (value > maxVal) {
                                    maxVal = value
                                    maxH = h
                                    maxW = w
                                }
                            }
                        }

                        // Accumulate gradient at the max position
                        gradients[b, maxH, maxW, c] += dvalues[b, i, j, c]
                    }
                }
            }
        }

        return if (padding == "same") {
            // Crop gradients to original input shape using loops
            val croppedGradients = mk.zeros<Double>(batchSize, inputHeight, inputWidth, channels)
            for (b in 0 until batchSize) {
                for (h in 0 until inputHeight) {
                    for (w in 0 until inputWidth) {
                        for (c in 0 until channels) {
                            croppedGradients[b, h, w, c] = gradients[b, padTop + h, padLeft + w, c]
                        }
                    }
                }
            }
            croppedGradients
        } else {
            gradients
        }
    }

    // Helper function to cast NDArray to D4
    private fun NDArray<Double, *>.toD4(): NDArray<Double, D4> {
        return this.reshape(
            this.shape[0],
            this.shape[1],
            this.shape[2],
            this.shape[3]
        ) as NDArray<Double, D4>
    }
}