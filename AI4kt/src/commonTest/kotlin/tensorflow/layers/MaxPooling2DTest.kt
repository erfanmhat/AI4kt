package tensorflow.layers

import concat
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import tensorflow.contentEquals
import kotlin.test.*

class MaxPooling2DTest {
    @Test
    fun testForwardPass() {
        // Input shape: (batchSize = 1, height = 4, width = 4, channels = 1)
        val input = mk.ndarray(
            mk[
                mk[1.0, 2.0, 3.0, 4.0],
                mk[5.0, 6.0, 7.0, 8.0],
                mk[9.0, 10.0, 11.0, 12.0],
                mk[13.0, 14.0, 15.0, 16.0]
            ]
        ).reshape(1, 4, 4, 1)

        // Create a MaxPooling2D layer
        val maxPoolingLayer = MaxPooling2D(
            poolSize = Pair(2, 2),
            strides = Pair(2, 2),
            padding = "valid",
            inputShape = intArrayOf(1, 4, 4, 1)
        )

        // Perform forward pass
        val output = maxPoolingLayer.forward(input)

        // Expected output after max pooling
        val expectedOutput = mk.ndarray(
            mk[
                mk[6.0, 8.0],
                mk[14.0, 16.0]
            ]
        ).reshape(1, 2, 2, 1)

        // Assert that the output matches the expected output
        assertTrue(output.contentEquals(expectedOutput), "Forward pass output is incorrect.")
    }

    @Test
    fun testBackwardPass() {
        // Input shape: (batchSize = 1, height = 4, width = 4, channels = 1)
        val input = mk.ndarray(
            mk[
                mk[1.0, 2.0, 3.0, 4.0],
                mk[5.0, 6.0, 7.0, 8.0],
                mk[9.0, 10.0, 11.0, 12.0],
                mk[13.0, 14.0, 15.0, 16.0]
            ]
        ).reshape(1, 4, 4, 1)

        // Create a MaxPooling2D layer
        val maxPoolingLayer = MaxPooling2D(
            poolSize = Pair(2, 2),
            strides = Pair(2, 2),
            padding = "valid",
            inputShape = intArrayOf(1, 4, 4, 1)
        )

        // Perform forward pass to cache the input
        maxPoolingLayer.forward(input)

        // Gradients from the next layer (dvalues)
        val dvalues = mk.ndarray(
            mk[
                mk[1.0, 2.0],
                mk[3.0, 4.0]
            ]
        ).reshape(1, 2, 2, 1)

        // Perform backward pass
        val gradients = maxPoolingLayer.backward(dvalues)

        // Expected gradients after backward pass
        val expectedGradients = mk.ndarray(
            mk[
                mk[0.0, 0.0, 0.0, 0.0],
                mk[0.0, 1.0, 0.0, 2.0],
                mk[0.0, 0.0, 0.0, 0.0],
                mk[0.0, 3.0, 0.0, 4.0]
            ]
        ).reshape(1, 4, 4, 1)

        // Assert that the gradients match the expected gradients
        assertTrue(
            gradients.contentEquals(expectedGradients as NDArray<Double, *>),
            "Backward pass gradients are incorrect."
        )
    }

    @Test
    fun testForwardPassWithSamePadding() {
        // Input shape: (batchSize = 1, height = 4, width = 4, channels = 1)
        val input = mk.ndarray(
            mk[
                mk[1.0, 2.0, 3.0, 4.0],
                mk[5.0, 6.0, 7.0, 8.0],
                mk[9.0, 10.0, 11.0, 12.0],
                mk[13.0, 14.0, 15.0, 16.0]
            ]
        ).reshape(1, 4, 4, 1)

        // Create a MaxPooling2D layer with "same" padding
        val maxPoolingLayer = MaxPooling2D(
            poolSize = Pair(2, 2),
            strides = Pair(2, 2),
            padding = "same",
            inputShape = intArrayOf(1, 4, 4, 1)
        )

        // Perform forward pass
        val output = maxPoolingLayer.forward(input)

        // Expected output shape with "same" padding
        val expectedOutputShape = intArrayOf(1, 2, 2, 1)

        // Assert that the output shape matches the expected shape
        assertTrue(
            output.shape.contentEquals(expectedOutputShape),
            "Output shape with 'same' padding is incorrect."
        )
    }

    @Test
    fun testInvalidPaddingMode() {
        // Test invalid padding mode
        val exception = assertFailsWith<IllegalArgumentException> {
            MaxPooling2D(
                poolSize = Pair(2, 2),
                strides = Pair(2, 2),
                padding = "invalid_padding",
                inputShape = intArrayOf(1, 4, 4, 1)
            )
        }

        // Assert the exception message
        assertEquals(
            "Invalid padding mode: invalid_padding",
            exception.message,
            "Exception message for invalid padding mode is incorrect."
        )
    }

    @Test
    fun testForwardPassWithLargerInputAndStrides() {
        // Input shape: (batchSize = 1, height = 6, width = 6, channels = 1)
        val input = mk.ndarray(
            mk[
                mk[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                mk[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                mk[13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
                mk[19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
                mk[25.0, 26.0, 27.0, 28.0, 29.0, 30.0],
                mk[31.0, 32.0, 33.0, 34.0, 35.0, 36.0]
            ]
        ).reshape(1, 6, 6, 1)

        // Create a MaxPooling2D layer
        val maxPoolingLayer = MaxPooling2D(
            poolSize = Pair(3, 3),
            strides = Pair(3, 3),
            padding = "valid",
            inputShape = intArrayOf(1, 6, 6, 1)
        )

        // Perform forward pass
        val output = maxPoolingLayer.forward(input)

        // Expected output after max pooling
        val expectedOutput = mk.ndarray(
            mk[
                mk[15.0, 18.0],
                mk[33.0, 36.0]
            ]
        ).reshape(1, 2, 2, 1)

        // Assert that the output matches the expected output
        assertTrue(
            output.contentEquals(expectedOutput),
            "Forward pass output with larger input and strides is incorrect."
        )
    }

    @Test
    fun testBackwardPassWithNonMaxValues() {
        // Input shape: (batchSize = 1, height = 4, width = 4, channels = 1)
        val input = mk.ndarray(
            mk[
                mk[1.0, 2.0, 3.0, 4.0],
                mk[5.0, 6.0, 7.0, 8.0],
                mk[9.0, 10.0, 11.0, 12.0],
                mk[13.0, 14.0, 15.0, 16.0]
            ]
        ).reshape(1, 4, 4, 1)

        // Create a MaxPooling2D layer
        val maxPoolingLayer = MaxPooling2D(
            poolSize = Pair(2, 2),
            strides = Pair(2, 2),
            padding = "valid",
            inputShape = intArrayOf(1, 4, 4, 1)
        )

        // Perform forward pass to cache the input
        maxPoolingLayer.forward(input)

        // Gradients from the next layer (dvalues)
        val dvalues = mk.ndarray(
            mk[
                mk[0.5, 1.5],
                mk[2.5, 3.5]
            ]
        ).reshape(1, 2, 2, 1)

        // Perform backward pass
        val gradients = maxPoolingLayer.backward(dvalues)

        // Expected gradients after backward pass
        val expectedGradients = mk.ndarray(
            mk[
                mk[0.0, 0.0, 0.0, 0.0],
                mk[0.0, 0.5, 0.0, 1.5],
                mk[0.0, 0.0, 0.0, 0.0],
                mk[0.0, 2.5, 0.0, 3.5]
            ]
        ).reshape(1, 4, 4, 1)

        // Assert that the gradients match the expected gradients
        assertTrue(
            gradients.contentEquals(expectedGradients as NDArray<Double, *>),
            "Backward pass gradients with non-max values are incorrect."
        )
    }

    @Test
    fun testForwardPassWithMultipleChannels() {
        // Input shape: (batchSize = 1, height = 4, width = 4, channels = 2)
        val input = concat(
            mk.ndarray(
                mk[
                    mk[1.0, 2.0, 3.0, 4.0],
                    mk[5.0, 6.0, 7.0, 8.0],
                    mk[9.0, 10.0, 11.0, 12.0],
                    mk[13.0, 14.0, 15.0, 16.0]
                ]
            ).reshape(1, 4, 4, 1),
            mk.ndarray(
                mk[
                    mk[16.0, 15.0, 14.0, 13.0],
                    mk[12.0, 11.0, 10.0, 9.0],
                    mk[8.0, 7.0, 6.0, 5.0],
                    mk[4.0, 3.0, 2.0, 1.0]
                ]
            ).reshape(1, 4, 4, 1),
            axis = 3
        )

        // Create a MaxPooling2D layer
        val maxPoolingLayer = MaxPooling2D(
            poolSize = Pair(2, 2),
            strides = Pair(2, 2),
            padding = "valid",
            inputShape = intArrayOf(1, 4, 4, 2)
        )

        // Perform forward pass
        val output = maxPoolingLayer.forward(input)

        // Expected output after max pooling
        val expectedOutput = concat(
            mk.ndarray(
                mk[
                    mk[6.0, 8.0],
                    mk[14.0, 16.0]
                ]
            ).reshape(1, 2, 2, 1),
            mk.ndarray(
                mk[
                    mk[16.0, 13.0],
                    mk[8.0, 5.0]
                ]
            ).reshape(1, 2, 2, 1),
            axis = 3
        )

        println(output)
        println()
        println(expectedOutput)
        // Assert that the output matches the expected output
        assertTrue(output.contentEquals(expectedOutput), "Forward pass output with multiple channels is incorrect.")
    }

    @Test
    fun testBackwardPassWithMultipleChannels() {
        // Input shape: (batchSize = 1, height = 4, width = 4, channels = 2)
        val input = concat(
            mk.ndarray(
                mk[
                    mk[1.0, 2.0, 3.0, 4.0],
                    mk[5.0, 6.0, 7.0, 8.0],
                    mk[9.0, 10.0, 11.0, 12.0],
                    mk[13.0, 14.0, 15.0, 16.0]
                ]
            ).reshape(1, 4, 4, 1),
            mk.ndarray(
                mk[
                    mk[16.0, 15.0, 14.0, 13.0],
                    mk[12.0, 11.0, 10.0, 9.0],
                    mk[8.0, 7.0, 6.0, 5.0],
                    mk[4.0, 3.0, 2.0, 1.0]
                ]
            ).reshape(1, 4, 4, 1),
            axis = 3
        )

        // Create a MaxPooling2D layer
        val maxPoolingLayer = MaxPooling2D(
            poolSize = Pair(2, 2),
            strides = Pair(2, 2),
            padding = "valid",
            inputShape = intArrayOf(1, 4, 4, 2)
        )

        // Perform forward pass to cache the input
        maxPoolingLayer.forward(input)

        // Gradients from the next layer (dvalues)
        val dvalues = concat(
            mk.ndarray(
                mk[
                    mk[1.0, 2.0],
                    mk[3.0, 4.0]
                ]
            ).reshape(1, 2, 2, 1),
            mk.ndarray(
                mk[
                    mk[5.0, 6.0],
                    mk[7.0, 8.0]
                ]
            ).reshape(1, 2, 2, 1),
            axis = 3
        )

        // Perform backward pass
        val gradients = maxPoolingLayer.backward(dvalues)

        // Expected gradients after backward pass
        val expectedGradients = concat(
            mk.ndarray(
                mk[
                    mk[0.0, 0.0, 0.0, 0.0],
                    mk[0.0, 1.0, 0.0, 2.0],
                    mk[0.0, 0.0, 0.0, 0.0],
                    mk[0.0, 3.0, 0.0, 4.0]
                ]
            ).reshape(1, 4, 4, 1),
            mk.ndarray(
                mk[
                    mk[0.0, 0.0, 0.0, 0.0],
                    mk[0.0, 5.0, 0.0, 6.0],
                    mk[0.0, 0.0, 0.0, 0.0],
                    mk[0.0, 7.0, 0.0, 8.0]
                ]
            ).reshape(1, 4, 4, 1),
            axis = 3
        )

        // Assert that the gradients match the expected gradients
        assertTrue(
            gradients.contentEquals(expectedGradients as NDArray<Double, *>),
            "Backward pass gradients with multiple channels are incorrect."
        )
    }

    @Test
    fun testForwardPassWithNonSquarePoolSize() {
        // Input shape: (batchSize = 1, height = 4, width = 4, channels = 1)
        val input = mk.ndarray(
            mk[
                mk[1.0, 2.0, 3.0, 4.0],
                mk[5.0, 6.0, 7.0, 8.0],
                mk[9.0, 10.0, 11.0, 12.0],
                mk[13.0, 14.0, 15.0, 16.0]
            ]
        ).reshape(1, 4, 4, 1)

        // Create a MaxPooling2D layer with non-square pool size
        val maxPoolingLayer = MaxPooling2D(
            poolSize = Pair(2, 3),
            strides = Pair(2, 3),
            padding = "valid",
            inputShape = intArrayOf(1, 4, 4, 1)
        )

        // Perform forward pass
        val output = maxPoolingLayer.forward(input)

        // Expected output after max pooling
        val expectedOutput = mk.ndarray(
            mk[
                mk[7.0],
                mk[15.0]
            ]
        ).reshape(1, 2, 1, 1)

        // Assert that the output matches the expected output
        assertTrue(output.contentEquals(expectedOutput), "Forward pass output with non-square pool size is incorrect.")
    }
}