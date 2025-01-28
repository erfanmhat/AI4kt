package io.ai4kt.ai4kt.fibonacci.tensorflow.layers

import io.ai4kt.ai4kt.fibonacci.tensorflow.activations.ReLU
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import kotlin.random.Random
import kotlin.test.*

class CNNLayerTest {
    private val random = Random(42)

    @Test
    fun testForwardPass() {
        // Create a CNN layer with 1 input channel, 2 output channels, and a 3x3 kernel
        val cnnLayer = CNNLayer(
            inputChannels = 1,
            outputChannels = 2,
            kernelSize = 3,
            random = random
        )

        // Create a dummy input (batchSize=1, channels=1, height=4, width=4)
        val input = mk.ndarray(
            listOf(
                listOf(
                    listOf(
                        listOf(1.0, 2.0, 3.0, 4.0),
                        listOf(5.0, 6.0, 7.0, 8.0),
                        listOf(9.0, 10.0, 11.0, 12.0),
                        listOf(13.0, 14.0, 15.0, 16.0)
                    )
                )
            )
        )

        // Perform forward pass
        val output = cnnLayer.forward(input)

        // Verify output shape (batchSize=1, outputChannels=2, height=2, width=2)
        assertEquals(1, output.shape[0], "Batch size mismatch")
        assertEquals(2, output.shape[1], "Output channels mismatch")
        assertEquals(2, output.shape[2], "Output height mismatch")
        assertEquals(2, output.shape[3], "Output width mismatch")
    }

    @Test
    fun testBackwardPass() {
        // Create a CNN layer with 1 input channel, 2 output channels, and a 3x3 kernel
        val cnnLayer = CNNLayer(
            inputChannels = 1,
            outputChannels = 2,
            kernelSize = 3,
            random = random
        )

        // Create a dummy input (batchSize=1, channels=1, height=4, width=4)
        val input = mk.ndarray(
            listOf(
                listOf(
                    listOf(
                        listOf(1.0, 2.0, 3.0, 4.0),
                        listOf(5.0, 6.0, 7.0, 8.0),
                        listOf(9.0, 10.0, 11.0, 12.0),
                        listOf(13.0, 14.0, 15.0, 16.0)
                    )
                )
            )
        )

        // Perform forward pass
        val output = cnnLayer.forward(input)

        // Create dummy gradients (same shape as output)
        val dvalues = mk.ones<Double>(output.shape[0], output.shape[1], output.shape[2], output.shape[3])

        // Perform backward pass
        val dinputs = cnnLayer.backward(dvalues)

        // Verify gradients for weights
        assertNotNull(cnnLayer.dweights, "Gradients for weights should not be null")
        assertEquals(2, cnnLayer.dweights.shape[0], "Gradients for weights: output channels mismatch")
        assertEquals(1, cnnLayer.dweights.shape[1], "Gradients for weights: input channels mismatch")
        assertEquals(3, cnnLayer.dweights.shape[2], "Gradients for weights: kernel height mismatch")
        assertEquals(3, cnnLayer.dweights.shape[3], "Gradients for weights: kernel width mismatch")

        // Verify gradients for biases
        assertNotNull(cnnLayer.dbiases, "Gradients for biases should not be null")
        assertEquals(2, cnnLayer.dbiases.size, "Gradients for biases: output channels mismatch")

        // Verify gradients for inputs
        assertNotNull(dinputs, "Gradients for inputs should not be null")
        assertEquals(1, dinputs.shape[0], "Gradients for inputs: batch size mismatch")
        assertEquals(1, dinputs.shape[1], "Gradients for inputs: input channels mismatch")
        assertEquals(4, dinputs.shape[2], "Gradients for inputs: input height mismatch")
        assertEquals(4, dinputs.shape[3], "Gradients for inputs: input width mismatch")
    }

    @Test
    fun testForwardPassWithActivation() {
        // Create a CNN layer with ReLU activation
        val cnnLayer = CNNLayer(
            inputChannels = 1,
            outputChannels = 2,
            kernelSize = 3,
            random = random,
            activation = ReLU()
        )

        // Create a dummy input (batchSize=1, channels=1, height=4, width=4)
        val input = mk.ndarray(
            listOf(
                listOf(
                    listOf(
                        listOf(1.0, -2.0, 3.0, -4.0),
                        listOf(-5.0, 6.0, -7.0, 8.0),
                        listOf(9.0, -10.0, 11.0, -12.0),
                        listOf(-13.0, 14.0, -15.0, 16.0)
                    )
                )
            )
        )

        // Perform forward pass
        val output = cnnLayer.forward(input)

        // Verify output shape (batchSize=1, outputChannels=2, height=2, width=2)
        assertEquals(1, output.shape[0], "Batch size mismatch")
        assertEquals(2, output.shape[1], "Output channels mismatch")
        assertEquals(2, output.shape[2], "Output height mismatch")
        assertEquals(2, output.shape[3], "Output width mismatch")

        // Verify ReLU activation (all values should be >= 0)
        assertTrue(output.all { it >= 0.0 }, "ReLU activation should produce non-negative values")
    }

    @Test
    fun testBackwardPassWithActivation() {
        // Create a CNN layer with ReLU activation
        val cnnLayer = CNNLayer(
            inputChannels = 1,
            outputChannels = 2,
            kernelSize = 3,
            random = random,
            activation = ReLU()
        )

        // Create a dummy input (batchSize=1, channels=1, height=4, width=4)
        val input = mk.ndarray(
            listOf(
                listOf(
                    listOf(
                        listOf(1.0, -2.0, 3.0, -4.0),
                        listOf(-5.0, 6.0, -7.0, 8.0),
                        listOf(9.0, -10.0, 11.0, -12.0),
                        listOf(-13.0, 14.0, -15.0, 16.0)
                    )
                )
            )
        )

        // Perform forward pass
        val output = cnnLayer.forward(input)

        // Create dummy gradients (same shape as output)
        val dvalues = mk.ones<Double>(output.shape[0], output.shape[1], output.shape[2], output.shape[3])

        // Perform backward pass
        val dinputs = cnnLayer.backward(dvalues)

        // Verify gradients for weights
        assertNotNull(cnnLayer.dweights, "Gradients for weights should not be null")
        assertEquals(2, cnnLayer.dweights.shape[0], "Gradients for weights: output channels mismatch")
        assertEquals(1, cnnLayer.dweights.shape[1], "Gradients for weights: input channels mismatch")
        assertEquals(3, cnnLayer.dweights.shape[2], "Gradients for weights: kernel height mismatch")
        assertEquals(3, cnnLayer.dweights.shape[3], "Gradients for weights: kernel width mismatch")

        // Verify gradients for biases
        assertNotNull(cnnLayer.dbiases, "Gradients for biases should not be null")
        assertEquals(2, cnnLayer.dbiases.size, "Gradients for biases: output channels mismatch")

        // Verify gradients for inputs
        assertNotNull(dinputs, "Gradients for inputs should not be null")
        assertEquals(1, dinputs.shape[0], "Gradients for inputs: batch size mismatch")
        assertEquals(1, dinputs.shape[1], "Gradients for inputs: input channels mismatch")
        assertEquals(4, dinputs.shape[2], "Gradients for inputs: input height mismatch")
        assertEquals(4, dinputs.shape[3], "Gradients for inputs: input width mismatch")
    }
}