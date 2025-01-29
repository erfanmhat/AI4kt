package tensorflow.layers

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import tensorflow.activations.Activation
import tensorflow.activations.ReLU
import kotlin.test.*
import kotlin.random.Random

class CNNLayerTest {

    private lateinit var cnnLayer: CNNLayer
    private val random = Random(42)

    private fun createRandomInput(shape: IntArray): NDArray<Double, *> {
        // Create a zero array and fill it with random values
        return when (shape.size) {
            1 -> mk.zeros<Double>(shape[0]).map { random.nextDouble(-1.0, 1.0) }
            2 -> mk.zeros<Double>(shape[0], shape[1]).map { random.nextDouble(-1.0, 1.0) }
            3 -> mk.zeros<Double>(shape[0], shape[1], shape[2]).map { random.nextDouble(-1.0, 1.0) }
            4 -> mk.zeros<Double>(shape[0], shape[1], shape[2], shape[3]).map { random.nextDouble(-1.0, 1.0) }
            else -> throw IllegalArgumentException("Shape $shape is not supported")
        }
    }

    @Test
    fun testWeightInitialization() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = intArrayOf(3, 3)
        cnnLayer = CNNLayer(inputShape, filters, kernelSize, random = random)

        assertEquals(filters, cnnLayer.weights.shape[0])
        assertEquals(inputShape[2], cnnLayer.weights.shape[1])
        assertEquals(kernelSize[0], cnnLayer.weights.shape[2])
        assertEquals(kernelSize[1], cnnLayer.weights.shape[3])
        assertTrue(cnnLayer.weights.all { it in -0.1..0.1 })
    }

    @Test
    fun testBiasInitialization() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = intArrayOf(3, 3)
        cnnLayer = CNNLayer(inputShape, filters, kernelSize, random = random)

        assertEquals(filters, cnnLayer.biases.size)
        assertTrue(cnnLayer.biases.all { it == 0.01 })
    }

    @Test
    fun testForwardPassWithValidPadding() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = intArrayOf(3, 3)
        cnnLayer = CNNLayer(inputShape, filters, kernelSize, padding = "valid", random = random)

        val input = createRandomInput(inputShape)
        val output = cnnLayer.forward(input)

        val expectedHeight = inputShape[1] - kernelSize[0] + 1
        val expectedWidth = inputShape[2] - kernelSize[1] + 1
        assertEquals(intArrayOf(1, expectedHeight, expectedWidth, filters).toList(), output.shape.toList())
    }

    @Test
    fun testForwardPassWithSamePadding() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = intArrayOf(3, 3)
        cnnLayer = CNNLayer(inputShape, filters, kernelSize, padding = "same", random = random)

        val input = createRandomInput(inputShape)
        val output = cnnLayer.forward(input)

        val expectedHeight = inputShape[1]
        val expectedWidth = inputShape[2]
        assertEquals(intArrayOf(1, expectedHeight, expectedWidth, filters).toList(), output.shape.toList())
    }

    @Test
    fun testForwardPassWithDifferentStrides() {
        val inputShape = intArrayOf(1, 5, 5, 3) // Input with 3 channels
        val filters = 2
        val kernelSize = intArrayOf(2, 2)
        val strides = intArrayOf(2, 2)
        cnnLayer = CNNLayer(inputShape, filters, kernelSize, strides = strides, padding = "valid", random = random)

        val input = createRandomInput(inputShape)
        val output = cnnLayer.forward(input)

        val expectedHeight = (inputShape[1] - kernelSize[0]) / strides[0] + 1
        val expectedWidth = (inputShape[2] - kernelSize[1]) / strides[1] + 1
        assertEquals(intArrayOf(1, expectedHeight, expectedWidth, filters).toList(), output.shape.toList())
    }


    @Test
    fun testForwardPassWithActivationFunction() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = intArrayOf(3, 3)
        val activation = ReLU()
        cnnLayer = CNNLayer(inputShape, filters, kernelSize, random = random, activation = activation)

        val input = createRandomInput(inputShape)
        val output = cnnLayer.forward(input)

        assertEquals(intArrayOf(1, 3, 3, filters).toList(), output.shape.toList()) // Assuming valid padding
        assertTrue(output.all { it >= 0 }) // Check if output is non-negative due to ReLU
    }

    @Test
    fun testBackwardPass() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = intArrayOf(3, 3)
        cnnLayer = CNNLayer(inputShape, filters, kernelSize, random = random)

        val input = createRandomInput(inputShape)
        val output = cnnLayer.forward(input)

        val dvalues = createRandomInput(output.shape)
        val dinputs = cnnLayer.backward(dvalues)

        assertEquals(inputShape.toList(), dinputs.shape.toList())
        assertEquals(cnnLayer.dweights.shape.toList(), cnnLayer.weights.shape.toList())
        assertEquals(cnnLayer.dbiases.size, filters)
    }

    @Test
    fun testEdgeCaseWithMinimalInput() {
        val inputShape = intArrayOf(1, 1, 1, 1) // Minimal input
        val filters = 1
        val kernelSize = intArrayOf(1, 1)
        cnnLayer = CNNLayer(inputShape, filters, kernelSize, random = random)

        val input = createRandomInput(inputShape)
        val output = cnnLayer.forward(input)

        assertEquals(intArrayOf(1, 1, 1, filters).toList(), output.shape.toList())
    }
}
