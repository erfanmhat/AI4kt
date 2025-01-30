package tensorflow.layers

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import tensorflow.activations.ReLU
import kotlin.math.sqrt
import kotlin.test.*
import kotlin.random.Random

class Conv2DTest {

    private lateinit var conv2D: Conv2D
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
        val kernelSize = Pair(3, 3)
        conv2D = Conv2D(inputShape, filters, kernelSize, random = random)

        val scale = sqrt(2.0 / (inputShape[2] * kernelSize.first * kernelSize.second))
        assertEquals(filters, conv2D.weights.shape[0])
        assertEquals(inputShape[2], conv2D.weights.shape[1])
        assertEquals(kernelSize.first, conv2D.weights.shape[2])
        assertEquals(kernelSize.second, conv2D.weights.shape[3])
        assertTrue(conv2D.weights.all { it in -scale..scale })
    }

    @Test
    fun testWeightInitializationWithDifferentInputShape() {
        val inputShape = intArrayOf(1, 7, 7, 4) // New input shape
        val filters = 3
        val kernelSize = Pair(3, 3)
        conv2D = Conv2D(inputShape, filters, kernelSize, random = random)

        val scale = sqrt(2.0 / (inputShape[2] * kernelSize.first * kernelSize.second))
        assertEquals(filters, conv2D.weights.shape[0])
        assertEquals(inputShape[2], conv2D.weights.shape[1])
        assertEquals(kernelSize.first, conv2D.weights.shape[2])
        assertEquals(kernelSize.second, conv2D.weights.shape[3])
        assertTrue(conv2D.weights.all { it in -scale..scale })
    }

    @Test
    fun testWeightInitializationWithDifferentKernelSize() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = Pair(5, 5) // New kernel size
        conv2D = Conv2D(inputShape, filters, kernelSize, random = random)

        val scale = sqrt(2.0 / (inputShape[2] * kernelSize.first * kernelSize.second))
        assertEquals(filters, conv2D.weights.shape[0])
        assertEquals(inputShape[2], conv2D.weights.shape[1])
        assertEquals(kernelSize.first, conv2D.weights.shape[2])
        assertEquals(kernelSize.second, conv2D.weights.shape[3])
        assertTrue(conv2D.weights.all { it in -scale..scale })
    }

    @Test
    fun testWeightInitializationWithDifferentFilters() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 4 // New number of filters
        val kernelSize = Pair(3, 3)
        conv2D = Conv2D(inputShape, filters, kernelSize, random = random)

        val scale = sqrt(2.0 / (inputShape[2] * kernelSize.first * kernelSize.second))
        assertEquals(filters, conv2D.weights.shape[0])
        assertEquals(inputShape[2], conv2D.weights.shape[1])
        assertEquals(kernelSize.first, conv2D.weights.shape[2])
        assertEquals(kernelSize.second, conv2D.weights.shape[3])
        assertTrue(conv2D.weights.all { it in -scale..scale })
    }

    @Test
    fun testWeightInitializationWithDifferentStrides() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = Pair(3, 3)
        val strides = intArrayOf(2, 2) // Different strides
        conv2D = Conv2D(inputShape, filters, kernelSize, strides = strides, padding = "valid", random = random)

        val scale = sqrt(2.0 / (inputShape[2] * kernelSize.first * kernelSize.second))
        assertEquals(filters, conv2D.weights.shape[0])
        assertEquals(inputShape[2], conv2D.weights.shape[1])
        assertEquals(kernelSize.first, conv2D.weights.shape[2])
        assertEquals(kernelSize.second, conv2D.weights.shape[3])
        assertTrue(conv2D.weights.all { it in -scale..scale })
    }

    @Test
    fun testBiasInitialization() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = Pair(3, 3)
        conv2D = Conv2D(inputShape, filters, kernelSize, random = random)

        assertEquals(filters, conv2D.biases.size)
        assertTrue(conv2D.biases.all { it == 0.01 })
    }

    @Test
    fun testBiasInitializationWithDifferentInputShape() {
        val inputShape = intArrayOf(1, 7, 7, 4) // New input shape
        val filters = 3
        val kernelSize = Pair(3, 3)
        conv2D = Conv2D(inputShape, filters, kernelSize, random = random)

        assertEquals(filters, conv2D.biases.size)
        assertTrue(conv2D.biases.all { it == 0.01 })
    }

    @Test
    fun testBiasInitializationWithDifferentKernelSize() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = Pair(5, 5) // New kernel size
        conv2D = Conv2D(inputShape, filters, kernelSize, random = random)

        assertEquals(filters, conv2D.biases.size)
        assertTrue(conv2D.biases.all { it == 0.01 })
    }

    @Test
    fun testBiasInitializationWithDifferentFilters() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 4 // New number of filters
        val kernelSize = Pair(3, 3)
        conv2D = Conv2D(inputShape, filters, kernelSize, random = random)

        assertEquals(filters, conv2D.biases.size)
        assertTrue(conv2D.biases.all { it == 0.01 })
    }

    @Test
    fun testBiasInitializationWithDifferentRandomInitialization() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = Pair(3, 3)
        val random = Random(42) // Different random seed
        conv2D = Conv2D(inputShape, filters, kernelSize, random = random)

        assertEquals(filters, conv2D.biases.size)
        assertTrue(conv2D.biases.all { it == 0.01 })
    }


    @Test
    fun testForwardPassWithValidPadding() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = Pair(3, 3)
        conv2D = Conv2D(inputShape, filters, kernelSize, padding = "valid", random = random)

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        val expectedHeight = inputShape[1] - kernelSize.first + 1
        val expectedWidth = inputShape[2] - kernelSize.second + 1
        assertEquals(intArrayOf(1, expectedHeight, expectedWidth, filters).toList(), output.shape.toList())
    }

    @Test
    fun testForwardPassWithValidPaddingAndDifferentInputShape() {
        val inputShape = intArrayOf(1, 7, 7, 4) // New input shape
        val filters = 3
        val kernelSize = Pair(3, 3)
        conv2D = Conv2D(inputShape, filters, kernelSize, padding = "valid", random = random)

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        val expectedHeight = inputShape[1] - kernelSize.first + 1
        val expectedWidth = inputShape[2] - kernelSize.second + 1
        assertEquals(intArrayOf(1, expectedHeight, expectedWidth, filters).toList(), output.shape.toList())
    }

    @Test
    fun testForwardPassWithValidPaddingAndDifferentKernelSize() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = Pair(5, 5) // New kernel size
        conv2D = Conv2D(inputShape, filters, kernelSize, padding = "valid", random = random)

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        val expectedHeight = inputShape[1] - kernelSize.first + 1
        val expectedWidth = inputShape[2] - kernelSize.second + 1
        assertEquals(intArrayOf(1, expectedHeight, expectedWidth, filters).toList(), output.shape.toList())
    }

    @Test
    fun testForwardPassWithValidPaddingAndDifferentFilters() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 4 // New number of filters
        val kernelSize = Pair(3, 3)
        conv2D = Conv2D(inputShape, filters, kernelSize, padding = "valid", random = random)

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        val expectedHeight = inputShape[1] - kernelSize.first + 1
        val expectedWidth = inputShape[2] - kernelSize.second + 1
        assertEquals(intArrayOf(1, expectedHeight, expectedWidth, filters).toList(), output.shape.toList())
    }

    @Test
    fun testForwardPassWithValidPaddingAndDifferentStrides() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = Pair(3, 3)
        val strides = intArrayOf(2, 2) // Different strides
        conv2D = Conv2D(inputShape, filters, kernelSize, strides = strides, padding = "valid", random = random)

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        val expectedHeight = (inputShape[1] - kernelSize.first) / strides[0] + 1
        val expectedWidth = (inputShape[2] - kernelSize.second) / strides[1] + 1
        assertEquals(intArrayOf(1, expectedHeight, expectedWidth, filters).toList(), output.shape.toList())
    }


    @Test
    fun testForwardPassWithSamePadding() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = Pair(3, 3)
        conv2D = Conv2D(inputShape, filters, kernelSize, padding = "same", random = random)

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        val expectedHeight = inputShape[1]
        val expectedWidth = inputShape[2]
        assertEquals(intArrayOf(1, expectedHeight, expectedWidth, filters).toList(), output.shape.toList())
    }

    @Test
    fun testForwardPassWithSamePaddingAndDifferentInputShape() {
        val inputShape = intArrayOf(1, 7, 7, 4) // New input shape
        val filters = 3
        val kernelSize = Pair(3, 3)
        conv2D = Conv2D(inputShape, filters, kernelSize, padding = "same", random = random)

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        val expectedHeight = inputShape[1]
        val expectedWidth = inputShape[2]
        assertEquals(intArrayOf(1, expectedHeight, expectedWidth, filters).toList(), output.shape.toList())
    }

    @Test
    fun testForwardPassWithSamePaddingAndDifferentKernelSize() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = Pair(5, 5) // New kernel size
        conv2D = Conv2D(inputShape, filters, kernelSize, padding = "same", random = random)

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        val expectedHeight = inputShape[1]
        val expectedWidth = inputShape[2]
        assertEquals(intArrayOf(1, expectedHeight, expectedWidth, filters).toList(), output.shape.toList())
    }

    @Test
    fun testForwardPassWithSamePaddingAndDifferentFilters() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 4 // New number of filters
        val kernelSize = Pair(3, 3)
        conv2D = Conv2D(inputShape, filters, kernelSize, padding = "same", random = random)

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        val expectedHeight = inputShape[1]
        val expectedWidth = inputShape[2]
        assertEquals(intArrayOf(1, expectedHeight, expectedWidth, filters).toList(), output.shape.toList())
    }

    @Test
    fun testForwardPassWithSamePaddingAndDifferentStrides() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = Pair(3, 3)
        val strides = intArrayOf(2, 2) // Different strides
        conv2D = Conv2D(inputShape, filters, kernelSize, strides = strides, padding = "same", random = random)

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        val expectedHeight = inputShape[1] / strides[0]
        val expectedWidth = inputShape[2] / strides[1]
        assertEquals(intArrayOf(1, expectedHeight, expectedWidth, filters).toList(), output.shape.toList())
    }

    @Test
    fun testForwardPassWithDifferentStrides() {
        val inputShape = intArrayOf(1, 5, 5, 3) // Input with 3 channels
        val filters = 2
        val kernelSize = Pair(2, 2)
        val strides = intArrayOf(2, 2)
        conv2D = Conv2D(inputShape, filters, kernelSize, strides = strides, padding = "valid", random = random)

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        val expectedHeight = (inputShape[1] - kernelSize.first) / strides[0] + 1
        val expectedWidth = (inputShape[2] - kernelSize.second) / strides[1] + 1
        assertEquals(intArrayOf(1, expectedHeight, expectedWidth, filters).toList(), output.shape.toList())
    }

    @Test
    fun testForwardPassWithDifferentInputShapeAndStrides() {
        val inputShape = intArrayOf(1, 7, 7, 4) // New input shape
        val filters = 3
        val kernelSize = Pair(3, 3)
        val strides = intArrayOf(2, 2)
        conv2D = Conv2D(inputShape, filters, kernelSize, strides = strides, padding = "valid", random = random)

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        val expectedHeight = (inputShape[1] - kernelSize.first) / strides[0] + 1
        val expectedWidth = (inputShape[2] - kernelSize.second) / strides[1] + 1
        assertEquals(intArrayOf(1, expectedHeight, expectedWidth, filters).toList(), output.shape.toList())
    }

    @Test
    fun testForwardPassWithDifferentKernelSizeAndStrides() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = Pair(3, 3) // New kernel size
        val strides = intArrayOf(2, 2)
        conv2D = Conv2D(inputShape, filters, kernelSize, strides = strides, padding = "valid", random = random)

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        val expectedHeight = (inputShape[1] - kernelSize.first) / strides[0] + 1
        val expectedWidth = (inputShape[2] - kernelSize.second) / strides[1] + 1
        assertEquals(intArrayOf(1, expectedHeight, expectedWidth, filters).toList(), output.shape.toList())
    }

    @Test
    fun testForwardPassWithDifferentFiltersAndStrides() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 4 // New number of filters
        val kernelSize = Pair(2, 2)
        val strides = intArrayOf(2, 2)
        conv2D = Conv2D(inputShape, filters, kernelSize, strides = strides, padding = "valid", random = random)

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        val expectedHeight = (inputShape[1] - kernelSize.first) / strides[0] + 1
        val expectedWidth = (inputShape[2] - kernelSize.second) / strides[1] + 1
        assertEquals(intArrayOf(1, expectedHeight, expectedWidth, filters).toList(), output.shape.toList())
    }

    @Test
    fun testForwardPassWithDifferentPaddingAndStrides() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = Pair(2, 2)
        val strides = intArrayOf(2, 2)
        conv2D = Conv2D(
            inputShape,
            filters,
            kernelSize,
            strides = strides,
            padding = "same",
            random = random
        ) // Different padding

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        val expectedHeight = inputShape[1] / strides[0]
        val expectedWidth = inputShape[2] / strides[1]
        assertEquals(intArrayOf(1, expectedHeight, expectedWidth, filters).toList(), output.shape.toList())
    }


    @Test
    fun testForwardPassWithActivationFunction() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = Pair(3, 3)
        val activation = ReLU()
        conv2D = Conv2D(inputShape, filters, kernelSize, random = random, activation = activation)

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        assertEquals(intArrayOf(1, 3, 3, filters).toList(), output.shape.toList()) // Assuming valid padding
        assertTrue(output.all { it >= 0 }) // Check if output is non-negative due to ReLU
    }

    @Test
    fun testForwardPassWithDifferentInputShape() {
        val inputShape = intArrayOf(1, 7, 7, 4) // New input shape
        val filters = 3
        val kernelSize = Pair(3, 3)
        val activation = ReLU()
        conv2D = Conv2D(inputShape, filters, kernelSize, random = random, activation = activation)

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        assertEquals(intArrayOf(1, 5, 5, filters).toList(), output.shape.toList()) // Assuming valid padding
        assertTrue(output.all { it >= 0 }) // Check if output is non-negative due to ReLU
    }

    @Test
    fun testForwardPassWithDifferentKernelSize() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = Pair(5, 5) // New kernel size
        val activation = ReLU()
        conv2D = Conv2D(inputShape, filters, kernelSize, random = random, activation = activation)

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        assertEquals(intArrayOf(1, 1, 1, filters).toList(), output.shape.toList()) // Assuming valid padding
        assertTrue(output.all { it >= 0 }) // Check if output is non-negative due to ReLU
    }

    @Test
    fun testForwardPassWithDifferentFilters() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 4 // New number of filters
        val kernelSize = Pair(3, 3)
        val activation = ReLU()
        conv2D = Conv2D(inputShape, filters, kernelSize, random = random, activation = activation)

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        assertEquals(intArrayOf(1, 3, 3, filters).toList(), output.shape.toList()) // Assuming valid padding
        assertTrue(output.all { it >= 0 }) // Check if output is non-negative due to ReLU
    }

    @Test
    fun testForwardPassWithDifferentPadding() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = Pair(3, 3)
        val activation = ReLU()
        conv2D = Conv2D(
            inputShape,
            filters,
            kernelSize,
            random = random,
            padding = "same",
            activation = activation
        ) // Different padding

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        assertEquals(intArrayOf(1, 5, 5, filters).toList(), output.shape.toList()) // Assuming same padding
        assertTrue(output.all { it >= 0 }) // Check if output is non-negative due to ReLU
    }

//    @Test
//    fun testBackwardPass() {
//        val inputShape = intArrayOf(1, 5, 5, 3)
//        val filters = 2
//        val kernelSize = intArrayOf(3, 3)
//        cnnLayer = CNNLayer(inputShape, filters, kernelSize, random = random, padding = "valid")
//
//        val input = createRandomInput(inputShape)
//        val output = cnnLayer.forward(input)
//
//        val dvalues = createRandomInput(output.shape)
//        val dinputs = cnnLayer.backward(dvalues)
//
//        assertEquals(inputShape.toList(), dinputs.shape.toList())
//        assertEquals(cnnLayer.weights.shape.toList(), cnnLayer.dweights.shape.toList())
//        assertEquals(filters, cnnLayer.dbiases.size)
//    }
//
//    @Test
//    fun testBackwardPassDifferentInputShape() {
//        val inputShape = intArrayOf(1, 7, 7, 4) // New input shape
//        val filters = 3
//        val kernelSize = intArrayOf(3, 3)
//        cnnLayer = CNNLayer(inputShape, filters, kernelSize, random = random, padding = "valid")
//
//        val input = createRandomInput(inputShape)
//        val output = cnnLayer.forward(input)
//
//        val dvalues = createRandomInput(output.shape)
//        val dinputs = cnnLayer.backward(dvalues)
//
//        assertEquals(inputShape.toList(), dinputs.shape.toList())
//        assertEquals(cnnLayer.weights.shape.toList(), cnnLayer.dweights.shape.toList())
//        assertEquals(filters, cnnLayer.dbiases.size)
//    }
//
//    @Test
//    fun testBackwardPassDifferentKernelSize() {
//        val inputShape = intArrayOf(1, 5, 5, 3)
//        val filters = 2
//        val kernelSize = intArrayOf(5, 5) // New kernel size
//        cnnLayer = CNNLayer(inputShape, filters, kernelSize, random = random, padding = "valid")
//
//        val input = createRandomInput(inputShape)
//        val output = cnnLayer.forward(input)
//
//        val dvalues = createRandomInput(output.shape)
//        val dinputs = cnnLayer.backward(dvalues)
//
//        assertEquals(inputShape.toList(), dinputs.shape.toList())
//        assertEquals(cnnLayer.weights.shape.toList(), cnnLayer.dweights.shape.toList())
//        assertEquals(filters, cnnLayer.dbiases.size)
//    }
//
//    @Test
//    fun testBackwardPassDifferentFilters() {
//        val inputShape = intArrayOf(1, 5, 5, 3)
//        val filters = 4 // New number of filters
//        val kernelSize = intArrayOf(3, 3)
//        cnnLayer = CNNLayer(inputShape, filters, kernelSize, random = random, padding = "valid")
//
//        val input = createRandomInput(inputShape)
//        val output = cnnLayer.forward(input)
//
//        val dvalues = createRandomInput(output.shape)
//        val dinputs = cnnLayer.backward(dvalues)
//
//        assertEquals(inputShape.toList(), dinputs.shape.toList())
//        assertEquals(cnnLayer.weights.shape.toList(), cnnLayer.dweights.shape.toList())
//        assertEquals(filters, cnnLayer.dbiases.size)
//    }
//
//    @Test
//    fun testBackwardPassDifferentPadding() {
//        val inputShape = intArrayOf(1, 5, 5, 3)
//        val filters = 2
//        val kernelSize = intArrayOf(3, 3)
//        cnnLayer = CNNLayer(inputShape, filters, kernelSize, random = random, padding = "same") // Different padding
//
//        val input = createRandomInput(inputShape)
//        val output = cnnLayer.forward(input)
//
//        val dvalues = createRandomInput(output.shape)
//        val dinputs = cnnLayer.backward(dvalues)
//
//        assertEquals(inputShape.toList(), dinputs.shape.toList())
//        assertEquals(cnnLayer.weights.shape.toList(), cnnLayer.dweights.shape.toList())
//        assertEquals(filters, cnnLayer.dbiases.size)
//    }

    @Test
    fun testEdgeCaseWithMinimalInput() {
        val inputShape = intArrayOf(1, 1, 1, 1) // Minimal input
        val filters = 1
        val kernelSize = Pair(1, 1)
        conv2D = Conv2D(inputShape, filters, kernelSize, random = random)

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        assertEquals(intArrayOf(1, 1, 1, filters).toList(), output.shape.toList())
    }
}
