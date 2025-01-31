package tensorflow.layers

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import tensorflow.KernelSize
import tensorflow.activations.ReLU
import kotlin.math.ceil
import kotlin.math.sqrt
import kotlin.test.*
import kotlin.random.Random

class Conv2DTest {

    private lateinit var conv2D: Conv2D
    private val random = Random(42)

    private fun createRandomInput(shape: IntArray): NDArray<Double, *> {
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
        val kernelSize = KernelSize(3, 3)
        conv2D = Conv2D(inputShape, filters, kernelSize, random = random)

        val scale = sqrt(2.0 / (inputShape[3] * kernelSize.height * kernelSize.width))
        assertEquals(filters, conv2D.weights.shape[3])
        assertEquals(inputShape[3], conv2D.weights.shape[2])
        assertEquals(kernelSize.height, conv2D.weights.shape[0])
        assertEquals(kernelSize.width, conv2D.weights.shape[1])
        assertTrue(conv2D.weights.all { it in -scale..scale })
    }

    @Test
    fun testWeightInitializationWithDifferentInputShape() {
        val inputShape = intArrayOf(1, 7, 7, 4)
        val filters = 3
        val kernelSize = KernelSize(3, 3)
        conv2D = Conv2D(inputShape, filters, kernelSize, random = random)

        val scale = sqrt(2.0 / (inputShape[3] * kernelSize.height * kernelSize.width))
        assertEquals(filters, conv2D.weights.shape[3])
        assertEquals(inputShape[3], conv2D.weights.shape[2])
        assertEquals(kernelSize.height, conv2D.weights.shape[0])
        assertEquals(kernelSize.width, conv2D.weights.shape[1])
        assertTrue(conv2D.weights.all { it in -scale..scale })
    }

    @Test
    fun testWeightInitializationWithDifferentKernelSize() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = KernelSize(5, 5)
        conv2D = Conv2D(inputShape, filters, kernelSize, random = random)

        val scale = sqrt(2.0 / (inputShape[3] * kernelSize.height * kernelSize.width))
        assertEquals(filters, conv2D.weights.shape[3])
        assertEquals(inputShape[3], conv2D.weights.shape[2])
        assertEquals(kernelSize.height, conv2D.weights.shape[0])
        assertEquals(kernelSize.width, conv2D.weights.shape[1])
        assertTrue(conv2D.weights.all { it in -scale..scale })
    }

    @Test
    fun testWeightInitializationWithDifferentFilters() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 4
        val kernelSize = KernelSize(3, 3)
        conv2D = Conv2D(inputShape, filters, kernelSize, random = random)

        val scale = sqrt(2.0 / (inputShape[3] * kernelSize.height * kernelSize.width))
        assertEquals(filters, conv2D.weights.shape[3])
        assertEquals(inputShape[3], conv2D.weights.shape[2])
        assertEquals(kernelSize.height, conv2D.weights.shape[0])
        assertEquals(kernelSize.width, conv2D.weights.shape[1])
        assertTrue(conv2D.weights.all { it in -scale..scale })
    }

    @Test
    fun testWeightInitializationWithDifferentStrides() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = KernelSize(3, 3)
        val strides = intArrayOf(2, 2)
        conv2D = Conv2D(inputShape, filters, kernelSize, strides = strides, padding = "valid", random = random)

        val scale = sqrt(2.0 / (inputShape[3] * kernelSize.height * kernelSize.width))
        assertEquals(filters, conv2D.weights.shape[3])
        assertEquals(inputShape[3], conv2D.weights.shape[2])
        assertEquals(kernelSize.height, conv2D.weights.shape[0])
        assertEquals(kernelSize.width, conv2D.weights.shape[1])
        assertTrue(conv2D.weights.all { it in -scale..scale })
    }

    @Test
    fun testBiasInitialization() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = KernelSize(3, 3)
        conv2D = Conv2D(inputShape, filters, kernelSize, random = random)

        assertEquals(filters, conv2D.biases.size)
        assertTrue(conv2D.biases.all { it == 0.01 })
    }

    @Test
    fun testBiasInitializationWithDifferentInputShape() {
        val inputShape = intArrayOf(1, 7, 7, 4)
        val filters = 3
        val kernelSize = KernelSize(3, 3)
        conv2D = Conv2D(inputShape, filters, kernelSize, random = random)

        assertEquals(filters, conv2D.biases.size)
        assertTrue(conv2D.biases.all { it == 0.01 })
    }

    @Test
    fun testBiasInitializationWithDifferentKernelSize() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = KernelSize(5, 5)
        conv2D = Conv2D(inputShape, filters, kernelSize, random = random)

        assertEquals(filters, conv2D.biases.size)
        assertTrue(conv2D.biases.all { it == 0.01 })
    }

    @Test
    fun testBiasInitializationWithDifferentFilters() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 4
        val kernelSize = KernelSize(3, 3)
        conv2D = Conv2D(inputShape, filters, kernelSize, random = random)

        assertEquals(filters, conv2D.biases.size)
        assertTrue(conv2D.biases.all { it == 0.01 })
    }

    @Test
    fun testBiasInitializationWithDifferentRandomInitialization() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = KernelSize(3, 3)
        val random = Random(42)
        conv2D = Conv2D(inputShape, filters, kernelSize, random = random)

        assertEquals(filters, conv2D.biases.size)
        assertTrue(conv2D.biases.all { it == 0.01 })
    }


    @Test
    fun testForwardPassWithValidPadding() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = KernelSize(3, 3)
        conv2D = Conv2D(inputShape, filters, kernelSize, padding = "valid", random = random)

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        val expectedHeight = inputShape[1] - kernelSize.height + 1
        val expectedWidth = inputShape[2] - kernelSize.width + 1
        assertEquals(intArrayOf(1, expectedHeight, expectedWidth, filters).toList(), output.shape.toList())
    }

    @Test
    fun testForwardPassWithValidPaddingAndDifferentInputShape() {
        val inputShape = intArrayOf(1, 7, 7, 4)
        val filters = 3
        val kernelSize = KernelSize(3, 3)
        conv2D = Conv2D(inputShape, filters, kernelSize, padding = "valid", random = random)

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        val expectedHeight = inputShape[1] - kernelSize.height + 1
        val expectedWidth = inputShape[2] - kernelSize.width + 1
        assertEquals(intArrayOf(1, expectedHeight, expectedWidth, filters).toList(), output.shape.toList())
    }

    @Test
    fun testForwardPassWithValidPaddingAndDifferentKernelSize() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = KernelSize(5, 5)
        conv2D = Conv2D(inputShape, filters, kernelSize, padding = "valid", random = random)

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        val expectedHeight = inputShape[1] - kernelSize.height + 1
        val expectedWidth = inputShape[2] - kernelSize.width + 1
        assertEquals(intArrayOf(1, expectedHeight, expectedWidth, filters).toList(), output.shape.toList())
    }

    @Test
    fun testForwardPassWithValidPaddingAndDifferentFilters() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 4
        val kernelSize = KernelSize(3, 3)
        conv2D = Conv2D(inputShape, filters, kernelSize, padding = "valid", random = random)

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        val expectedHeight = inputShape[1] - kernelSize.height + 1
        val expectedWidth = inputShape[2] - kernelSize.width + 1
        assertEquals(intArrayOf(1, expectedHeight, expectedWidth, filters).toList(), output.shape.toList())
    }

    @Test
    fun testForwardPassWithValidPaddingAndDifferentStrides() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = KernelSize(3, 3)
        val strides = intArrayOf(2, 2)
        conv2D = Conv2D(inputShape, filters, kernelSize, strides = strides, padding = "valid", random = random)

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        val expectedHeight = (inputShape[1] - kernelSize.height) / strides[0] + 1
        val expectedWidth = (inputShape[2] - kernelSize.width) / strides[1] + 1
        assertEquals(intArrayOf(1, expectedHeight, expectedWidth, filters).toList(), output.shape.toList())
    }


    @Test
    fun testForwardPassWithSamePadding() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = KernelSize(3, 3)
        conv2D = Conv2D(inputShape, filters, kernelSize, padding = "same", random = random)

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        val expectedHeight = inputShape[1]
        val expectedWidth = inputShape[2]
        assertEquals(intArrayOf(1, expectedHeight, expectedWidth, filters).toList(), output.shape.toList())
    }

    @Test
    fun testForwardPassWithSamePaddingAndDifferentInputShape() {
        val inputShape = intArrayOf(1, 7, 7, 4)
        val filters = 3
        val kernelSize = KernelSize(3, 3)
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
        val kernelSize = KernelSize(5, 5)
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
        val filters = 4
        val kernelSize = KernelSize(3, 3)
        conv2D = Conv2D(inputShape, filters, kernelSize, padding = "same", random = random)

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        val expectedHeight = inputShape[1]
        val expectedWidth = inputShape[2]
        assertEquals(intArrayOf(1, expectedHeight, expectedWidth, filters).toList(), output.shape.toList())
    }

    @Test
    fun testForwardPassWithSamePaddingAndDifferentStrides() {
        val inputShape = intArrayOf(1, 5, 5, 3) // Batch size, height, width, channels
        val filters = 2
        val kernelSize = KernelSize(3, 3) // Kernel height and width
        val strides = intArrayOf(2, 2) // Stride height and width
        conv2D = Conv2D(inputShape, filters, kernelSize, strides = strides, padding = "same", random = random)

        // Ensure weights are initialized with TensorFlow's shape: [kernel_h, kernel_w, input_channels, filters]
        val expectedWeightsShape = listOf(kernelSize.height, kernelSize.width, inputShape[3], filters)
        assertEquals(expectedWeightsShape, conv2D.weights.shape.toList())

        // Create random input tensor
        val input = createRandomInput(inputShape)

        // Perform forward pass
        val output = conv2D.forward(input)

        // Calculate expected output shape for "same" padding
        val expectedHeight = ceil(inputShape[1] / strides[0].toDouble()).toInt()
        val expectedWidth = ceil(inputShape[2] / strides[1].toDouble()).toInt()
        val expectedShape = intArrayOf(1, expectedHeight, expectedWidth, filters).toList()

        // Assert that the output shape matches the expected shape
        assertEquals(expectedShape, output.shape.toList())
    }

    @Test
    fun testForwardPassWithSamePaddingAndDifferentStridesEdgeCase() {
        val inputShape = intArrayOf(1, 6, 6, 3) // Batch size, height, width, channels
        val filters = 2
        val kernelSize = KernelSize(3, 3)
        val strides = intArrayOf(2, 2)
        conv2D = Conv2D(inputShape, filters, kernelSize, strides = strides, padding = "same", random = random)

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        // Calculate expected output shape for "same" padding
        val expectedHeight = ceil(inputShape[1] / strides[0].toDouble()).toInt()
        val expectedWidth = ceil(inputShape[2] / strides[1].toDouble()).toInt()
        val expectedShape = intArrayOf(1, expectedHeight, expectedWidth, filters).toList()

        // Assert that the output shape matches the expected shape
        assertEquals(expectedShape, output.shape.toList())
    }

    @Test
    fun testForwardPassWithDifferentStrides() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = KernelSize(2, 2)
        val strides = intArrayOf(2, 2)
        conv2D = Conv2D(inputShape, filters, kernelSize, strides = strides, padding = "valid", random = random)

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        val expectedHeight = (inputShape[1] - kernelSize.height) / strides[0] + 1
        val expectedWidth = (inputShape[2] - kernelSize.width) / strides[1] + 1
        assertEquals(intArrayOf(1, expectedHeight, expectedWidth, filters).toList(), output.shape.toList())
    }

    @Test
    fun testForwardPassWithDifferentInputShapeAndStrides() {
        val inputShape = intArrayOf(1, 7, 7, 4)
        val filters = 3
        val kernelSize = KernelSize(3, 3)
        val strides = intArrayOf(2, 2)
        conv2D = Conv2D(inputShape, filters, kernelSize, strides = strides, padding = "valid", random = random)

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        val expectedHeight = (inputShape[1] - kernelSize.height) / strides[0] + 1
        val expectedWidth = (inputShape[2] - kernelSize.width) / strides[1] + 1
        assertEquals(intArrayOf(1, expectedHeight, expectedWidth, filters).toList(), output.shape.toList())
    }

    @Test
    fun testForwardPassWithDifferentKernelSizeAndStrides() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = KernelSize(3, 3)
        val strides = intArrayOf(2, 2)
        conv2D = Conv2D(inputShape, filters, kernelSize, strides = strides, padding = "valid", random = random)

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        val expectedHeight = (inputShape[1] - kernelSize.height) / strides[0] + 1
        val expectedWidth = (inputShape[2] - kernelSize.width) / strides[1] + 1
        assertEquals(intArrayOf(1, expectedHeight, expectedWidth, filters).toList(), output.shape.toList())
    }

    @Test
    fun testForwardPassWithDifferentFiltersAndStrides() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 4
        val kernelSize = KernelSize(2, 2)
        val strides = intArrayOf(2, 2)
        conv2D = Conv2D(inputShape, filters, kernelSize, strides = strides, padding = "valid", random = random)

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        val expectedHeight = (inputShape[1] - kernelSize.height) / strides[0] + 1
        val expectedWidth = (inputShape[2] - kernelSize.width) / strides[1] + 1
        assertEquals(intArrayOf(1, expectedHeight, expectedWidth, filters).toList(), output.shape.toList())
    }

    @Test
    fun testForwardPassWithDifferentPaddingAndStrides() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = KernelSize(2, 2)
        val strides = intArrayOf(2, 2)
        conv2D = Conv2D(
            inputShape,
            filters,
            kernelSize,
            strides = strides,
            padding = "same",
            random = random
        )

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        val expectedHeight = ceil(inputShape[1] / strides[0].toDouble()).toInt()
        val expectedWidth = ceil(inputShape[2] / strides[1].toDouble()).toInt()
        assertEquals(intArrayOf(1, expectedHeight, expectedWidth, filters).toList(), output.shape.toList())
    }


    @Test
    fun testForwardPassWithActivationFunction() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = KernelSize(3, 3)
        val activation = ReLU()
        conv2D = Conv2D(inputShape, filters, kernelSize, random = random, activation = activation)

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        assertEquals(intArrayOf(1, 3, 3, filters).toList(), output.shape.toList())
        assertTrue(output.all { it >= 0 })
    }

    @Test
    fun testForwardPassWithDifferentInputShape() {
        val inputShape = intArrayOf(1, 7, 7, 4)
        val filters = 3
        val kernelSize = KernelSize(3, 3)
        val activation = ReLU()
        conv2D = Conv2D(inputShape, filters, kernelSize, random = random, activation = activation)

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        assertEquals(intArrayOf(1, 5, 5, filters).toList(), output.shape.toList())
        assertTrue(output.all { it >= 0 })
    }

    @Test
    fun testForwardPassWithDifferentKernelSize() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = KernelSize(5, 5)
        val activation = ReLU()
        conv2D = Conv2D(inputShape, filters, kernelSize, random = random, activation = activation)

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        assertEquals(intArrayOf(1, 1, 1, filters).toList(), output.shape.toList())
        assertTrue(output.all { it >= 0 })
    }

    @Test
    fun testForwardPassWithDifferentFilters() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 4
        val kernelSize = KernelSize(3, 3)
        val activation = ReLU()
        conv2D = Conv2D(inputShape, filters, kernelSize, random = random, activation = activation)

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        assertEquals(intArrayOf(1, 3, 3, filters).toList(), output.shape.toList())
        assertTrue(output.all { it >= 0 })
    }

    @Test
    fun testForwardPassWithDifferentPadding() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = KernelSize(3, 3)
        val activation = ReLU()
        conv2D = Conv2D(
            inputShape,
            filters,
            kernelSize,
            random = random,
            padding = "same",
            activation = activation
        )

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        assertEquals(intArrayOf(1, 5, 5, filters).toList(), output.shape.toList())
        assertTrue(output.all { it >= 0 })
    }

    @Test
    fun testBackwardPass() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = KernelSize(3, 3)
        conv2D = Conv2D(inputShape, filters, kernelSize, random = random, padding = "valid")

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        val dvalues = createRandomInput(output.shape)
        val dinputs = conv2D.backward(dvalues)

        assertEquals(inputShape.toList(), dinputs.shape.toList())
        assertEquals(conv2D.weights.shape.toList(), conv2D.dweights.shape.toList())
        assertEquals(filters, conv2D.dbiases.size)
    }

    @Test
    fun testBackwardPassDifferentInputShape() {
        val inputShape = intArrayOf(1, 7, 7, 4)
        val filters = 3
        val kernelSize = KernelSize(3, 3)
        conv2D = Conv2D(inputShape, filters, kernelSize, random = random, padding = "valid")

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        val dvalues = createRandomInput(output.shape)
        val dinputs = conv2D.backward(dvalues)

        assertEquals(inputShape.toList(), dinputs.shape.toList())
        assertEquals(conv2D.weights.shape.toList(), conv2D.dweights.shape.toList())
        assertEquals(filters, conv2D.dbiases.size)
    }

    @Test
    fun testBackwardPassDifferentKernelSize() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = KernelSize(5, 5)
        conv2D = Conv2D(inputShape, filters, kernelSize, random = random, padding = "valid")

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        val dvalues = createRandomInput(output.shape)
        val dinputs = conv2D.backward(dvalues)

        assertEquals(inputShape.toList(), dinputs.shape.toList())
        assertEquals(conv2D.weights.shape.toList(), conv2D.dweights.shape.toList())
        assertEquals(filters, conv2D.dbiases.size)
    }

    @Test
    fun testBackwardPassDifferentFilters() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 4
        val kernelSize = KernelSize(3, 3)
        conv2D = Conv2D(inputShape, filters, kernelSize, random = random, padding = "valid")

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        val dvalues = createRandomInput(output.shape)
        val dinputs = conv2D.backward(dvalues)

        assertEquals(inputShape.toList(), dinputs.shape.toList())
        assertEquals(conv2D.weights.shape.toList(), conv2D.dweights.shape.toList())
        assertEquals(filters, conv2D.dbiases.size)
    }

    @Test
    fun testBackwardPassDifferentPadding() {
        val inputShape = intArrayOf(1, 5, 5, 3)
        val filters = 2
        val kernelSize = KernelSize(3, 3)
        conv2D = Conv2D(inputShape, filters, kernelSize, random = random, padding = "same")

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        val dvalues = createRandomInput(output.shape)
        val dinputs = conv2D.backward(dvalues)

        assertEquals(inputShape.toList(), dinputs.shape.toList())
        assertEquals(conv2D.weights.shape.toList(), conv2D.dweights.shape.toList())
        assertEquals(filters, conv2D.dbiases.size)
    }

    @Test
    fun testEdgeCaseWithMinimalInput() {
        val inputShape = intArrayOf(1, 1, 1, 1)
        val filters = 1
        val kernelSize = KernelSize(1, 1)
        conv2D = Conv2D(inputShape, filters, kernelSize, random = random)

        val input = createRandomInput(inputShape)
        val output = conv2D.forward(input)

        assertEquals(intArrayOf(1, 1, 1, filters).toList(), output.shape.toList())
    }

//    @Test
//    fun testConvolveBackwardInputs() {
//        // Define input parameters
//        val batchSize = 1
//        val inputHeight = 4
//        val inputWidth = 4
//        val inputChannels = 1
//        val filters = 1
//        val kernelHeight = 3
//        val kernelWidth = 3
//        val strideHeight = 1
//        val strideWidth = 1
//        val padding = "valid"
//
//        // Create dummy inputs (batch of images)
//        val inputs = mk.ndarray(mk[
//            mk[mk[1.0, 2.0, 3.0, 4.0],
//                mk[5.0, 6.0, 7.0, 8.0],
//                mk[9.0, 10.0, 11.0, 12.0],
//                mk[13.0, 14.0, 15.0, 16.0]]
//        ]).reshape(batchSize, inputHeight, inputWidth, inputChannels)
//
//        // Create dummy weights (filters)
//        val weights = mk.ndarray(mk[
//            mk[mk[1.0, 0.0, 0.0], mk[0.0, 1.0, 0.0], mk[0.0, 0.0, 1.0]]
//        ]).reshape(filters, inputChannels, kernelHeight, kernelWidth)
//
//        // Create dummy biases
//        val biases = mk.zeros<Double>(filters)
//
//        // Create dummy strides
//        val strides = intArrayOf(strideHeight, strideWidth)
//
//        // Create an instance of Conv2D
//        val conv2D = Conv2D(
//            inputShape = intArrayOf(inputHeight, inputWidth, inputChannels), // Input shape: [height, width, channels]
//            filters = filters,
//            kernelSize = KernelSize(kernelHeight, kernelWidth),
//            strides = strides,
//            padding = padding,
//            random = Random(42)
//        )
//
//        // Set weights and biases (for testing purposes)
//        conv2D.weights = weights
//        conv2D.biases = biases
//
//        // Perform the forward pass to cache the inputs
//        val output = conv2D.forward(inputs)
//
//        // Create dummy dvalues (gradients of loss w.r.t. output)
//        val dvalues = mk.ndarray(mk[
//            mk[mk[1.0, 2.0], mk[3.0, 4.0]]
//        ]).reshape(batchSize, 2, 2, filters)
//
//        // Call the function to be tested
//        val dinputs = conv2D.convolveBackwardInputs(dvalues, weights, strides)
//
//        // Define the expected output
//        val expectedDinputs = mk.ndarray(mk[
//            mk[mk[1.0, 2.0, 0.0], mk[3.0, 4.0, 0.0], mk[0.0, 0.0, 0.0]]
//        ]).reshape(batchSize, inputHeight, inputWidth, inputChannels)
//
//        // Assert that the result matches the expected output
//        assertEquals(expectedDinputs, dinputs)
//    }
}
