package io.ai4kt.ai4kt.fibonacci.tensorflow.layers

import io.ai4kt.ai4kt.fibonacci.tensorflow.activations.ReLU
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.all
import kotlin.test.Test
import kotlin.test.assertTrue

class DNNLayerTest {

    private val mk = Multik

    @Test
    fun testForwardPassWithoutActivation() {
        // Create a DNNLayer with 3 inputs, 5 neurons, and no activation function
        val layer = DNNLayer(3, 5)

        // Example input data (2 samples, 3 features each)
        val inputs: D2Array<Double> = mk.ndarray(
            listOf(
                listOf(1.0, 2.0, 3.0),
                listOf(4.0, 5.0, 6.0)
            )
        )

        // Perform forward pass
        val output = layer.forward(inputs)

        // Verify that the output has the correct shape (2 samples, 5 neurons)
        assertTrue(output.shape.contentEquals(intArrayOf(2, 5)), "Output shape is incorrect")
    }

    @Test
    fun testForwardPassWithReLU() {
        // Create a DNNLayer with 3 inputs, 5 neurons, and ReLU activation
        val layer = DNNLayer(3, 5, ReLU())

        // Example input data (2 samples, 3 features each)
        val inputs: D2Array<Double> = mk.ndarray(
            listOf(
                listOf(1.0, 2.0, 3.0),
                listOf(-4.0, -5.0, -6.0)
            )
        )

        // Perform forward pass
        val output = layer.forward(inputs)

        // Verify that the output has the correct shape (2 samples, 5 neurons)
        assertTrue(output.shape.contentEquals(intArrayOf(2, 5)), "Output shape is incorrect")

        // Verify that ReLU activation was applied (no negative values in output)
        assertTrue(output.all { it >= 0.0 }, "ReLU activation was not applied correctly")
    }

    @Test
    fun testBackwardPassWithoutActivation() {
        // Create a DNNLayer with 3 inputs, 5 neurons, and no activation function
        val layer = DNNLayer(3, 5)

        // Example input data (2 samples, 3 features each)
        val inputs: D2Array<Double> = mk.ndarray(
            listOf(
                listOf(1.0, 2.0, 3.0),
                listOf(4.0, 5.0, 6.0)
            )
        )

        // Perform forward pass to cache inputs
        layer.forward(inputs)

        // Example gradient from the next layer (2 samples, 5 neurons)
        val dvalues: D2Array<Double> = mk.ndarray(
            listOf(
                listOf(0.1, 0.2, 0.3, 0.4, 0.5),
                listOf(0.5, 0.4, 0.3, 0.2, 0.1)
            )
        )

        // Perform backward pass
        val dinputs = layer.backward(dvalues)

        // Verify that the gradients for inputs have the correct shape (2 samples, 3 features)
        assertTrue(dinputs.shape.contentEquals(intArrayOf(2, 3)), "Gradients for inputs have incorrect shape")
    }

    @Test
    fun testBackwardPassWithReLU() {
        // Create a DNNLayer with 3 inputs, 5 neurons, and ReLU activation
        val layer = DNNLayer(3, 5, ReLU())

        // Example input data (2 samples, 3 features each)
        val inputs: D2Array<Double> = mk.ndarray(
            listOf(
                listOf(1.0, 2.0, 3.0),
                listOf(-4.0, -5.0, -6.0)
            )
        )

        // Perform forward pass to cache inputs
        layer.forward(inputs)

        // Example gradient from the next layer (2 samples, 5 neurons)
        val dvalues: D2Array<Double> = mk.ndarray(
            listOf(
                listOf(0.1, 0.2, 0.3, 0.4, 0.5),
                listOf(0.5, 0.4, 0.3, 0.2, 0.1)
            )
        )

        // Perform backward pass
        val dinputs = layer.backward(dvalues)

        // Verify that the gradients for inputs have the correct shape (2 samples, 3 features)
        assertTrue(dinputs.shape.contentEquals(intArrayOf(2, 3)), "Gradients for inputs have incorrect shape")
    }
}