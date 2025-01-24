package io.ai4kt.ai4kt.fibonacci.tensorflow.activations

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.test.Test
import kotlin.test.assertEquals

class ReLUTest {

    private val mk = Multik

    @Test
    fun testForward() {
        val relu = ReLU()

        // Create a 2D input array
        val inputs: D2Array<Double> = mk.ndarray(
            listOf(
                listOf(1.0, -2.0, 3.0),
                listOf(-4.0, 5.0, -6.0)
            )
        )

        // Perform forward pass
        val output = relu.forward(inputs)

        // Expected output after applying ReLU
        val expectedOutput: D2Array<Double> = mk.ndarray(
            listOf(
                listOf(1.0, 0.0, 3.0),
                listOf(0.0, 5.0, 0.0)
            )
        )

        // Verify that the output matches the expected output
        assertEquals(expectedOutput, output, "ReLU forward pass failed")
    }

    @Test
    fun testBackward() {
        val relu = ReLU()

        // Create a 2D input array
        val inputs: D2Array<Double> = mk.ndarray(
            listOf(
                listOf(1.0, -2.0, 3.0),
                listOf(-4.0, 5.0, -6.0)
            )
        )

        // Perform forward pass to set the internal `inputs` state
        relu.forward(inputs)

        // Create a 2D gradient array (dvalues)
        val dvalues: D2Array<Double> = mk.ndarray(
            listOf(
                listOf(0.1, 0.2, 0.3),
                listOf(0.4, 0.5, 0.6)
            )
        )

        // Perform backward pass
        val dinputs = relu.backward(dvalues, inputs)

        // Expected gradients after applying ReLU backward pass
        val expectedDinputs: D2Array<Double> = mk.ndarray(
            listOf(
                listOf(0.1, 0.0, 0.3),
                listOf(0.0, 0.5, 0.0)
            )
        )

        // Verify that the gradients match the expected gradients
        assertEquals(expectedDinputs, dinputs, "ReLU backward pass failed")
    }
}