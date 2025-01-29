package tensorflow.activations

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import kotlin.test.Test
import kotlin.test.assertTrue

class SoftmaxTest {

    private val mk = Multik

    @Test
    fun testForward() {
        val softmax = Softmax()

        // Create a 2D input array
        val inputs = mk.ndarray(mk[mk[1.0, 2.0, 3.0], mk[4.0, 5.0, 6.0]])

        // Perform forward pass
        val output = softmax.forward(inputs)

        // Verify that the output is a valid probability distribution
        for (i in 0 until output.shape[0]) {
            val row = output[i]
            val sum = mk.math.sum(row)
            assertTrue(sum > 0.99 && sum < 1.01, "Row $i does not sum to 1 (sum = $sum)")
        }
    }

    @Test
    fun testBackward() {
        val softmax = Softmax()

        // Create a 2D input array
        val inputs = mk.ndarray(mk[mk[1.0, 2.0, 3.0], mk[4.0, 5.0, 6.0]])

        // Perform forward pass to set the internal `output` state
        softmax.forward(inputs)

        // Create a 2D gradient array (dvalues)
        val dvalues = mk.ndarray(mk[mk[0.1, 0.2, 0.3], mk[0.4, 0.5, 0.6]])

        // Perform backward pass
        val dinputs = softmax.backward(dvalues, inputs)

        // Verify that the shape of dinputs matches the shape of inputs
        assertTrue(dinputs.shape.contentEquals(inputs.shape), "Shape of dinputs does not match inputs")

        // Verify that dinputs is not all zeros (basic check for correctness)
        val isNonZero = dinputs.any { it != 0.0 }
        assertTrue(isNonZero, "dinputs should not be all zeros")
    }
}