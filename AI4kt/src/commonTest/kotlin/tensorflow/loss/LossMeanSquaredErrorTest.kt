package io.ai4kt.ai4kt.fibonacci.tensorflow.loss

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.toList
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class LossMeanSquaredErrorTest {

    private val loss = LossMeanSquaredError()

    @Test
    fun testForwardPass() {
        // Create sample output and y arrays
        val output: D2Array<Double> = mk.ndarray(
            listOf(
                listOf(1.0, 2.0, 3.0),
                listOf(4.0, 5.0, 6.0)
            )
        )
        val y: D2Array<Double> = mk.ndarray(
            listOf(
                listOf(1.0, 2.0, 3.0),
                listOf(4.0, 5.0, 6.0)
            )
        )

        // Expected result: MSE should be 0.0 since output and y are identical
        val expected: D1Array<Double> = mk.ndarray(listOf(0.0, 0.0))
        val result = loss.forward(output, y)

        // Assert that the result matches the expected output
        assertTrue { result.toList().toTypedArray().contentEquals(expected.toList().toTypedArray()) }
    }

    @Test
    fun testForwardPassWithDifferentValues() {
        // Create sample output and y arrays
        val output: D2Array<Double> = mk.ndarray(
            listOf(
                listOf(1.0, 2.0, 3.0),
                listOf(4.0, 5.0, 6.0)
            )
        )
        val y: D2Array<Double> = mk.ndarray(
            listOf(
                listOf(2.0, 3.0, 4.0),
                listOf(5.0, 6.0, 7.0)
            )
        )

        // Expected result: MSE for each sample
        val expected: D1Array<Double> = mk.ndarray(listOf(1.0, 1.0))
        val result = loss.forward(output, y)

        // Assert that the result matches the expected output
        assertTrue { result.toList().toTypedArray().contentEquals(expected.toList().toTypedArray()) }
    }

    @Test
    fun testForwardPassWithInvalidShapes() {
        // Create sample output and y arrays with different shapes
        val output: D2Array<Double> = mk.ndarray(
            listOf(
                listOf(1.0, 2.0, 3.0),
                listOf(4.0, 5.0, 6.0)
            )
        )
        val y: D2Array<Double> = mk.ndarray(
            listOf(
                listOf(1.0, 2.0),
                listOf(3.0, 4.0)
            )
        )

        // Assert that an IllegalArgumentException is thrown
        assertFailsWith<IllegalArgumentException> {
            loss.forward(output, y)
        }
    }

    @Test
    fun testBackwardPass() {
        // Create sample output and yTrue arrays
        val output: D2Array<Double> = mk.ndarray(
            listOf(
                listOf(1.0, 2.0, 3.0),
                listOf(4.0, 5.0, 6.0)
            )
        )
        val yTrue: D2Array<Double> = mk.ndarray(
            listOf(
                listOf(2.0, 3.0, 4.0),
                listOf(5.0, 6.0, 7.0)
            )
        )

        // Expected gradient
        val expected: D2Array<Double> = mk.ndarray(
            listOf(
                listOf(-1.0, -1.0, -1.0),
                listOf(-1.0, -1.0, -1.0)
            )
        )
        val result = loss.backward(output, yTrue)

        // Assert that the result matches the expected gradient
        for (i in 0 until result.shape[0]) {
            for (j in 0 until result.shape[1]) {
                assertTrue(
                    kotlin.math.abs(result[i, j] - expected[i, j]) < 1e-6,
                    "Mismatch at index ($i, $j): expected ${expected[i, j]}, got ${result[i, j]}"
                )
            }
        }
    }

    @Test
    fun testBackwardPassWithInvalidShapes() {
        // Create sample output and yTrue arrays with different shapes
        val output: D2Array<Double> = mk.ndarray(
            listOf(
                listOf(1.0, 2.0, 3.0),
                listOf(4.0, 5.0, 6.0)
            )
        )
        val yTrue: D2Array<Double> = mk.ndarray(
            listOf(
                listOf(1.0, 2.0),
                listOf(3.0, 4.0)
            )
        )

        // Assert that an IllegalArgumentException is thrown
        assertFailsWith<IllegalArgumentException> {
            loss.backward(output, yTrue)
        }
    }
}