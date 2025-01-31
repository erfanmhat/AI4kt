package tensorflow.layers

import org.jetbrains.kotlinx.multik.api.*
import kotlin.test.Test
import kotlin.test.assertEquals

class FlattenTest {
    @Test
    fun testForward() {
        // Create a 3D input array with shape (2, 3, 4)
        val input = mk.ndarray(
            mk[
                mk[
                    mk[1.0, 2.0, 3.0, 4.0],
                    mk[5.0, 6.0, 7.0, 8.0],
                    mk[9.0, 10.0, 11.0, 12.0]
                ], mk[
                    mk[13.0, 14.0, 15.0, 16.0],
                    mk[17.0, 18.0, 19.0, 20.0],
                    mk[21.0, 22.0, 23.0, 24.0]
                ]
            ]
        )
        val flatten =
            Flatten(batchSize = 2, inputShape = input.shape.toList().subList(1, input.shape.size).toIntArray())

        // Expected output shape after flattening
        val expectedOutputShape = intArrayOf(2, 12)

        // Perform the forward pass
        val output = flatten.forward(input)

        // Check the output shape
        assertEquals(expectedOutputShape.toList(), output.shape.toList())
    }

    @Test
    fun testBackward() {
        // Create a 3D input array with shape (2, 3, 4)
        val input = mk.ndarray(
            mk[
                mk[
                    mk[1.0, 2.0, 3.0, 4.0],
                    mk[5.0, 6.0, 7.0, 8.0],
                    mk[9.0, 10.0, 11.0, 12.0]
                ], mk[
                    mk[13.0, 14.0, 15.0, 16.0],
                    mk[17.0, 18.0, 19.0, 20.0],
                    mk[21.0, 22.0, 23.0, 24.0]
                ]
            ]
        )
        val flatten =
            Flatten(batchSize = 2, inputShape = input.shape.toList().subList(1, input.shape.size).toIntArray())

        // Perform the forward pass to set the input shape
        flatten.forward(input)

        // Create a 2D gradient array with shape (2, 12)
        val dvalues = mk.ndarray(
            mk[
                mk[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                mk[13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0]
            ]
        )

        // Perform the backward pass
        val output = flatten.backward(dvalues)

        // Check the output shape matches the original input shape
        assertEquals(input.shape.toList(), output.shape.toList())
    }
}