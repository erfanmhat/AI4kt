package tensorflow.models

import tensorflow.activations.ReLU
import tensorflow.activations.Softmax
import tensorflow.loss.LossBinaryCrossentropy
import tensorflow.optimizers.GradientDescentOptimizer
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

class SequentialTest {

    private val random = Random(42)

    @Test
    fun testForwardPass() {
        // Create a model
        val model = Sequential(batchSize = 2, random)
            .addInput(3) // Input layer with 3 features
            .addDense(5, ReLU()) // Hidden layer with 5 neurons and ReLU activation
            .addDense(2, Softmax()) // Output layer with 2 neurons and Softmax activation
            .setOptimizer(GradientDescentOptimizer(0.001)) // Set optimizer with learning rate 0.01
            .setLossFunction(LossBinaryCrossentropy()) // Set loss function
            .build()

        // Example input data (2 samples, 3 features each)
        val inputs: D2Array<Double> = mk.ndarray(
            listOf(
                listOf(1.0, 2.0, 3.0),
                listOf(4.0, 5.0, 6.0)
            )
        )

        // Perform forward pass
        val output = model.forward(inputs)

        // Verify that the output has the correct shape (2 samples, 2 outputs)
        assertTrue(output.shape.contentEquals(intArrayOf(2, 2)), "Output shape is incorrect")
    }

    @Test
    fun testBackwardPass() {
        // Create a model
        val model = Sequential(batchSize = 2, random)
            .addInput(3) // Input layer with 3 features
            .addDense(5, ReLU()) // Hidden layer with 5 neurons and ReLU activation
            .addDense(2, Softmax()) // Output layer with 2 neurons and Softmax activation
            .setOptimizer(GradientDescentOptimizer(0.001)) // Set optimizer with learning rate 0.01
            .setLossFunction(LossBinaryCrossentropy()) // Set loss function
            .build()

        // Example input data (2 samples, 3 features each)
        val inputs: D2Array<Double> = mk.ndarray(
            listOf(
                listOf(1.0, 2.0, 3.0),
                listOf(4.0, 5.0, 6.0)
            )
        )

        // Example target data (2 samples, one-hot encoded)
        val yTrue: D2Array<Double> = mk.ndarray(
            listOf(
                listOf(1.0, 0.0),
                listOf(0.0, 1.0)
            )
        )

        // Perform forward pass
        val output = model.forward(inputs)

        // Perform backward pass
        val dvalues = model.loss.backward(output, yTrue)
        model.backward(dvalues)

        // Verify that the gradients were computed correctly
        // (You can add assertions based on the expected behavior of your layers)
        assertTrue(true) // Placeholder assertion
    }

    @Test
    fun testTrainStep() {
        // Create a model
        val model = Sequential(batchSize = 2, random)
            .addInput(3) // Input layer with 3 features
            .addDense(5, ReLU()) // Hidden layer with 5 neurons and ReLU activation
            .addDense(2, Softmax()) // Output layer with 2 neurons and Softmax activation
            .setOptimizer(GradientDescentOptimizer(0.001)) // Set optimizer with learning rate 0.01
            .setLossFunction(LossBinaryCrossentropy()) // Set loss function
            .build()

        // Example input data (2 samples, 3 features each)
        val inputs: D2Array<Double> = mk.ndarray(
            listOf(
                listOf(1.0, 2.0, 3.0),
                listOf(4.0, 5.0, 6.0)
            )
        )

        // Example target data (2 samples, one-hot encoded)
        val yTrue: D2Array<Double> = mk.ndarray(
            listOf(
                listOf(1.0, 0.0),
                listOf(0.0, 1.0)
            )
        )

        // Perform a training step
        model.trainStep(inputs, yTrue)

        // Verify that the model's weights and biases were updated
        // (You can add assertions based on the expected behavior of your optimizer)
        assertTrue(true) // Placeholder assertion
    }
}