package io.ai4kt.ai4kt.fibonacci.tensorflow.layers

import io.ai4kt.ai4kt.fibonacci.tensorflow.plusD1Array
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.random.Random

class DNNLayer(n_inputs: Int, n_neurons: Int) {
    var weights: D2Array<Double>
    var biases: D1Array<Double>

    init {
        weights = mk.ndarray(List(n_inputs) { List(n_neurons) { Random.nextDouble(-0.1, 0.1) } })
        biases = mk.zeros<Double>(n_neurons)
    }

    fun forward(inputs: D2Array<Double>): D2Array<Double> {
        return mk.linalg.dot(inputs, weights).plusD1Array(biases)

    }
}


fun main() {
    val X = mk.ndarray(
        mk[
            mk[1.0, 2.0, 3.0, 2.5],
            mk[2.0, 5.0, -1.0, 2.0],
            mk[-1.5, 2.7, 3.3, -0.8]
        ]
    )
    val l1 = DNNLayer(4, 5)
    val l2 = DNNLayer(5, 2)
    val l1_out = l1.forward(X)
    val l2_out = l2.forward(l1_out)
    println(l1_out)
    println()
    println(l2_out)
}