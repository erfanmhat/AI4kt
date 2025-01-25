package io.ai4kt.ai4kt.fibonacci.sklearn

import io.ai4kt.ai4kt.fibonacci.tensorflow.broadcast
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*

class MinMaxScaler(private val featureRange: Pair<Double, Double> = (0.0 to 1.0)) {
    private var dataMin: D1Array<Double>? = null
    private var dataMax: D1Array<Double>? = null

    fun fit(data: D2Array<Double>) {
        dataMin = mk.math.min(data, axis = 0)
        dataMax = mk.math.max(data, axis = 0)
    }

    fun transform(data: D2Array<Double>): D2Array<Double> {
        require(dataMin != null && dataMax != null) { "The scaler has not been fitted yet." }

        val (minRange, maxRange) = featureRange
        val dataScaledToZeroAndOne: D2Array<Double> =
            (data - dataMin!!.broadcast(data.shape[0])) / (dataMax!!.broadcast(data.shape[0]) - dataMin!!.broadcast(data.shape[0]))
        return minRange + dataScaledToZeroAndOne * (maxRange - minRange)
    }

    fun fitTransform(data: D2Array<Double>): D2Array<Double> {
        fit(data)
        return transform(data)
    }
}

fun main() {
    val data = mk.d2array(3, 4) { it.toDouble() }.reshape(3, 4)
    println("Original Data:")
    println(data)

    val scaler = MinMaxScaler(featureRange = 0.0 to 1.0)
    val scaledData = scaler.fitTransform(data)

    println("\nScaled Data:")
    println(scaledData)
}