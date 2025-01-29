actual val firstElement: Int = 1
actual val secondElement: Int = 1

fun main(){
    for(i in generateFibi()){
        println(i)
    }
}