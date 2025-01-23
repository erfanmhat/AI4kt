import com.vanniktech.maven.publish.SonatypeHost
import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)
}

group = "io.ai4kt.ai4kt"
version = "1.0.0"

kotlin {
    jvm()
    androidTarget {
        publishLibraryVariants("release")
        @OptIn(ExperimentalKotlinGradlePluginApi::class)
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_1_8)
        }
    }
    iosX64()
    iosArm64()
    iosSimulatorArm64()
    linuxX64()

    sourceSets {
        val commonMain by getting {
            dependencies {
                implementation("org.jetbrains.kotlinx:kotlinx-io-core:0.6.0")
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.0")
                implementation("org.jetbrains.kotlinx:multik-core:0.2.3")
                implementation("org.jetbrains.kotlinx:multik-kotlin:0.2.3")

//                implementation("org.jetbrains.kotlinx:multik-default:0.2.3")
            }
        }
        val commonTest by getting {
            dependencies {
                implementation(libs.kotlin.test)
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.7.0")
            }
        }
    }
}

android {
    namespace = "io.ai4kt.ai4kt"
    compileSdk = 34
    defaultConfig {
        minSdk = 21
    }
}

mavenPublishing {
    publishToMavenCentral(SonatypeHost.CENTRAL_PORTAL)

    signAllPublications()

    coordinates(group.toString(), "ai4kt", version.toString())

    pom {
        name = "AI4kt"
        description = "Bringing Python AI libraries to Kotlin."
        inceptionYear = "2025"
        url = "https://github.com/erfanmhat/ai4kt"
        licenses {
            license {
                name = "Apache 2.0"
                url = "https://www.apache.org/licenses/LICENSE-2.0.txt"
                distribution = "repo"
            }
        }
        developers {
            developer {
                id = "erfanmhat"
                name = "Erfan Mahdavi Athar"
                url = "https://github.com/erfanmhat"
            }
        }
        scm {
            url = "https://github.com/erfanmhat/ai4kt"
            connection = "scm:git:git://github.com/erfanmhat/ai4kt.git"
            developerConnection = "scm:git:ssh://github.com/erfanmhat/ai4kt.git"
        }
    }
}