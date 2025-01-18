pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositories {
//        google()
//        mavenCentral()
        maven("https://en-mirror.ir")
    }
}

rootProject.name = "AI4kt"
include(":library")
