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

rootProject.name = "multiplatform-library-template"
include(":library")
