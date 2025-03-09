buildscript {
    val kotlinVersion = "1.7.10" // Define as a variable for consistency
    repositories {
        google()
        mavenCentral()
       
    }
    dependencies {
        classpath("com.android.tools.build:gradle:7.3.1")
        classpath("com.chaquo.python:gradle:16.0.0")
        classpath("org.jetbrains.kotlin:kotlin-gradle-plugin:$kotlinVersion")
        
    }
}
allprojects {
    repositories {
        google()
        mavenCentral()
       
    }
}
val newBuildDir: Directory = rootProject.layout.buildDirectory.dir("../../build").get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}



tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}

