pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "Ichi"

include(":app")
include(":core-audio")
include(":core-ui")
include(":feature-chord")
include(":feature-keyboard")
include(":feature-stems")
