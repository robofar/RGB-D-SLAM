cmake_minimum_required(VERSION 3.14)
project(fSLAM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include(FetchContent)
FetchContent_Declare(
    sophus
    GIT_REPOSITORY https://github.com/strasdat/Sophus.git
    GIT_TAG 1.22.10
)
set(SOPHUS_USE_BASIC_LOGGING ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(sophus)

FetchContent_Declare(
    gtsam
    GIT_REPOSITORY https://github.com/borglab/gtsam.git
    GIT_TAG 4.2
)
set(GTSAM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GTSAM_BUILD_EXAMPLES_ALWAYS OFF CACHE BOOL "" FORCE)
set(GTSAM_BUILD_UNSTABLE OFF CACHE BOOL "" FORCE)
set(GTSAM_WITH_TBB OFF CACHE BOOL "" FORCE)
set(GTSAM_BUILD_WITH_MARCH_NATIVE OFF CACHE BOOL "" FORCE)
set(GTSAM_USE_SYSTEM_EIGEN ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(gtsam)

FetchContent_Declare(rerun_sdk URL
    https://github.com/rerun-io/rerun/releases/latest/download/rerun_cpp_sdk.zip)
FetchContent_MakeAvailable(rerun_sdk)

add_executable(fslam
    src/main.cpp
    src/TUMDataset.cpp
    src/ReplicaDataset.cpp
    src/FeatureManager.cpp
    src/MapPoint.cpp
    src/Map.cpp
    src/KeyFrameDatabase.cpp
    src/Tracking.cpp
    src/PoseEstimator.cpp
    src/Utils.cpp
    src/Visualizer.cpp
    src/LocalBA.cpp
)

target_include_directories(fslam PRIVATE
    include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(fslam PRIVATE
    ${OpenCV_LIBS}
    Eigen3::Eigen
    Sophus::Sophus
    gtsam
    rerun_sdk
)
