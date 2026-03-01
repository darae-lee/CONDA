set(CMAKE_CXX_STANDARD 17)

find_package(yaml-cpp REQUIRED)

add_executable(test test.cpp)
target_link_libraries(test ${PROJECT_NAME} yaml-cpp)

add_executable(generate_groundtruth generate_groundtruth.cpp)
target_link_libraries(generate_groundtruth ${PROJECT_NAME} yaml-cpp)
