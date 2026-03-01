set(CMAKE_CXX_STANDARD 17)

file(GLOB CPP_SOURCES *.cpp)
list(REMOVE_ITEM CPP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/conda_internals.cpp")

add_library(${PROJECT_NAME} ${CPP_SOURCES})
add_library(${PROJECT_NAME}_s STATIC ${CPP_SOURCES})

target_link_libraries(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/lib/libconda_internals.a)
target_link_libraries(${PROJECT_NAME}_s ${PROJECT_SOURCE_DIR}/lib/libconda_internals.a)
