set(TEST_SOURCE_DIR ${PROJECT_SOURCE_DIR}/test/)

add_executable(performance_geometry ${sources} ${TEST_SOURCE_DIR}/PerformanceGeometry.cpp)
target_include_directories(performance_geometry PUBLIC ${GTEST_INCLUDE_DIRS})
target_link_libraries(performance_geometry ${GTEST_LIBRARIES} -pthread)