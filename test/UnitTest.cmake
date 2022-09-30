set(TEST_SOURCE_DIR ${PROJECT_SOURCE_DIR}/test/)

add_executable(TestGeometry ${TEST_SOURCE_DIR}/TestGeometry.cpp)
target_include_directories(TestGeometry PUBLIC ${GTEST_INCLUDE_DIRS})
target_link_libraries(TestGeometry ${GTEST_LIBRARIES} -pthread GmlsModule)

add_executable(TestLinearAlgebra ${TEST_SOURCE_DIR}/TestLinearAlgebra.cpp)
target_include_directories(TestLinearAlgebra PUBLIC ${GTEST_INCLUDE_DIRS})
target_link_libraries(TestLinearAlgebra ${GTEST_LIBRARIES} -pthread GmlsModule)