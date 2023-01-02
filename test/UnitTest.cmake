set(TEST_SOURCE_DIR ${PROJECT_SOURCE_DIR}/test/)

add_executable(TestGeometry ${TEST_SOURCE_DIR}/TestGeometry.cpp)
target_include_directories(TestGeometry PUBLIC ${GTEST_INCLUDE_DIRS})
target_link_libraries(TestGeometry ${GTEST_LIBRARIES} -pthread GmlsModule)

add_executable(TestParticle ${TEST_SOURCE_DIR}/TestParticle.cpp)
target_include_directories(TestParticle PUBLIC ${GTEST_INCLUDE_DIRS})
target_link_libraries(TestParticle ${GTEST_LIBRARIES} -pthread GmlsModule)

add_executable(TestLinearAlgebra ${TEST_SOURCE_DIR}/TestLinearAlgebra.cpp)
target_include_directories(TestLinearAlgebra PUBLIC ${GTEST_INCLUDE_DIRS})
target_link_libraries(TestLinearAlgebra ${GTEST_LIBRARIES} -pthread GmlsModule)

add_executable(TestPoisson ${TEST_SOURCE_DIR}/TestPoisson.cpp)
target_include_directories(TestPoisson PUBLIC ${GTEST_INCLUDE_DIRS})
target_link_libraries(TestPoisson ${GTEST_LIBRARIES} -pthread GmlsModule)

add_executable(TestTopologyOptimization ${TEST_SOURCE_DIR}/TestTopologyOptimization.cpp)
target_include_directories(TestTopologyOptimization PUBLIC ${GTEST_INCLUDE_DIRS})
target_link_libraries(TestTopologyOptimization ${GTEST_LIBRARIES} -pthread GmlsModule)

add_executable(TestStokes ${TEST_SOURCE_DIR}/TestStokes.cpp)
target_include_directories(TestStokes PUBLIC ${GTEST_INCLUDE_DIRS})
target_link_libraries(TestStokes ${GTEST_LIBRARIES} -pthread GmlsModule)

add_executable(TestDefault ${TEST_SOURCE_DIR}/TestDefault.cpp)
target_include_directories(TestDefault PUBLIC ${GTEST_INCLUDE_DIRS})
target_link_libraries(TestDefault ${GTEST_LIBRARIES} -pthread GmlsModule)