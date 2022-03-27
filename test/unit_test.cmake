set(TEST_SOURCE_DIR ${PROJECT_SOURCE_DIR}/test/)

add_executable(test_geometry ${TEST_SOURCE_DIR}/TestGeometry.cpp)
target_include_directories(test_geometry PUBLIC ${GTEST_INCLUDE_DIRS})
target_link_libraries(test_geometry ${GTEST_LIBRARIES} -pthread gmls_module)

add_executable(test_particle ${TEST_SOURCE_DIR}/TestParticle.cpp)
target_include_directories(test_particle PUBLIC ${GTEST_INCLUDE_DIRS})
target_link_libraries(test_particle ${GTEST_LIBRARIES} -pthread gmls_module)

add_executable(test_poisson ${TEST_SOURCE_DIR}/TestPoisson.cpp)
target_include_directories(test_poisson PUBLIC ${GTEST_INCLUDE_DIRS})
target_link_libraries(test_poisson ${GTEST_LIBRARIES} -pthread gmls_module)