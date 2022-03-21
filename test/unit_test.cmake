set(TEST_SOURCE_DIR ${PROJECT_SOURCE_DIR}/test/)

add_executable(test_geometry ${sources} ${TEST_SOURCE_DIR}/TestGeometry.cpp)
target_include_directories(test_geometry PUBLIC ${GTEST_INCLUDE_DIRS})
target_link_libraries(test_geometry ${GTEST_LIBRARIES} -pthread)

add_executable(test_particle ${sources} ${TEST_SOURCE_DIR}/TestParticle.cpp)
target_include_directories(test_particle PUBLIC ${GTEST_INCLUDE_DIRS})
target_link_libraries(test_particle ${GTEST_LIBRARIES} -pthread)