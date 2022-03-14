set(HYDROGEN_TEST_SOURCE_DIR ${PROJECT_SOURCE_DIR}/test/)

add_executable(unit_test_host ${HYDROGEN_TEST_SOURCE_DIR}/unit_test_host.cpp)
add_executable(unit_test_device ${HYDROGEN_TEST_SOURCE_DIR}/unit_test_device.cpp)
add_executable(unit_test_object ${HYDROGEN_TEST_SOURCE_DIR}/unit_test_object.cpp)
add_executable(unit_test_array ${HYDROGEN_TEST_SOURCE_DIR}/unit_test_array.cpp)

target_include_directories(unit_test_host PUBLIC ${GTEST_INCLUDE_DIRS} -pthread)
target_include_directories(unit_test_device PUBLIC ${GTEST_INCLUDE_DIRS} -pthread)
target_include_directories(unit_test_object PUBLIC ${GTEST_INCLUDE_DIRS} -pthread)
target_include_directories(unit_test_array PUBLIC ${GTEST_INCLUDE_DIRS} -pthread)

target_link_libraries(unit_test_host ${GTEST_LIBRARIES})
target_link_libraries(unit_test_device ${GTEST_LIBRARIES})
target_link_libraries(unit_test_object ${GTEST_LIBRARIES})
target_link_libraries(unit_test_array ${GTEST_LIBRARIES})

add_custom_target(unit_test)
add_dependencies(unit_test unit_test_host)
add_dependencies(unit_test unit_test_device)
add_dependencies(unit_test unit_test_object)
add_dependencies(unit_test unit_test_array)