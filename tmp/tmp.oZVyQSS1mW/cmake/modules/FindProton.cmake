set(PROTON_SEARCH_PATH
    ../proton/)

find_path(PROTON_INCLUDE_PATH proton.hpp
    HINTS
    PATHS ${PROTON_SEARCH_PATH}/include)

include(FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(Proton REQUIRED_VARS PROTON_INCLUDE_PATH)