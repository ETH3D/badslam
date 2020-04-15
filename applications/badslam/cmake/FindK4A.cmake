# CMake module which defines the "K4A" target if the user points
# out the location of the K4A SDK in the K4ASDK_ROOT cached variable. No
# automatic detection of the SDK install location is attempted.

if(NOT K4ASDK_ROOT OR K4ASDK_ROOT EQUAL "K4ASDK_ROOT-NOTFOUND")
  message(STATUS "K4ASDK_ROOT has not been set. If you would like to use the K4A SDK support, please set this to the root path of the K4A SDK manually.")
  set(K4ASDK_ROOT "K4ASDK_ROOT-NOTFOUND" CACHE PATH "K4A SDK root directory")
else()
  file(TO_CMAKE_PATH ${K4ASDK_ROOT} K4ASDK_ROOT)
  
  if(NOT K4ASDK_TARGET_ARCH)
    message(WARNING "K4ASDK_TARGET_ARCH is not set. Assuming amd64.")
    set(K4ASDK_TARGET_ARCH amd64)
  endif()
  
  set(K4ASDK_HEADERS ${K4ASDK_ROOT}/sdk/include)
  
  if(WIN32)
    set(K4ASDK_LIBDIR ${K4ASDK_ROOT}/sdk/windows-desktop/${K4ASDK_TARGET_ARCH}/release/lib)
    set(K4A_IMPORTLIB ${K4ASDK_LIBDIR}/k4a.lib)
    set(K4ARECORD_IMPORTLIB ${K4ASDK_LIBDIR}/k4arecord.lib)
  else()
    message(FATAL_ERROR "Unknown platform for the K4A SDK find script.")
  endif()
  
  # Define K4APrebuilt target if k4a.lib was found
  if(EXISTS ${K4A_IMPORTLIB})
    set(K4A_FOUND TRUE)
    add_library(K4APrebuilt UNKNOWN IMPORTED)
    set_target_properties(K4APrebuilt PROPERTIES
        IMPORTED_LOCATION ${K4A_IMPORTLIB}
        INTERFACE_INCLUDE_DIRECTORIES ${K4ASDK_HEADERS}
    )
  else()
    set(K4A_FOUND FALSE)
  endif()
  
  # Define K4ARecordPrebuilt target if k4a.lib was found
  if(EXISTS ${K4ARECORD_IMPORTLIB})
    set(K4ARECORD_FOUND TRUE)
    add_library(K4ARecordPrebuilt UNKNOWN IMPORTED)
    set_target_properties(K4ARecordPrebuilt PROPERTIES
        IMPORTED_LOCATION ${K4ARECORD_IMPORTLIB}
        INTERFACE_INCLUDE_DIRECTORIES ${K4ASDK_HEADERS}
    )
  else()
    set(K4ARECORD_FOUND FALSE)
  endif()
endif()
