list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

set(IREE_ITA_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")

# Only include merlin samples if the expected subdirectories are present.
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/samples/CMakeLists.txt"
   AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/samples/custom_dispatch_ukernels_saturn/dummy_kernel")
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/samples merlin-samples)
endif()

#add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/ukernels merlin-ukernels)