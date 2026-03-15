if(NOT TARGET merlin_defs)
  add_library(merlin_defs INTERFACE)
  add_library(merlin::defs ALIAS merlin_defs)
endif()

iree_add_all_subdirs()
