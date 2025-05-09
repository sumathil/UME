
option(USE_KOKKOS "Enable support for Kokkos code")

if(USE_KOKKOS)

  find_package(Kokkos REQUIRED)
  find_package(MPI REQUIRED)

  set(COMMON_LINK_LIBRARIES MPI::MPI_CXX Kokkos::kokkos CUDA::cudart)
  set(COMMON_COMPILE_DEFINITIONS
    ${COMMON_COMPILE_DEFINITIONS}
    HAVE_MPI
    )
endif()
