
add_executable(task task.cpp)

if (NOT MSVC)
 target_link_libraries(task PRIVATE nvcomp)
 find_package(CUDAToolkit REQUIRED)
 target_link_libraries(task PRIVATE CUDA::cudart)
 set(THREADS_PREFER_PTHREAD_FLAG ON)
 find_package(Threads REQUIRED)
 target_link_libraries(task PRIVATE Threads::Threads)
else()
 message(FATAL_ERROR "No implement a build for the MSVC environment")
endif()
