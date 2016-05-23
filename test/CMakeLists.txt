
set(testSources
   runGPIRingBuffer.cpp
   ../src/caffe/gpi_ring_buffer.cpp
   )

add_executable(runGPIRingBuffer ${testSources})
target_link_libraries(runGPIRingBuffer ${GPI2_GPI_LIBRARIES})

