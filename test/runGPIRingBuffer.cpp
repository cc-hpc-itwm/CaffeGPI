#include "caffe/gpi_ring_buffer.hpp"
#include "caffe/util/GPIhelper.h"
#include <iostream>
#include <stdlib.h>
#include <unistd.h>
//#include <GASPI_Ext.h>


//void quick() {
//
//  SUCCESS_OR_DIE(gaspi_proc_init(GASPI_BLOCK));
//
//  gaspi_config_t config;
//  SUCCESS_OR_DIE(gaspi_config_get (&config));
//  std::cout << config.queue_depth << std::endl;
//
//
//  gaspi_rank_t nRanks;
//  gaspi_rank_t rank;
//
//  SUCCESS_OR_DIE(gaspi_proc_num (&nRanks));
//  SUCCESS_OR_DIE(gaspi_proc_rank (&rank));
//
//  srand(17);
//  const long bufferOffsetA = 20;
//  const long bufferOffsetB = 56;
//  const long bufferSize = 10000;
//  const long chunkMax = 1000;
//  const long nData = 0x100000;
//  typedef int DType;
//
//  DType* bite = new DType[chunkMax];
//
//  const gaspi_segment_id_t segmentA = 5;
//  const gaspi_segment_id_t segmentB = 7;
//
//  SUCCESS_OR_DIE(gaspi_segment_create(segmentA,
//                                      bufferSize * sizeof(DType) + bufferOffsetA,
//                                      GASPI_GROUP_ALL,
//                                      GASPI_BLOCK,
//                                      GASPI_MEM_UNINITIALIZED));
//  SUCCESS_OR_DIE(gaspi_segment_create(segmentB,
//                                      bufferSize * sizeof(DType) + bufferOffsetB,
//                                      GASPI_GROUP_ALL,
//                                      GASPI_BLOCK,
//                                      GASPI_MEM_UNINITIALIZED));
//
//  if (nRanks == 2) {
//    if (rank == 0) {
//      const gaspi_segment_id_t segment_id_local = segmentA;
//      gaspi_pointer_t ptr;
//      gaspi_segment_ptr (segment_id_local, &ptr);
//      DType* buffer = (int*) &(((char*)ptr)[bufferOffsetA]);
//      const gaspi_segment_id_t segment_id_remote = segmentB;
//      gaspi_notification_id_t notification_id_local = 17;
//      gaspi_notification_id_t notification_id_remote = 1111;
//      const gaspi_queue_id_t queue = 1;
//
//      long counter = 0;
//      while (counter < nData) {
//        long chunk = std::max(1l, long((rand() / double(RAND_MAX)) * chunkMax));
//        chunk = std::min(chunk, nData - counter);
//        for (long i=0; i<chunk; i++) buffer[i] = ++counter;
//        SUCCESS_OR_DIE(gaspi_write(segment_id_local, bufferOffsetA, 1, segment_id_remote, bufferOffsetB, chunk * sizeof(DType), queue, GASPI_BLOCK));
//        gaspi_number_t entries;
//        SUCCESS_OR_DIE(gaspi_queue_size(queue, &entries));
//        gaspi_printf("queue %lu\n", entries);
//        usleep(100000);
//      }
//
//    } else if (rank == 1) {
//      const gaspi_segment_id_t segment_id_local = segmentB;
//      const gaspi_segment_id_t segment_id_remote = segmentA;
//      const gaspi_notification_id_t notification_id_local = 1111;
//      const gaspi_notification_id_t notification_id_remote = 17;
//      const gaspi_queue_id_t queue = 2;
//
//      long counter = 0;
//      while (counter < nData) {
//        long chunk = 51;
//        chunk = std::min(chunk, nData - counter);
//        SUCCESS_OR_DIE(gaspi_write(segment_id_local, 1000, 0, segment_id_remote, 1000, chunk, queue, GASPI_BLOCK));
//        gaspi_number_t entries;
//        SUCCESS_OR_DIE(gaspi_queue_size(queue, &entries));
//        gaspi_printf("queue %lu\n", entries);
//        usleep(100000);
//      }
//    }
//  } else {
//    std::cout << "Exactly two ranks are necessary for this test." << std::endl;
//  }
//
//  delete[] bite;
//
//  SUCCESS_OR_DIE(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
//
//  SUCCESS_OR_DIE(gaspi_proc_term (GASPI_BLOCK));
//}



int main() {

//  quick();
//  return 0;

  SUCCESS_OR_DIE(gaspi_proc_init(GASPI_BLOCK));

  gaspi_config_t config;
  SUCCESS_OR_DIE(gaspi_config_get (&config));
  std::cout << config.queue_size_max << std::endl;


  gaspi_rank_t nRanks;
  gaspi_rank_t rank;

  SUCCESS_OR_DIE(gaspi_proc_num (&nRanks));
  SUCCESS_OR_DIE(gaspi_proc_rank (&rank));

  srand(17);
  const long bufferOffsetA = 20;
  const long bufferOffsetB = 76;
  const long bufferSize = 10000;
  const long chunkMax = 100;
  const long nData = 0x10000000;
  typedef double Dtype;

  Dtype* bite = new Dtype[chunkMax];

  const gaspi_segment_id_t segmentA = 5;
  const gaspi_segment_id_t segmentB = 7;

  SUCCESS_OR_DIE(gaspi_segment_create(segmentA,
                                      bufferSize * sizeof(Dtype) + bufferOffsetA,
                                      GASPI_GROUP_ALL,
                                      GASPI_BLOCK,
                                      GASPI_MEM_UNINITIALIZED));
  SUCCESS_OR_DIE(gaspi_segment_create(segmentB,
                                      bufferSize * sizeof(Dtype) + bufferOffsetB,
                                      GASPI_GROUP_ALL,
                                      GASPI_BLOCK,
                                      GASPI_MEM_UNINITIALIZED));

  if (nRanks == 2) {
    if (rank == 0) {
      const gaspi_segment_id_t segment_id_local = segmentA;
      const gaspi_segment_id_t segment_id_remote = segmentB;
      gaspi_notification_id_t notification_id_local = 17;
      gaspi_notification_id_t notification_id_remote = 1111;
      const gaspi_queue_id_t queue = 1;

      caffe::RingBufferWrite<Dtype> buffer(bufferSize, segment_id_local, notification_id_local,
                                           bufferOffsetA, 1, segment_id_remote,
                                           notification_id_remote, bufferOffsetB,
                                           queue);

      long counter = 0;
      while (counter < nData) {
        long chunk = std::min(std::max(1l, long((rand() / double(RAND_MAX)) * chunkMax)), chunkMax);
        chunk = std::min(chunk, nData - counter);
        for (long i=0; i<chunk; i++) bite[i] = ++counter;
        while (buffer.GetFreeSpace() < chunk) usleep(1);
        buffer.Write(bite, chunk);
//        usleep(1);
      }

    } else if (rank == 1) {
      const gaspi_segment_id_t segment_id_local = segmentB;
      const gaspi_segment_id_t segment_id_remote = segmentA;
      const gaspi_notification_id_t notification_id_local = 1111;
      const gaspi_notification_id_t notification_id_remote = 17;
      const gaspi_queue_id_t queue = 2;

      caffe::RingBufferRead<Dtype> buffer(bufferSize, segment_id_local, notification_id_local,
                                          bufferOffsetB, 0, segment_id_remote,
                                          notification_id_remote, bufferOffsetA,
                                          queue);

      long counter = 0;
      while (counter < nData) {
        long chunk = 55;
        chunk = std::min(chunk, nData - counter);
        while (buffer.GetNumData() < chunk) usleep(1);
        buffer.Read(bite, chunk);
        int check = 0;
        for (long i=0; i<chunk; i++) {
          check |= (bite[i] != ++counter);
        }
        if (check) std::cout
          << "Error in Block starting with " << counter-chunk << std::endl;
//        printf("%lu\n", counter);
        //usleep(1);
      }
    }
  } else {
    std::cout << "Exactly two ranks are necessary for this test." << std::endl;
  }

  delete[] bite;

  SUCCESS_OR_DIE(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  SUCCESS_OR_DIE(gaspi_proc_term (GASPI_BLOCK));
  return 0;
}
