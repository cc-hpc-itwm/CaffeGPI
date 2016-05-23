#ifndef CAFFE_GPI_RING_BUFFER_HPP
#define CAFFE_GPI_RING_BUFFER_HPP

#include "caffe/util/GPIhelper.h"


namespace caffe {

template <typename Dtype>
class RingBufferWrite {
public:
  RingBufferWrite(const unsigned long buffer_size,
                  const gaspi_segment_id_t segment_id_local,
                  const gaspi_notification_id_t notification_id_local,
                  const gaspi_offset_t buffer_offset_local,
                  const gaspi_rank_t remote_rank,
                  const gaspi_segment_id_t segment_id_remote,
                  const gaspi_notification_id_t notification_id_remote,
                  const gaspi_offset_t offset_remote,
                  const gaspi_queue_id_t queue);

  unsigned long GetFreeSpace(void);
  int Write(const Dtype* p, const unsigned long len);

private:

  void UpdateReadPointer(void);

  int error_;

  const unsigned long size_;
  unsigned long rp_;
  unsigned long wp_;
  gaspi_pointer_t buffer;

  const gaspi_segment_id_t segment_id_local_;
  const gaspi_notification_id_t notification_id_local_;
  const gaspi_offset_t buffer_offset_local_;
  const gaspi_rank_t remote_rank_;
  const gaspi_segment_id_t segment_id_remote_;
  const gaspi_notification_id_t notification_id_remote_;
  const gaspi_offset_t buffer_offset_remote_;
  const gaspi_queue_id_t queue_;
};

template <typename Dtype>
class RingBufferRead {
public:
  RingBufferRead(const unsigned long buffer_size,
                 const gaspi_segment_id_t segment_id_local,
                 const gaspi_notification_id_t notification_id_local,
                 const gaspi_offset_t buffer_offset_local,
                 const gaspi_rank_t remote_rank,
                 const gaspi_segment_id_t segment_id_remote,
                 const gaspi_notification_id_t notification_id_remote,
                 const gaspi_offset_t offset_remote,
                 const gaspi_queue_id_t queue);

  unsigned long GetNumData(void);
  int Read(Dtype* p, const unsigned long len);

private:

  void UpdateWritePointer(void);

  int error_;

  const unsigned long size_;
  unsigned long rp_;
  unsigned long wp_;
  gaspi_pointer_t buffer;

  const gaspi_segment_id_t segment_id_local_;
  const gaspi_notification_id_t notification_id_local_;
  const gaspi_offset_t buffer_offset_local_;
  const gaspi_rank_t remote_rank_;
  const gaspi_segment_id_t segment_id_remote_;
  const gaspi_notification_id_t notification_id_remote_;
  const gaspi_offset_t buffer_offset_remote_;
  const gaspi_queue_id_t queue_;
};


}

#endif
