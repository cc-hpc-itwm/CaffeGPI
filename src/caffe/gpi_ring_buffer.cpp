#include "caffe/gpi_ring_buffer.hpp"

#include <limits>
#include <string.h>

namespace caffe {

template <typename Dtype>
RingBufferWrite<Dtype>::RingBufferWrite(const unsigned long buffer_size,
                                        const gaspi_segment_id_t segment_id_local,
                                        const gaspi_notification_id_t notification_id_local,
                                        const gaspi_offset_t buffer_offset_local,
                                        const gaspi_rank_t remote_rank,
                                        const gaspi_segment_id_t segment_id_remote,
                                        const gaspi_notification_id_t notification_id_remote,
                                        const gaspi_offset_t offset_remote,
                                        const gaspi_queue_id_t queue)
  : error_(0),
    size_(buffer_size),
    rp_(0),
    wp_(0),
    segment_id_local_(segment_id_local),
    notification_id_local_(notification_id_local),
    buffer_offset_local_(buffer_offset_local),
    remote_rank_(remote_rank),
    segment_id_remote_(segment_id_remote),
    notification_id_remote_(notification_id_remote),
    buffer_offset_remote_(offset_remote),
    queue_(queue) {

    if (size_ > std::numeric_limits<gaspi_notification_t>::max() )
      error_ = -1; //max size is one smaller than capacity of gaspi_notification_t

    SUCCESS_OR_DIE(gaspi_segment_ptr(segment_id_local, &buffer));
    buffer = ((char*)buffer) + buffer_offset_local;

    gaspi_config_t config;
    SUCCESS_OR_DIE(gaspi_config_get(&config));
    queue_depth_ = config.queue_depth;
  }

template <typename Dtype>
unsigned long RingBufferWrite<Dtype>::GetFreeSpace(void) {
  UpdateReadPointer();
  if (rp_ <= wp_) {
    return size_ + rp_ - wp_ - 1;
  } else {
    return rp_ - wp_ - 1;
  }
}

template <typename Dtype>
void RingBufferWrite<Dtype>::UpdateReadPointer(void) {
  gaspi_notification_t f;
  SUCCESS_OR_DIE(gaspi_notify_reset(segment_id_local_, notification_id_local_,  &f));
  if (f > 0)
    rp_ = f-1; //zero is not allowed as a notification
//  gaspi_printf("rp: %lu\n", rp_);
}

template <typename Dtype>
int RingBufferWrite<Dtype>::Write(const Dtype* p,
                                  const unsigned long len) {
  if (len > GetFreeSpace()) return -1;
  ClearQueue();

  if (rp_ <= wp_) {
    //first chunk
    const long chunk = std::min(len, size_ - wp_);
    const long rest = len - chunk;
    memcpy(((char*)buffer) + wp_ * sizeof(Dtype),
           p,
           chunk * sizeof(Dtype));

    if (rest) {
      SUCCESS_OR_DIE(gaspi_write(segment_id_local_,
                                 buffer_offset_local_ +  wp_ * sizeof(Dtype),
                                 remote_rank_,
                                 segment_id_remote_,
                                 buffer_offset_remote_ + wp_ * sizeof(Dtype),
                                 chunk * sizeof(Dtype),
                                 queue_,
                                 GASPI_BLOCK));
      wp_ = 0;
      //second chunk
      memcpy(((char*)buffer) + wp_ * sizeof(Dtype),
             ((char*)p) + chunk * sizeof(Dtype),
             rest * sizeof(Dtype));
      SUCCESS_OR_DIE(gaspi_write_notify(segment_id_local_,
                                        buffer_offset_local_ +  wp_ * sizeof(Dtype),
                                        remote_rank_,
                                        segment_id_remote_,
                                        buffer_offset_remote_ + wp_ * sizeof(Dtype),
                                        rest * sizeof(Dtype),
                                        notification_id_remote_,
                                        wp_ + rest + 1,//zero is not allowed as notification
                                        queue_,
                                        GASPI_BLOCK));
      wp_ += rest;
    } else {
      const unsigned long wpnew = (wp_ + chunk) % size_;
      SUCCESS_OR_DIE(gaspi_write_notify(segment_id_local_,
                                        buffer_offset_local_ +  wp_ * sizeof(Dtype),
                                        remote_rank_,
                                        segment_id_remote_,
                                        buffer_offset_remote_ + wp_ * sizeof(Dtype),
                                        chunk * sizeof(Dtype),
                                        notification_id_remote_,
                                        wpnew + 1,//zero is not allowed as notification
                                        queue_,
                                        GASPI_BLOCK));
      wp_ = wpnew;
    }
  } else {
    memcpy(((char*)buffer) + wp_ * sizeof(Dtype),
           p,
           len * sizeof(Dtype));
    SUCCESS_OR_DIE(gaspi_write_notify(segment_id_local_,
                                      buffer_offset_local_ +  wp_ * sizeof(Dtype),
                                      remote_rank_,
                                      segment_id_remote_,
                                      buffer_offset_remote_ + wp_ * sizeof(Dtype),
                                      len * sizeof(Dtype),
                                      notification_id_remote_,
                                      wp_ + len + 1,//zero is not allowed as notification
                                      queue_,
                                      GASPI_BLOCK));
    wp_ += len;
  }
//  gaspi_printf("wp: %lu\n", wp_);
//  {
//    gaspi_number_t entries;
//    SUCCESS_OR_DIE(gaspi_queue_size(queue_, &entries));
//    gaspi_printf("queue %lu\n", entries);
//  }
  return 0;
}

template <typename Dtype>
void RingBufferWrite<Dtype>::ClearQueue(void) {
  gaspi_number_t entries;
  SUCCESS_OR_DIE(gaspi_queue_size(queue_, &entries));

  if ((long(queue_depth_) - long(entries)) < 3) {
    SUCCESS_OR_DIE(gaspi_wait(queue_, GASPI_BLOCK));
  }
}

//----------------------------------------------------------------------------

template <typename Dtype>
RingBufferRead<Dtype>::RingBufferRead(const unsigned long buffer_size,
                                      const gaspi_segment_id_t segment_id_local,
                                      const gaspi_notification_id_t notification_id_local,
                                      const gaspi_offset_t buffer_offset_local,
                                      const gaspi_rank_t remote_rank,
                                      const gaspi_segment_id_t segment_id_remote,
                                      const gaspi_notification_id_t notification_id_remote,
                                      const gaspi_offset_t offset_remote,
                                      const gaspi_queue_id_t queue)
  : error_(0),
    size_(buffer_size),
    rp_(0),
    wp_(0),
    segment_id_local_(segment_id_local),
    notification_id_local_(notification_id_local),
    buffer_offset_local_(buffer_offset_local),
    remote_rank_(remote_rank),
    segment_id_remote_(segment_id_remote),
    notification_id_remote_(notification_id_remote),
    buffer_offset_remote_(offset_remote),
    queue_(queue) {

    if (size_ > std::numeric_limits<gaspi_notification_t>::max() )
      error_ = -1; //max size is one smaller than capacity of gaspi_notification_t

    SUCCESS_OR_DIE(gaspi_segment_ptr(segment_id_local, &buffer));
    buffer = ((char*)buffer) + buffer_offset_local;

    gaspi_config_t config;
    SUCCESS_OR_DIE(gaspi_config_get(&config));
    queue_depth_ = config.queue_depth;
  }

template <typename Dtype>
unsigned long RingBufferRead<Dtype>::GetNumData(void) {
  UpdateWritePointer();
  if (rp_ <= wp_) {
    return wp_ - rp_;
  } else {
    return size_ - rp_ + wp_;
  }
}

template <typename Dtype>
void RingBufferRead<Dtype>::UpdateWritePointer(void) {
  gaspi_notification_t f;
  SUCCESS_OR_DIE(gaspi_notify_reset(segment_id_local_, notification_id_local_,  &f));
  if (f > 0)
    wp_ = f-1; //zero is not allowed as a notification
//  gaspi_printf("wp: %lu\n", wp_);
}

template <typename Dtype>
int RingBufferRead<Dtype>::Add(Dtype* p,
                               const unsigned long len) {
  static long counter = 0;
  if (len > GetNumData()) return -1;
  if (rp_ <= wp_) {
    const Dtype* s = ((Dtype*) buffer) + rp_;
    for (long i=0; i<len; i++)
      p[i] += s[i];
    rp_ += len;
  } else {
    const unsigned long chunk = std::min(len, size_ - rp_);
    const unsigned long rest = len - chunk;
    const Dtype* s = ((Dtype*)buffer) + rp_;
    for (long i=0; i<chunk; i++)
      p[i] += s[i];
    rp_ += chunk;
    rp_ = rp_ % size_;

    if (rest > 0) {
      const Dtype* s = ((Dtype*)buffer) + rp_;
      Dtype* const d = p + chunk;
      for (long i=0; i<rest; i++)
        p[i] += s[i];
      rp_ += rest;
    }
  }
//  {
//    gaspi_number_t entries;
//    SUCCESS_OR_DIE(gaspi_queue_size(queue_, &entries));
//    gaspi_printf("queue %lu\n", entries);
//  }

  ClearQueue();
  SUCCESS_OR_DIE(gaspi_notify(segment_id_remote_, remote_rank_, notification_id_remote_,
                              rp_ + 1, queue_, GASPI_BLOCK));
//  gaspi_printf("rp: %lu\n", rp_);
  return 0;
}

template <typename Dtype>
int RingBufferRead<Dtype>::Read(Dtype* p,
                                const unsigned long len) {
  static long counter = 0;
  if (len > GetNumData()) return -1;
  if (rp_ <= wp_) {
    memcpy(p,
           ((char*)buffer) + rp_ * sizeof(Dtype),
           len * sizeof(Dtype));
    rp_ += len;
  } else {
    const unsigned long chunk = std::min(len, size_ - rp_);
    const unsigned long rest = len - chunk;

    memcpy(p,
           ((char*)buffer) + rp_ * sizeof(Dtype),
           chunk * sizeof(Dtype));
    rp_ += chunk;
    rp_ = rp_ % size_;

    if (rest > 0) {
      memcpy(((char*)p) + chunk * sizeof(Dtype),
             ((char*)buffer) + rp_ * sizeof(Dtype),
             rest * sizeof(Dtype));
      rp_ += rest;
    }
  }
//  {
//    gaspi_number_t entries;
//    SUCCESS_OR_DIE(gaspi_queue_size(queue_, &entries));
//    gaspi_printf("queue %lu\n", entries);
//  }

  ClearQueue();
  SUCCESS_OR_DIE(gaspi_notify(segment_id_remote_, remote_rank_, notification_id_remote_,
                              rp_ + 1, queue_, GASPI_BLOCK));
//  gaspi_printf("rp: %lu\n", rp_);
  return 0;
}

template <typename Dtype>
void RingBufferRead<Dtype>::ClearQueue(void) {
  gaspi_number_t entries;
  SUCCESS_OR_DIE(gaspi_queue_size(queue_, &entries));

  if ((long(queue_depth_) - long(entries)) < 1) {
    SUCCESS_OR_DIE(gaspi_wait(queue_, GASPI_BLOCK));
  }
}

template class RingBufferWrite<float>;
template class RingBufferRead<float>;
template class RingBufferWrite<double>;
template class RingBufferRead<double>;
template class RingBufferWrite<int>;
template class RingBufferRead<int>;
}
