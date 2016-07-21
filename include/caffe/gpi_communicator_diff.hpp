#ifndef CAFFE_GPI_COMMUNICATOR_DIFF_HPP
#define CAFFE_GPI_COMMUNICATOR_DIFF_HPP

#include "caffe/util/GPIhelper.h"
#include "caffe/blob.hpp"
#include "gpi_ring_buffer.hpp"

#include <vector>

namespace caffe {

template <typename Dtype>
class CommunicatorDiff {
public:
  CommunicatorDiff(const long buffer_size,
                   const gaspi_notification_id_t notification_id_base_,
                   const gaspi_segment_id_t segment_id,
                   const gaspi_queue_id_t queue,
                   const gaspi_rank_t rank,
                   const gaspi_rank_t num_ranks);
  ~CommunicatorDiff();

  void operator()(void);
  bool CommunicateLayerDiffFinished(void);
  bool CommunicateLayerDiffReadFinished(int index);
  void AddCalculatedBlob(Blob<Dtype>* blob);
  void ResetCommunicationStatus(void);

private:
  int GetDiffTreeBranchingFactor() const;
  std::vector<gaspi_rank_t> GetDiffTreeWriteRanks(gaspi_rank_t rank,
                                                  int branching_factor) const;
  std::vector<gaspi_rank_t> GetDiffTreeReadRanks(gaspi_rank_t rank,
                                                 int branching_factor) const;

  gaspi_segment_id_t segment_id_;
  gaspi_rank_t rank_;
  gaspi_rank_t num_ranks_;

  vector<RingBufferRead<Dtype> > com_buffers_diff_read_;
  vector<RingBufferWrite<Dtype> > com_buffers_diff_write_;
  vector<int> com_buffers_diff_read_status_;
  vector<int> com_buffers_diff_write_status_;
  vector<Blob<Dtype>* > calculated_blobs_;
};

}
#endif
