#ifndef CAFFE_GPI_COMMUNICATOR_MODEL_HPP
#define CAFFE_GPI_COMMUNICATOR_MODEL_HPP

#include "caffe/util/GPIhelper.h"
#include "caffe/blob.hpp"

#include <vector>

namespace caffe {

class TransferForwardProducer {
public:
  TransferForwardProducer(const unsigned long buffer_size,
                          const gaspi_rank_t rank,
                          const gaspi_segment_id_t segment_id,
                          const gaspi_offset_t buffer_offset_local,
                          const gaspi_offset_t buffer_offset_remote,
                          const gaspi_notification_id_t notification_id_local,
                          const gaspi_notification_id_t notification_id_remote,
                          const gaspi_queue_id_t queue);
  void operator()(void);
  void LiftLocalStatus();
  unsigned long GetLocalAcknowledgement();
  unsigned long GetRemoteAcknowledgement();

  void status(std::ostream& s) const;


private:

  long size_;
  gaspi_rank_t rank_;
  gaspi_segment_id_t segment_id_;
  gaspi_offset_t buffer_offset_local_;
  gaspi_offset_t buffer_offset_remote_;
  gaspi_notification_id_t notification_id_local_;
  gaspi_notification_id_t notification_id_remote_;
  gaspi_queue_id_t queue_;

  unsigned long status_we_have_;
  unsigned long status_we_have_sent_;
  unsigned long status_we_acknowledge_;

  unsigned long status_acknowledged_by_remote_;
};

class TransferForwardConsumer {
public:
  TransferForwardConsumer(const gaspi_rank_t rank,
                          const gaspi_segment_id_t segment_id,
                          const gaspi_notification_id_t notification_id_local,
                          const gaspi_notification_id_t notification_id_remote,
                          const gaspi_queue_id_t queue);
  unsigned long GetStatus(void);
  void SetAcknoledgement();
  void status(std::ostream& s) const;

private:

  void ClearQueue(void);

  gaspi_rank_t rank_;
  gaspi_segment_id_t segment_id_;
  gaspi_notification_id_t notification_id_local_;
  gaspi_notification_id_t notification_id_remote_;
  gaspi_queue_id_t queue_;
  gaspi_uint queue_depth_;

  unsigned long status_;
};

template <typename Dtype>
class CommunicatorModel {
public:

  CommunicatorModel(Blob<Dtype>* blob,
                    const gaspi_segment_id_t segment_id,
                    const gaspi_queue_id_t queue_transfer,
                    const gaspi_queue_id_t queue_acknowledge,
                    const gaspi_rank_t rank,
                    const gaspi_rank_t num_ranks);
  ~CommunicatorModel();


  void operator()(void);

  void status(std::ostream& s) const;

private:
  std::vector<gaspi_rank_t> GetDataTreeWriteRanks(gaspi_rank_t rank,
                                                  gaspi_rank_t num_ranks,
                                                  int branching_factor);
  std::vector<gaspi_rank_t> GetDataTreeReadRanks(gaspi_rank_t rank,
                                                 int branching_factor);
  int GetDataTreeBranchingFactor(long num_ranks);

  static const long buffer_offset_ = 0;
  static const long notification_base_id_ = 1000;

  Blob<Dtype>* blob_;
  gaspi_segment_id_t segment_id_;
  gaspi_queue_id_t queue_send_;
  gaspi_queue_id_t queue_acknowledge_;

  std::vector<TransferForwardConsumer> consumer_;
  std::vector<TransferForwardProducer> producer_;
};

}
#endif
