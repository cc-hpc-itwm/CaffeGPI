#include "caffe/gpi_communicator_model.hpp"
#include <algorithm>

namespace caffe {

TransferForwardProducer::TransferForwardProducer(
  const unsigned long buffer_size,
  const gaspi_rank_t rank,
  const gaspi_segment_id_t segment_id,
  const gaspi_offset_t buffer_offset_local,
  const gaspi_offset_t buffer_offset_remote,
  const gaspi_notification_id_t notification_id_local,
  const gaspi_notification_id_t notification_id_remote,
  const gaspi_queue_id_t queue)
: size_(buffer_size),
  rank_(rank),
  segment_id_(segment_id),
  buffer_offset_local_(buffer_offset_local),
  buffer_offset_remote_(buffer_offset_remote),
  notification_id_local_(notification_id_local),
  notification_id_remote_(notification_id_remote),
  queue_(queue),
  status_we_have_(0),
  status_we_have_sent_(0),
  status_we_acknowledge_(0),
  status_acknowledged_by_remote_(0){

}

void TransferForwardProducer::operator()(void) {
  GetRemoteAcknowledgement();
  GetLocalAcknowledgement();

  if ((status_we_have_ > status_we_have_sent_)
      && ((status_we_have_ <= status_acknowledged_by_remote_))) {
    SUCCESS_OR_DIE(gaspi_write_notify(segment_id_,
                                      buffer_offset_local_,
                                      rank_,
                                      segment_id_,
                                      buffer_offset_remote_,
                                      size_,
                                      notification_id_remote_,
                                      status_we_have_ + 1,//zero is not allowed as notification
                                      queue_,
                                      GASPI_BLOCK));
    status_we_have_sent_ = status_we_have_;
  }
}

void TransferForwardProducer::LiftLocalStatus(long status) {
  if (status > status_we_have_) {
    status_we_have_ = status;
  }
}

unsigned long TransferForwardProducer::GetLocalAcknowledgement(void) {
  if (status_we_have_sent_ > status_we_acknowledge_) {
    gaspi_return_t err = gaspi_wait(queue_, GASPI_TEST);
    if (err == GASPI_SUCCESS) {
      status_we_acknowledge_ = status_we_have_sent_;
    } else if (err != GASPI_TIMEOUT) {
      SUCCESS_OR_DIE(err);
    }
  }
  return status_we_acknowledge_;
}

unsigned long TransferForwardProducer::GetRemoteAcknowledgement(void) {
  gaspi_notification_t v;
  SUCCESS_OR_DIE(gaspi_notify_reset(segment_id_, notification_id_local_, &v));
  if (v > 0) {
    status_acknowledged_by_remote_ = v - 1;
  }
  return status_acknowledged_by_remote_;
}

void TransferForwardProducer::status(std::ostream& s) const {
  s << "size_=" << size_
    << " rank_=" << rank_
    << " segment_id_=" << long(segment_id_)
    << " buffer_offset_local_=" << buffer_offset_local_
    << " buffer_offset_remote_=" << buffer_offset_remote_
    << " notification_id_local_=" << notification_id_local_
    << " notification_id_remote_=" << notification_id_remote_
    << " queue_=" << long(queue_)
    << " status_we_have_=" << status_we_have_
    << " status_we_have_sent_=" << status_we_have_sent_
    << " status_we_acknowledge_=" << status_we_acknowledge_
    << " status_acknowledged_by_remote_=" << status_acknowledged_by_remote_;
}

//------------------------------------------------------------------------

TransferForwardConsumer::TransferForwardConsumer(
  const gaspi_rank_t rank,
  const gaspi_segment_id_t segment_id,
  const gaspi_notification_id_t notification_id_local,
  const gaspi_notification_id_t notification_id_remote,
  const gaspi_queue_id_t queue)
  : rank_(rank),
    segment_id_(segment_id),
    notification_id_local_(notification_id_local),
    notification_id_remote_(notification_id_remote),
    queue_(queue),
    status_(0){
  gaspi_config_t config;
  SUCCESS_OR_DIE(gaspi_config_get(&config));
  queue_depth_ = config.queue_depth;
}

unsigned long TransferForwardConsumer::GetStatus(void) {
  gaspi_notification_t v;
  SUCCESS_OR_DIE(gaspi_notify_reset(segment_id_, notification_id_local_, &v));
  if (v > 0) {
    status_ = v - 1;
  }
  return status_;
}

void TransferForwardConsumer::SetAcknowledgement(void) {
  ClearQueue();
  SUCCESS_OR_DIE(gaspi_notify(segment_id_, rank_, notification_id_remote_,
                              status_ + 1, queue_, GASPI_BLOCK));
}

void TransferForwardConsumer::ClearQueue(void) {
  gaspi_number_t entries;
  SUCCESS_OR_DIE(gaspi_queue_size(queue_, &entries));

  if ((long(queue_depth_) - long(entries)) < 5) {
    SUCCESS_OR_DIE(gaspi_wait(queue_, GASPI_BLOCK));
  }
}

void TransferForwardConsumer::status(std::ostream& s) const {
  s << "rank_=" << rank_
    << " segement_id_=" << long(segment_id_)
    << " notification_id_local_=" << notification_id_local_
    << " notification_id_remote_=" << notification_id_remote_
    << " queue_=" << long(queue_)
    << " queue_depth_=" << queue_depth_;
}

//------------------------------------------------------------------------

template <typename Dtype>
CommunicatorModel<Dtype>::CommunicatorModel(
  Blob<Dtype>* blob,
  const gaspi_segment_id_t segment_id,
  const gaspi_queue_id_t queue_transfer,
  const gaspi_queue_id_t queue_acknowledge,
  const gaspi_rank_t rank,
  const gaspi_rank_t num_ranks)
: blob_(blob),
  segment_id_(segment_id),
  queue_send_(queue_transfer),
  queue_acknowledge_(queue_acknowledge),
  status_(0),
  acknowledgement_local_(0),
  acknowledgement_total_(0){

  const long segment_size =  blob->count() * sizeof(Dtype);
  SUCCESS_OR_DIE(gaspi_segment_use(segment_id_, blob->mutable_cpu_data(),
                                   segment_size, GASPI_GROUP_ALL,
                                   GASPI_BLOCK, 0));

  const int bf = GetDataTreeBranchingFactor(num_ranks);
  std::vector<gaspi_rank_t> ranks_read = GetDataTreeReadRanks(rank, bf);
  std::vector<gaspi_rank_t> ranks_write = GetDataTreeWriteRanks(rank, num_ranks, bf);

  long buffer_index = 0;

  for (int i = 0; i < ranks_write.size(); i++) {
    const int rank_remote = ranks_write[i];

    std::vector<gaspi_rank_t> ranks_read_remote
      = GetDataTreeReadRanks(rank_remote, bf);
    std::vector<gaspi_rank_t> ranks_write_remote
      = GetDataTreeWriteRanks(rank_remote, num_ranks, bf);
    const long buffer_index_remote = ranks_write_remote.size()
        + std::find(ranks_read_remote.begin(), ranks_read_remote.end(), rank)
        - ranks_read_remote.begin();

    producer_.push_back(TransferForwardProducer(
      segment_size, rank_remote, segment_id, buffer_offset_, buffer_offset_,
      notification_base_id_ + buffer_index,
      notification_base_id_ + buffer_index_remote, queue_transfer));
    buffer_index++;
  }

  for (int i = 0; i < ranks_read.size(); i++) {
    const int rank_remote = ranks_read[i];

    std::vector<gaspi_rank_t> ranks_write_remote
      = GetDataTreeWriteRanks(rank_remote, num_ranks, bf);
    const long buffer_index_remote =
        std::find(ranks_write_remote.begin(), ranks_write_remote.end(), rank)
        - ranks_write_remote.begin();

    consumer_.push_back(TransferForwardConsumer(
      rank_remote, segment_id, notification_base_id_ + buffer_index,
      notification_base_id_ + buffer_index_remote, queue_acknowledge));
    buffer_index++;
  }
}

template <typename Dtype>
CommunicatorModel<Dtype>::~CommunicatorModel() {
  SUCCESS_OR_DIE(gaspi_segment_delete(segment_id_));
}



template <typename Dtype>
void CommunicatorModel<Dtype>::operator()(void) {
  UpdateStatus();
  UpdateAcknowledgement();
  SendModel();
}

template <typename Dtype>
void CommunicatorModel<Dtype>::UpdateStatus() {
  for (int i = 0; i < consumer_.size(); i++) {
    status_ = consumer_[i].GetStatus();
  }
}

template <typename Dtype>
void CommunicatorModel<Dtype>::UpdateAcknowledgement() {
  unsigned long acknowledgement = acknowledgement_local_;
  for (int i = 0; i < producer_.size(); i++) {
     acknowledgement =
       std::min(acknowledgement, producer_[i].GetLocalAcknowledgement());
  }
  if (acknowledgement > acknowledgement_total_) {
    for (int i = 0; i < consumer_.size(); i++) {
      if (acknowledgement == consumer_[i].GetStatus()) {
        consumer_[i].SetAcknowledgement();
      }
    }
    acknowledgement_total_ = acknowledgement;
  }
}

template <typename Dtype>
void CommunicatorModel<Dtype>::SendModel() {
  for (int i = 0; i < producer_.size(); i++) {
    producer_[i].LiftLocalStatus(status_);
    producer_[i]();
  }
}

template <typename Dtype>
void CommunicatorModel<Dtype>::Acknowledge(void) {
  acknowledgement_local_++;
  blob_->mutable_cpu_data();
  UpdateAcknowledgement();
}

template <typename Dtype>
void CommunicatorModel<Dtype>::UpdateModelOnMaster(void) {
  if (!HaveUpdateSource()) {
    status_++;
  }
}

template <typename Dtype>
bool CommunicatorModel<Dtype>::HaveUpdateSource(void) const {
  return !consumer_.size();
}

template <typename Dtype>
bool CommunicatorModel<Dtype>::Complete() {
  UpdateAcknowledgement();
  return (acknowledgement_total_==acknowledgement_local_);
}

template <typename Dtype>
std::vector<gaspi_rank_t> CommunicatorModel<Dtype>::GetDataTreeWriteRanks(
  gaspi_rank_t rank, gaspi_rank_t num_ranks, int branching_factor) {
  std::vector<gaspi_rank_t> r;
  for (long i = 1; i <= branching_factor; i++) {
    const long remote_rank = branching_factor * long(rank) + i;
    if (remote_rank < long(num_ranks)) r.push_back(remote_rank);
  }
  return r;
}

template <typename Dtype>
std::vector<gaspi_rank_t> CommunicatorModel<Dtype>::GetDataTreeReadRanks(
  gaspi_rank_t rank, int branching_factor) {
  std::vector<gaspi_rank_t> r;
  if (rank > 0)
    r.push_back((int(rank) - 1) / branching_factor);
  return r;
}

template <typename Dtype>
int CommunicatorModel<Dtype>::GetDataTreeBranchingFactor(long num_ranks) {
  static const long branch_max = 100;

  long hops_final = num_ranks;
  long branch_final = 2;
  for (long branch = 2; branch <= branch_max; branch++) {
    long num_levels;
    {
      long ranks_in_level = 1;
      long ranks_in_tree = 1;
      for (num_levels = 0; ranks_in_tree < num_ranks; num_levels++) {
        ranks_in_level *= branch;
        ranks_in_tree += ranks_in_level;
      }
    }
    const long hops = branch * num_levels;
    if (hops < hops_final) {
      hops_final = hops;
      branch_final = branch;
    }
  }
  return branch_final;
}

template <typename Dtype>
void CommunicatorModel<Dtype>::status(std::ostream& s) const {
  s << "segment_id_=" << long(segment_id_)
    << " queue_send_=" << long(queue_send_)
    << " queue_acknowledge_=" << long(queue_acknowledge_)
    << std::endl;
  s  << "print consumer:" << std::endl;
  for (int i = 0; i < consumer_.size(); i++) {
    consumer_[i].status(s);
    s << std::endl;
  }
  s << "end consumer print producer:" << std::endl;
  for (int i = 0; i < producer_.size(); i++) {
    producer_[i].status(s);
    s << std::endl;
  }
}

template class CommunicatorModel<float>;
template class CommunicatorModel<double>;

}
