#include "caffe/gpi_communicator_diff.hpp"

namespace caffe {

template <typename Dtype>
void CommunicatorDiff<Dtype>::operator()(void) {
  for (long i = 0; i < com_buffers_diff_read_.size(); i++) {
    RingBufferRead<Dtype>& buffer = com_buffers_diff_read_[i];
    while (com_buffers_diff_read_status_[i] < calculated_blobs_.size()) {
      Blob<Dtype>& blob = *calculated_blobs_[com_buffers_diff_read_status_[i]];
      if (buffer.Add(blob.mutable_cpu_diff(), blob.count())) {
        break;
      } else {
        com_buffers_diff_read_status_[i]++;
      }
    }
  }
  for (long i = 0; i < com_buffers_diff_write_.size(); i++) {
    RingBufferWrite<Dtype>& buffer = com_buffers_diff_write_[i];
    while ((com_buffers_diff_write_status_[i] < calculated_blobs_.size())
           && CommunicateLayerDiffReadFinished(com_buffers_diff_write_status_[i])) {
      Blob<Dtype>& blob = *calculated_blobs_[com_buffers_diff_write_status_[i]];
//todo aggregate diffs from other cpu too
      if (buffer.Write(blob.cpu_diff(), blob.count())) {
        break;
      } else {
        com_buffers_diff_write_status_[i]++;
      }
    }
  }
}

template <typename Dtype>
bool CommunicatorDiff<Dtype>::CommunicateLayerDiffFinished() {
  int running = !CommunicateLayerDiffReadFinished(calculated_blobs_.size() - 1);
  for (long i = 0; i < com_buffers_diff_write_status_.size(); i++) {
    running |= (com_buffers_diff_write_status_[i] < calculated_blobs_.size());
  }
  return !running;
}

template <typename Dtype>
bool CommunicatorDiff<Dtype>::CommunicateLayerDiffReadFinished(int index) {
  int running = 0;
  for (long i = 0; i < com_buffers_diff_read_status_.size(); i++) {
    running |= (com_buffers_diff_read_status_[i] <= index);
  }
  return !running;
}

template <typename Dtype>
void CommunicatorDiff<Dtype>::AddCalculatedBlob(Blob<Dtype>* blob) {
  calculated_blobs_.push_back(blob);
}

template <typename Dtype>
void CommunicatorDiff<Dtype>::ResetCommunicationStatus(void) {
  for (int i = 0; i < com_buffers_diff_read_status_.size(); i++)
    com_buffers_diff_read_status_[i] = 0;
  for (int i = 0; i < com_buffers_diff_write_status_.size(); i++)
    com_buffers_diff_write_status_[i] = 0;
  calculated_blobs_.resize(0);
}

template <typename Dtype>
CommunicatorDiff<Dtype>::CommunicatorDiff(
  const long buffer_size,
  const gaspi_notification_id_t notification_id_base,
  const gaspi_segment_id_t segment_id,
  const gaspi_queue_id_t queue,
  const gaspi_rank_t rank,
  const gaspi_rank_t num_ranks) :
  segment_id_(segment_id),
  rank_(rank),
  num_ranks_(num_ranks) {
  const int bf = GetDiffTreeBranchingFactor();
  std::vector<gaspi_rank_t> ranks_read = GetDiffTreeReadRanks(rank_, bf);
  std::vector<gaspi_rank_t> ranks_write = GetDiffTreeWriteRanks(rank_, bf);

  const long diff_segment_size
    = std::max(buffer_size * (ranks_read.size() + ranks_write.size()), 1ul);
  SUCCESS_OR_DIE(gaspi_segment_create(segment_id_,
                                      diff_segment_size * sizeof(Dtype),
                                      GASPI_GROUP_ALL,
                                      GASPI_BLOCK,
                                      GASPI_MEM_UNINITIALIZED));

  long buffer_index = 0;

  for (int i = 0; i < ranks_write.size(); i++) {
    const int rank_remote = ranks_write[i];

    std::vector<gaspi_rank_t> ranks_read_remote = GetDiffTreeReadRanks(rank_remote, bf);
    std::vector<gaspi_rank_t> ranks_write_remote = GetDiffTreeWriteRanks(rank_remote, bf);
    const long buffer_index_remote = ranks_write_remote.size()
        + std::find(ranks_read_remote.begin(), ranks_read_remote.end(), rank_)
        - ranks_read_remote.begin();

    com_buffers_diff_write_.push_back(RingBufferWrite<Dtype>(
      buffer_size, segment_id_, notification_id_base + buffer_index,
      buffer_index * buffer_size * sizeof(Dtype),
      rank_remote, segment_id_, notification_id_base + buffer_index_remote,
      buffer_index_remote * buffer_size * sizeof(Dtype), queue));
    com_buffers_diff_write_status_.push_back(0);
    buffer_index++;
  }

  for (int i = 0; i < ranks_read.size(); i++) {
    const int rank_remote = ranks_read[i];

    std::vector<gaspi_rank_t> ranks_write_remote = GetDiffTreeWriteRanks(rank_remote, bf);
    long buffer_index_remote = 0;
    for (int j = 0; (j < ranks_write_remote.size()) && (ranks_write_remote[j] != rank_); j++) {
      buffer_index_remote++;
    }
    com_buffers_diff_read_.push_back(RingBufferRead<Dtype>(
      buffer_size, segment_id_, notification_id_base + buffer_index,
      buffer_index * buffer_size * sizeof(Dtype),
      ranks_read[i], segment_id_, notification_id_base + buffer_index_remote,
      buffer_index_remote * buffer_size * sizeof(Dtype), queue));
    com_buffers_diff_read_status_.push_back(0);
    buffer_index ++;
  }
}

template <typename Dtype>
CommunicatorDiff<Dtype>::~CommunicatorDiff() {
  SUCCESS_OR_DIE(gaspi_segment_delete(segment_id_));
}

template <typename Dtype>
int CommunicatorDiff<Dtype>::GetDiffTreeBranchingFactor() const {
  return 2;
}

template <typename Dtype>
std::vector<gaspi_rank_t> CommunicatorDiff<Dtype>::GetDiffTreeWriteRanks(
  gaspi_rank_t rank, int branching_factor) const {
  std::vector<gaspi_rank_t>  r;

  if (rank > 0) {
    long power = 1; // = branching_factor ** 0
    while((power * branching_factor) <= long(rank)) {
      power *= branching_factor;
    }
    r.push_back(long(rank) % power);
  }
  return r;
}

template <typename Dtype>
std::vector<gaspi_rank_t> CommunicatorDiff<Dtype>::GetDiffTreeReadRanks(
  gaspi_rank_t rank, int branching_factor) const {
  std::vector<gaspi_rank_t> r;

  long power = 1;
  while (power < long(num_ranks_)) {
    if (power > long(rank)) {
      for(long i = 1; i < branching_factor; i++) {
        long n = i * power + long(rank);
        if (n < long(num_ranks_)) {
          r.push_back(n);
        }
      }
    }
    power *= branching_factor;
  }
  return r;
}

template class CommunicatorDiff<float>;
template class CommunicatorDiff<double>;

}
