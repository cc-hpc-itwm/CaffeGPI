#ifndef CAFFE_PARALLEL_DATA_LAYER_HPP_
#define CAFFE_PARALLEL_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>
#include "hdf5.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
//#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/filler.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/db.hpp"
#include "caffe/util/GPIhelper.h"

namespace caffe {
template <typename Dtype>
class ParallelInMemDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ParallelInMemDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~ParallelInMemDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ParallelInMemData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);

  uint8_t* rawData_;
  std::vector<int> Labels_;
  long label_id_; //offset in global label order
  long maxLabel_;
  long samplesPerRank_;
  long batchesPerRank_;
  long batchesGlobal_;
  long batchMemSize_;
  long rankOffset_;
  gaspi_pointer_t segment_0_P_;
  gaspi_number_t qID_;
  gaspi_segment_id_t segment_id_;
};


}  // namespace caffe

#endif  // CAFFE_DATA_LAYERS_HPP_
