#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/parallel_inMem_data_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/GPIhelper.h"


namespace caffe {

template <typename Dtype>
ParallelInMemDataLayer<Dtype>::~ParallelInMemDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void ParallelInMemDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //GPI setup
  gaspi_rank_t currentRank=0;
  gaspi_rank_t numRanks=0;
  SUCCESS_OR_DIE( gaspi_proc_rank(&currentRank));
  SUCCESS_OR_DIE( gaspi_proc_num(&numRanks));

  const int new_height = this->layer_param_.parallel_inmem_data_param().new_height();
  const int new_width  = this->layer_param_.parallel_inmem_data_param().new_width();
  const bool is_color  = this->layer_param_.parallel_inmem_data_param().is_color();
  string root_folder = this->layer_param_.parallel_inmem_data_param().root_folder();
  string labelFileName = this->layer_param_.parallel_inmem_data_param().label();
  int headerSize = this->layer_param_.parallel_inmem_data_param().headersize();
  const int batch_size = this->layer_param_.parallel_inmem_data_param().batch_size();
  bool is_test = this->layer_param_.parallel_inmem_data_param().test();

  int numChannels = 1;
  if (is_color)
    numChannels =3;

  int imageSize=new_height*new_width*numChannels;

  if (currentRank==0)
    LOG(INFO) << "Opening label file " << labelFileName<<std::endl;

  std::ifstream labelFile(labelFileName.c_str());
  int label;
  while (labelFile >> label)
     Labels_.push_back(label);  
  labelFile.close();

  const string& source = this->layer_param_.parallel_inmem_data_param().source();
  if (currentRank==0)
    LOG(INFO) << "Opening data file " << source<<std::endl;

  //open binary  
  std::ifstream inFile;
  inFile.open(source.c_str(),  std::ios::binary);
  
  //get bin file size
  inFile.seekg(0,inFile.end);
  long binFileSize = inFile.tellg();
  inFile.seekg(0,inFile.beg);

  if (currentRank==0)
  {
    LOG(INFO) << " binary size = "<< binFileSize << " target size: "<<
        (headerSize+new_height*new_width*numChannels)*Labels_.size();
  }

  //offset for each rank
  batchesGlobal_ = Labels_.size()/batch_size; //round here
  batchesPerRank_ = batchesGlobal_ / numRanks; //round again
  batchMemSize_ = imageSize * batch_size;
  samplesPerRank_ = batchesPerRank_ * batch_size;
  rankOffset_ = samplesPerRank_ * (imageSize+headerSize);
  maxLabel_ = batchesPerRank_ * numRanks * batch_size;
  label_id_ = currentRank*samplesPerRank_;

  if (currentRank==0)
  {
    LOG(INFO) << " # global batches: "<< batchesGlobal_ 
	<< ", # batches per Rank: "<< batchesPerRank_
        << ", # samples per Rank : "<< samplesPerRank_;
  }

  //gaspi setup
  if (is_test) //use different segments for train and val data, seg 0 for solver com
	segment_id_ = 2;
  else
	segment_id_ = 1;

  qID_ = currentRank%7; //use 8 queues 
  
  //allocate global mem: n storage batches + input buffer  
  SUCCESS_OR_DIE(
        gaspi_segment_create( segment_id_, (batchesPerRank_+1)*batchMemSize_*sizeof(uint8_t), GASPI_GROUP_ALL,
            GASPI_BLOCK, GASPI_MEM_INITIALIZED)
  );

  //get pointer to global mem    
  SUCCESS_OR_DIE(
      gaspi_segment_ptr (segment_id_, &segment_0_P_)
  );

  //get local pointer to segment
  rawData_ = (uint8_t*) segment_0_P_;

  //open binary  
  long currentOffset = (headerSize+imageSize) * label_id_*sizeof(uint8_t);
  inFile.seekg(currentOffset );
 
  //read from file
  long pos=0;
  for (long i = 0; i < samplesPerRank_; ++i)
  {
    if (i%10000==0)
    	std::cerr<<"Rank "<<currentRank<<": read "<<i<<" samples from file\n";
    uint8_t tmp;	
    for (long e=0;e<headerSize;++e)
    {
        //read header   
        inFile.read(reinterpret_cast<char*>(&tmp),sizeof(uint8_t));
    }
    if (numChannels==1)
    {
      for (long e=0;e<imageSize;++e,++pos)
      {
        inFile.read(reinterpret_cast<char*>(&tmp),sizeof(uint8_t));
        rawData_[pos]=tmp;
      }
    }
    else //open cvs hav BGR color order!
    {
        uint8_t r,g,b;
        for (long e=0;e<imageSize;++e,++pos)
        {
                inFile.read(reinterpret_cast<char*>(&r),sizeof(uint8_t));
                inFile.read(reinterpret_cast<char*>(&g),sizeof(uint8_t));
                inFile.read(reinterpret_cast<char*>(&b),sizeof(uint8_t));

                rawData_[pos]=(Dtype)b;
                pos++;e++;
                rawData_[pos]=(Dtype)g;
                pos++;e++;
                rawData_[pos]=(Dtype)r;
        }
    }
  }

  inFile.close();

  vector<int> top_shape(4);
  top_shape[0]=1;
  top_shape[1]=numChannels;
  top_shape[2]=new_height;
  top_shape[3]=new_width;
  this->transformed_data_.Reshape(top_shape);

  // Reshape prefetch_data and top[0] according to the batch_size.
  top_shape[0] = batch_size;
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  // label
  vector<int> label_shape(1, batch_size);
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->label_.Reshape(label_shape);
  }

/*
  label_id_ = (currentRank+1)*samplesPerRank_;
  if (label_id_>=maxLabel_)
        label_id_=0;
*/

}

// This function is called on prefetch thread
template <typename Dtype>
void ParallelInMemDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  //GPI setup
  CPUTimer batch_timer;
  batch_timer.Start();
  gaspi_rank_t currentRank=0;
  gaspi_rank_t numRanks=0;
  SUCCESS_OR_DIE( gaspi_proc_rank(&currentRank));
  SUCCESS_OR_DIE( gaspi_proc_num(&numRanks));

  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  ParallelInMemDataParameter parallel_inmem_data_param = this->layer_param_.parallel_inmem_data_param();
  const int batch_size = parallel_inmem_data_param.batch_size();
  const int new_height = parallel_inmem_data_param.new_height();
  const int new_width = parallel_inmem_data_param.new_width();
  const bool is_color = parallel_inmem_data_param.is_color();

  int numChannels = 1;
  if (is_color)
    numChannels =3;

  int imageSize=new_height*new_width*numChannels;

  vector<int> top_shape(4);
  top_shape[0]=1;
  top_shape[1]=numChannels;
  top_shape[2]=new_height;
  top_shape[3]=new_width;
  this->transformed_data_.Reshape(top_shape);
   
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  //compute rank of target batch
  long batchId = label_id_ / batch_size;
  gaspi_rank_t sRank = batchId / batchesPerRank_;
  long sbatchId = batchId - sRank*batchesPerRank_;
  long lbatchId = batchId - currentRank*batchesPerRank_;
  long sOffset = sbatchId*batch_size*imageSize*sizeof(uint8_t);

  //std::cout<<"BATCH "<<currentRank<<" "<<label_id_<<" "<<batchId<<" "<<sRank<<" "<<sbatchId<<" "<<std::endl;

  //get local pointer to segment
  rawData_ = (uint8_t*) segment_0_P_;
  if (sRank == currentRank) //data is local
  {
	rawData_+= lbatchId*batch_size*imageSize*sizeof(uint8_t);
  }
  else //get data first
  {
	SUCCESS_OR_DIE(
		gaspi_read(segment_id_, samplesPerRank_*imageSize*sizeof(uint8_t), sRank, segment_id_, sOffset,
			batch_size*imageSize*sizeof(uint8_t),qID_, GASPI_BLOCK)
	);
	SUCCESS_OR_LOG(
        	gaspi_wait(qID_, GASPI_BLOCK)
   	);

	rawData_+=samplesPerRank_*imageSize*sizeof(uint8_t);
	
  }


  // get batch
  for (long item_id = 0; item_id < batch_size; ++item_id) 
  {
    // get a blob
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
 
    if (numChannels==1)
    {   
      cv::Mat cv_img(new_height, new_width, CV_8UC1, rawData_);
      this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    }
    else //open cvs hav BGR color order!
    {
        cv::Mat cv_img(new_height, new_width, CV_8UC3, rawData_);

	/*
        if(Labels_[label_id_]==10)
        {
                std::stringstream tmpName;
                tmpName <<"/home/keuper/tmp/tmp_"<<currentRank<<"_"<<sRank<<"_"<<label_id_<<"_"<<Labels_[label_id_]<<".png";
                cv::imwrite( tmpName.str(), cv_img);
        }
	*/

	this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
        
    }
    
    rawData_+=imageSize*sizeof(uint8_t);
	 
    prefetch_label[item_id] = Labels_[label_id_];

    label_id_++;	
    if (label_id_ >= maxLabel_)
    {
	label_id_=0;
    } 

 }

 batch_timer.Stop();
 if (currentRank==0)
        std::cout<<"BATCH time: "<<batch_timer.MilliSeconds() << " ms."<<std::endl; 
  
}

template <typename Dtype>
void ParallelInMemDataLayer<Dtype>::ShuffleImages() {
//empty dummy to meet enheritance conditions
}



INSTANTIATE_CLASS(ParallelInMemDataLayer);
REGISTER_LAYER_CLASS(ParallelInMemData);

}  // namespace caffe
#endif  // USE_OPENCV
