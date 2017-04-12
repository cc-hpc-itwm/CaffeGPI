#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/parallel_data_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/GPIhelper.h"


namespace caffe {

template <typename Dtype>
ParallelDataLayer<Dtype>::~ParallelDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void ParallelDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //GPI setup
  gaspi_rank_t currentRank=0;
  gaspi_rank_t numRanks=0;
  SUCCESS_OR_DIE( gaspi_proc_rank(&currentRank));
  SUCCESS_OR_DIE( gaspi_proc_num(&numRanks));

  const int new_height = this->layer_param_.parallel_data_param().new_height();
  const int new_width  = this->layer_param_.parallel_data_param().new_width();
  const bool is_color  = this->layer_param_.parallel_data_param().is_color();
  string root_folder = this->layer_param_.parallel_data_param().root_folder();
  string labelFileName = this->layer_param_.parallel_data_param().label();
  int headerSize = this->layer_param_.parallel_data_param().headersize();
  const int batch_size = this->layer_param_.parallel_data_param().batch_size();
  int numChannels = 1;
  if (is_color)
    numChannels =3;

  if (currentRank==0)
    LOG(INFO) << "Opening label file " << labelFileName<<std::endl;

  std::ifstream labelFile(labelFileName.c_str());
  int label;
  while (labelFile >> label)
     Labels_.push_back(label);  
  labelFile.close();

  const string& source = this->layer_param_.parallel_data_param().source();
  if (currentRank==0)
    LOG(INFO) << "Opening data file " << source<<std::endl;

  //open binary  
  std::ifstream inFile;
  inFile.open(source.c_str(),  std::ios::binary);
  
  //get bin file size
  inFile.seekg(0,inFile.end);
  long binFileSize = inFile.tellg();
  inFile.seekg(0,inFile.beg);
  inFile.close();

  if (currentRank==0)
  {
    LOG(INFO) << " binary size = "<< binFileSize << " target size: "<<
        (headerSize+new_height*new_width*numChannels)*Labels_.size();
  }

  //offset for each rank
  label_id_ = (Labels_.size()/numRanks)*currentRank;
  data_id_=0;  

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

}

// This function is called on prefetch thread
template <typename Dtype>
void ParallelDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  //GPI setup
  CPUTimer batch_timer;
  batch_timer.Start();
  gaspi_rank_t currentRank=0;
  gaspi_rank_t numRanks=0;
  SUCCESS_OR_DIE( gaspi_proc_rank(&currentRank));
  SUCCESS_OR_DIE( gaspi_proc_num(&numRanks));

  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  ParallelDataParameter parallel_data_param = this->layer_param_.parallel_data_param();
  const int batch_size = parallel_data_param.batch_size();
  const int new_height = parallel_data_param.new_height();
  const int new_width = parallel_data_param.new_width();
  const bool is_color = parallel_data_param.is_color();
  string root_folder = parallel_data_param.root_folder();
  const string& source = this->layer_param_.parallel_data_param().source();
  int headerSize = this->layer_param_.parallel_data_param().headersize();

  int numChannels = 1;
  if (is_color)
    numChannels =3;

  int imageSize=new_height*new_width*numChannels;

  //open binary  
  std::ifstream inFile;
  inFile.open(source.c_str(),  std::ios::binary);
  long currentOffset = (headerSize+imageSize) * label_id_*sizeof(uint8_t); 
  inFile.seekg(currentOffset );

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

  //pre allocate image mem
  rawData_= new uint8_t[imageSize * sizeof(uint8_t)];
  uint8_t tmp;

  // get batch
  for (long item_id = 0; item_id < batch_size; ++item_id , label_id_++) 
  {
    if (label_id_ >= Labels_.size())
    {
        label_id_=0;
        inFile.seekg(0,inFile.beg);
    }

    // get a blob
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
 
    for (long e=0;e<headerSize;++e)
    {
        //read header   
        inFile.read(reinterpret_cast<char*>(&tmp),sizeof(uint8_t));
    }
    if (numChannels==1)
    {   
      for (long e=0;e<imageSize;++e)
      {
        inFile.read(reinterpret_cast<char*>(&tmp),sizeof(uint8_t));
        rawData_[e]=(Dtype)tmp;
      }
      cv::Mat cv_img(new_height, new_width, CV_8UC1, rawData_);
      this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    }
    else //open cvs hav BGR color order!
    {
        uint8_t r,g,b;
        for (long e=0;e<imageSize;++e)
        {
                inFile.read(reinterpret_cast<char*>(&r),sizeof(uint8_t));
                inFile.read(reinterpret_cast<char*>(&g),sizeof(uint8_t));
                inFile.read(reinterpret_cast<char*>(&b),sizeof(uint8_t));

                rawData_[e]=(Dtype)b;
                e++;
                rawData_[e]=(Dtype)g;
                e++;
                rawData_[e]=(Dtype)r;
        }
        cv::Mat cv_img(new_height, new_width, CV_8UC3, rawData_);
        this->data_transformer_->Transform(cv_img, &(this->transformed_data_));

	/*
        if(currentRank==0&&Labels_[label_id_]==0)
        {
                std::stringstream tmpName;
                tmpName <<"/u/k/keuper/tmp/tmp_"<<currentRank<<"_"<<label_id_<<"_"<<Labels_[label_id_]<<".png";
                cv::imwrite( tmpName.str(), cv_img);
        }
	*/
        
    }
 
    prefetch_label[item_id] = Labels_[label_id_];

 }

 delete[] rawData_;
 inFile.close(); 
  
 batch_timer.Stop();
 //if (currentRank==0)
 //       std::cout<<"BATCH time: "<<batch_timer.MilliSeconds() << " ms."<<std::endl; 
  
}

template <typename Dtype>
void ParallelDataLayer<Dtype>::ShuffleImages() {
//empty dummy to meet enheritance conditions
}



INSTANTIATE_CLASS(ParallelDataLayer);
REGISTER_LAYER_CLASS(ParallelData);

}  // namespace caffe
#endif  // USE_OPENCV
