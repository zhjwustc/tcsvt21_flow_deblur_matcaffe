#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>


#include <cmath>
#include "caffe/util/math_functions.hpp"
#include "caffe/common.hpp"
#include "caffe/util/util_img.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/im2col.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
using namespace std;
#define ELLISION 1e-6
using namespace cv;

/**
 * @todo parallelization for all image processing
 */
namespace caffe {

template <typename Dtype>
void BiLinearResizeMat_cpu(const Dtype* src, const int src_height, const int src_width,
		Dtype* dst, const int dst_height, const int dst_width)
{
	const Dtype scale_w = src_width / (Dtype)dst_width;
	const Dtype scale_h = src_height / (Dtype)dst_height;
	Dtype* dst_data = dst;
	const Dtype* src_data = src;

	for(int dst_h = 0; dst_h < dst_height; ++dst_h){
		Dtype fh = dst_h * scale_h;

		int src_h = std::floor(fh);

		fh -= src_h;
		const Dtype w_h0 = std::abs((Dtype)1.0 - fh);
		const Dtype w_h1 = std::abs(fh);

		const int dst_offset_1 =  dst_h * dst_width;
		const int src_offset_1 =  src_h * src_width;

		Dtype* dst_data_ptr = dst_data + dst_offset_1;

		for(int dst_w = 0 ; dst_w < dst_width; ++dst_w){

			Dtype fw = dst_w * scale_w;
			int src_w = std::floor(fw);
			fw -= src_w;
			const Dtype w_w0 = std::abs((Dtype)1.0 - fw);
			const Dtype w_w1 = std::abs(fw);


			Dtype dst_value = 0;

			const int src_idx = src_offset_1 + src_w;
			dst_value += (w_h0 * w_w0 * src_data[src_idx]);
			int flag = 0;
			if (src_w + 1 < src_width){
				dst_value += (w_h0 * w_w1 * src_data[src_idx + 1]);
				++flag;
			}
			if (src_h + 1 < src_height){
				dst_value += (w_h1 * w_w0 * src_data[src_idx + src_width]);
				++flag;
			}

			if (flag>1){
				dst_value += (w_h1 * w_w1 * src_data[src_idx + src_width + 1]);
//				++flag;
			}
			*(dst_data_ptr++) = dst_value;
		}
	}

}


template void BiLinearResizeMat_cpu(const float* src, const int src_height, const int src_width,
		float* dst, const int dst_height, const int dst_width);

template void BiLinearResizeMat_cpu(const double* src, const int src_height, const int src_width,
		double* dst, const int dst_height, const int dst_width);

template <typename Dtype>
void RuleBiLinearResizeMat_cpu(const Dtype* src,
		Dtype* dst, const int dst_h, const int dst_w,
		const Dtype* loc1, const Dtype* weight1, const Dtype* loc2,const Dtype* weight2,
		const	Dtype* loc3,const Dtype* weight3,const Dtype* loc4, const Dtype* weight4)
{

	Dtype* dst_data = dst;
	const Dtype* src_data = src;

	int loop_n = dst_h  * dst_w ;
	for(int i=0 ; i< loop_n; i++)
	{


		dst_data[i] += (weight1[i] * src_data[static_cast<int>(loc1[i])]);
		dst_data[i] += (weight2[i] * src_data[static_cast<int>(loc2[i])]);
		dst_data[i] += (weight3[i] * src_data[static_cast<int>(loc3[i])]);
		dst_data[i] += (weight4[i] * src_data[static_cast<int>(loc4[i])]);

	}

}

template void RuleBiLinearResizeMat_cpu(const float* src,
		float* dst, const int dst_h, const int dst_w,
		const float* loc1, const float* weight1, const float* loc2,const float* weight2,
		const	float* loc3,const float* weight3,const float* loc4, const float* weight4);
template void RuleBiLinearResizeMat_cpu(const double* src,
		double* dst, const int dst_h, const int dst_w,
		const double* loc1, const double* weight1, const double* loc2,const double* weight2,
		const	double* loc3,const double* weight3,const double* loc4, const double* weight4);



template <typename Dtype>
void GetBiLinearResizeMatRules_cpu( const int src_height, const int src_width,
		 const int dst_height, const int dst_width,
		Dtype* loc1, Dtype* weight1, Dtype* loc2, Dtype* weight2,
		Dtype* loc3, Dtype* weight3, Dtype* loc4, Dtype* weight4)
{
	const Dtype scale_w = src_width / (Dtype)dst_width;
	const Dtype scale_h = src_height / (Dtype)dst_height;


	int loop_n = dst_height * dst_width;


	for(int i=0 ; i< loop_n; i++)
	{
		int dst_h = i /dst_width;
		Dtype fh = dst_h * scale_h;
		int src_h ;
		if(typeid(Dtype).name() == typeid(double).name())
			 src_h = floor(fh);
		else
			 src_h = floorf(fh);

		fh -= src_h;
		const Dtype w_h0 = std::abs((Dtype)1.0 - fh);
		const Dtype w_h1 = std::abs(fh);

		const int dst_offset_1 =  dst_h * dst_width;
		const int src_offset_1 =  src_h * src_width;

		int dst_w = i %dst_width;
		Dtype fw = dst_w * scale_w;

		int src_w ;
		if(typeid(Dtype).name() == typeid(double).name())
			src_w = floor(fw);
		else
			src_w = floorf(fw);

		fw -= src_w;
		const Dtype w_w0 = std::abs((Dtype)1.0 - fw);
		const Dtype w_w1 = std::abs(fw);

		const int dst_idx = dst_offset_1 + dst_w;
//		dst_data[dst_idx] = 0;

		const int src_idx = src_offset_1 + src_w;

		loc1[dst_idx] = static_cast<Dtype>(src_idx);
		weight1[dst_idx] = w_h0 * w_w0;


		loc2[dst_idx] = 0;
		weight2[dst_idx] = 0;

		weight3[dst_idx] = 0;
		loc3[dst_idx] = 0;

		loc4[dst_idx] = 0;
		weight4[dst_idx] = 0;

		if (src_w + 1 < src_width)
		{
			loc2[dst_idx] = static_cast<Dtype>(src_idx + 1);
			weight2[dst_idx] = w_h0 * w_w1;
//			dst_data[dst_idx] += (w_h0 * w_w1 * src_data[src_idx + 1]);
		}

		if (src_h + 1 < src_height)
		{
//			dst_data[dst_idx] += (w_h1 * w_w0 * src_data[src_idx + src_width]);
			weight3[dst_idx] = w_h1 * w_w0;
			loc3[dst_idx] = static_cast<Dtype>(src_idx + src_width);
		}

		if (src_w + 1 < src_width && src_h + 1 < src_height)
		{
			loc4[dst_idx] = static_cast<Dtype>(src_idx + src_width + 1);
			weight4[dst_idx] = w_h1 * w_w1;
//			dst_data[dst_idx] += (w_h1 * w_w1 * src_data[src_idx + src_width + 1]);
		}

	}

}


template void GetBiLinearResizeMatRules_cpu(  const int src_height, const int src_width,
		 const int dst_height, const int dst_width,
		float* loc1, float* weight1, float* loc2, float* weight2,
				float* loc3, float* weight3, float* loc4, float* weight4);

template void GetBiLinearResizeMatRules_cpu(  const int src_height, const int src_width,
		 const int dst_height, const int dst_width,
		double* loc1, double* weight1, double* loc2, double* weight2,
				double* loc3, double* weight3, double* loc4, double* weight4);




template <typename Dtype>
void ResizeBlob_cpu(const Blob<Dtype>* src, const int src_n, const int src_c,
		Blob<Dtype>* dst, const int dst_n, const int dst_c) {


	const int src_channels = src->channels();
	const int src_height = src->height();
	const int src_width = src->width();
	const int src_offset = (src_n * src_channels + src_c) * src_height * src_width;

	const int dst_channels = dst->channels();
	const int dst_height = dst->height();
	const int dst_width = dst->width();
	const int dst_offset = (dst_n * dst_channels + dst_c) * dst_height * dst_width;


	const Dtype* src_data = &(src->cpu_data()[src_offset]);
	Dtype* dst_data = &(dst->mutable_cpu_data()[dst_offset]);
	BiLinearResizeMat_cpu(src_data,  src_height,  src_width,
			dst_data,  dst_height,  dst_width);
}

template void ResizeBlob_cpu(const Blob<float>* src, const int src_n, const int src_c,
		Blob<float>* dst, const int dst_n, const int dst_c);
template void ResizeBlob_cpu(const Blob<double>* src, const int src_n, const int src_c,
		Blob<double>* dst, const int dst_n, const int dst_c);


template <typename Dtype>
void ResizeBlob_cpu(const Blob<Dtype>* src,Blob<Dtype>* dst)
{
	CHECK(src->num() == dst->num())<<"src->num() == dst->num()";
	CHECK(src->channels() == dst->channels())<< "src->channels() == dst->channels()";

	for(int n=0;n< src->num();++n)
	{
		for(int c=0; c < src->channels() ; ++c)
		{
			ResizeBlob_cpu(src,n,c,dst,n,c);
		}
	}
}
template void ResizeBlob_cpu(const Blob<float>* src,Blob<float>* dst);
template void ResizeBlob_cpu(const Blob<double>* src,Blob<double>* dst);



template <typename Dtype>
void ResizeBlob_cpu(const Blob<Dtype>* src,Blob<Dtype>* dst,
		Blob<Dtype>* loc1, Blob<Dtype>* loc2, Blob<Dtype>* loc3, Blob<Dtype>* loc4){

	CHECK(src->num() == dst->num())<<"src->num() == dst->num()";
	CHECK(src->channels() == dst->channels())<< "src->channels() == dst->channels()";

	GetBiLinearResizeMatRules_cpu(  src->height(),src->width(),
			 dst->height(), dst->width(),
			loc1->mutable_cpu_data(), loc1->mutable_cpu_diff(), loc2->mutable_cpu_data(), loc2->mutable_cpu_diff(),
			loc3->mutable_cpu_data(), loc3->mutable_cpu_diff(), loc4->mutable_cpu_data(), loc4->mutable_cpu_diff());


	ResizeBlob_cpu(src, dst );

}
template void ResizeBlob_cpu(const Blob<float>* src,Blob<float>* dst,
		Blob<float>* loc1, Blob<float>* loc2, Blob<float>* loc3, Blob<float>* loc4);
template void ResizeBlob_cpu(const Blob<double>* src,Blob<double>* dst,
		Blob<double>* loc1, Blob<double>* loc2, Blob<double>* loc3, Blob<double>* loc4);


/**
 *  src.Shape(nums_, channels, height, width)
 *  dst.Reshape(nums_*width_out_*height_out_???channels_, kernel_h, kernel_w);
 */
template <typename Dtype>
void GenerateSubBlobs_cpu(const Blob<Dtype>& src,
		Blob<Dtype>& dst,const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w, const int stride_h,
	    const int stride_w)
{
	const int nums_ = src.num();
	const int channels_ = src.channels();
	const int height_ = src.height();
	const int width_ = src.width();
	const int height_col_ =(height_ + 2 * pad_h - kernel_h) / stride_h + 1;
	const int width_col_ = (width_ + 2 * pad_w - kernel_w) / stride_w + 1;

	/*
	 * actually after im2col_v2, data is stored as
	 * col_buffer_.Reshape(1*height_out_*width_out_, channels_  , kernel_h_ , kernel_w_);
	 * */
	dst.Reshape(height_col_*width_col_*nums_,channels_,  kernel_h, kernel_w);
	caffe_set(dst.count(),Dtype(0),dst.mutable_cpu_data());
	for(int n = 0; n < nums_; n++){

		const Dtype*  src_data = src.cpu_data() + src.offset(n);
		Dtype*  dst_data = dst.mutable_cpu_data() + dst.offset(n*height_col_*width_col_);
		caffe::im2col_v2_cpu(src_data, channels_, height_,
	            width_, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
	            dst_data);

	}
}

template void GenerateSubBlobs_cpu(const Blob<float>& src,
		Blob<float>& dst,const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w, const int stride_h,
	    const int stride_w);
template void GenerateSubBlobs_cpu(const Blob<double>& src,
		Blob<double>& dst,const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w, const int stride_h,
	    const int stride_w);

/**
 *  end_h is not included
 */
template <typename Dtype>
void CropBlobs_cpu( const Blob<Dtype>&src,
		const int start_h, const int start_w,
		const int end_h, const int end_w, Blob<Dtype>&dst)
{
	const int in_h = src.height();
	const int in_w = src.width();
	const int num = src.num();
	const int channels = src.channels();
	const int out_h = end_h - start_h;
	const int out_w = end_w - start_w;
	CHECK(out_h > 0) <<" end_h should be larger than start_h";
	CHECK(out_w > 0) <<" end_w should be larger than start_w";
	CHECK_LE(out_h ,in_h) <<" out_h should nor be larger than input_height";
	CHECK_LE(out_w ,in_w) <<" out_w should nor be larger than input_width";

	dst.Reshape(num,channels,out_h,out_w);
	if((out_h != in_h) || (out_w != in_w)){
		for(int n=0; n < num; n++)
		{
			for(int c=0; c<channels; c++)
			{
				Dtype* dst_data =dst.mutable_cpu_data() + dst.offset(n,c);
				const Dtype* src_data = src.cpu_data() + src.offset(n,c);

				for(int h=0; h< out_h; ++h)
				{
					const Dtype* src_data_p = src_data + (h+start_h)*in_w + start_w;
					Dtype* dst_data_p = dst_data+ h*out_w;
					for(int w=0; w<out_w;++w)
					{
						*(dst_data_p++)= *(src_data_p + w);
					}
				}
			}
		}
	}
	else
	{
		caffe::caffe_copy(src.count(),src.cpu_data(),dst.mutable_cpu_data());
	}
}

template void  CropBlobs_cpu( const Blob<float>&src,
		const int start_h, const int start_w,
				const int end_h, const int end_w, Blob<float>&dst);

template void  CropBlobs_cpu( const Blob<double>&src,
		const int start_h, const int start_w,
				const int end_h, const int end_w, Blob<double>&dst);



template <typename Dtype>
void CropBlobs_cpu( const Blob<Dtype>&src, const int src_num_id, const int start_h,
		const int start_w, const int end_h, const int end_w, Blob<Dtype>&dst,
		const int dst_num_id,const int dst_start_h  , const int dst_start_w  ){
	const int in_h = src.height();
	const int in_w = src.width();
	const int dst_w = dst.width();
	const int dst_h = dst.height();
	const int channels = src.channels();
	const int out_h = end_h - start_h;
	const int out_w = end_w - start_w;
	CHECK(out_h > 0) <<" end_h should be larger than start_h";
	CHECK(out_w > 0) <<" end_w should be larger than start_w";
//	CHECK(out_h <=in_h) <<" out_h should nor be larger than input_height";
//	CHECK(out_w <=in_w) <<" out_w should nor be larger than input_width";

	CHECK_GT(src.num(), src_num_id);
	CHECK_GT(dst.num(), dst_num_id);
	CHECK_EQ(channels, dst.channels());
//	CHECK_GE(dst.height(), end_h);
//	CHECK_GE(dst.width(), end_w);

	for(int c=0; c<channels; c++)
	{
		Dtype* dst_data =dst.mutable_cpu_data() + dst.offset(dst_num_id,c);
		const Dtype* src_data = src.cpu_data() + src.offset(src_num_id,c);


		for(int h=0; h< out_h; ++h)
		{
			int true_dst_h = h+dst_start_h;
			int true_src_h = h+start_h;
			if(true_dst_h >= 0 && true_dst_h < dst_h && true_src_h >= 0 && true_src_h < in_h)
			{
				int h_off_src = true_src_h*in_w;
				int h_off_dst = true_dst_h*dst_w;

				int true_dst_w =  dst_start_w;
				int true_src_w =  start_w;
				for(int w=0; w<out_w;++w)
				{
					if(true_dst_w >= 0 && true_dst_w < dst_w && true_src_w >= 0 && true_src_w < in_w)
						dst_data[h_off_dst + true_dst_w] = src_data[h_off_src+ true_src_w];
					++true_dst_w;
					++true_src_w;
				}
			}
		}
	}

}

template void CropBlobs_cpu( const Blob<float>&src, const int src_num_id, const int start_h,
		const int start_w, const int end_h, const int end_w, Blob<float>&dst,
		const int dst_num_id,const int dst_start_h , const int dst_start_w );

template void CropBlobs_cpu( const Blob<double>&src, const int src_num_id, const int start_h,
		const int start_w, const int end_h, const int end_w, Blob<double>&dst,
		const int dst_num_id,const int dst_start_h  , const int dst_start_w );

/**
 * src(n,c,h,w)  ===>   dst(n_ori,c,new_h,new_w)
 *
 */
template <typename Dtype>
void ConcateSubImagesInBlobs_cpu(const Blob<Dtype>& src,
		Blob<Dtype>& dst,const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w, const int stride_h,
	    const int stride_w, const int out_img_h, const int out_img_w)
{
	const int in_nums = src.num();


	const int height_col_ =(out_img_h + 2 * pad_h - kernel_h) / stride_h + 1;
	const int width_col_ = (out_img_w + 2 * pad_w - kernel_w) / stride_w + 1;

//	std::cout<<"in_nums:"<<in_nums<<" kernel_h:"<<kernel_h<<" kernel_w:"<<kernel_w
//			<<" pad_h:"<<pad_h<<" pad_w:"<<pad_w<<" stride_h:"<<stride_h<<
//			" stride_w:"<<stride_w<<"  out_img_h:"<<out_img_h <<" out_img_w:"<<out_img_w
//			<< " height_col:"<<height_col_<<" width_col:"<<width_col_<<std::endl;

	dst.Reshape(in_nums/height_col_/width_col_,src.channels(),  out_img_h, out_img_w);
//	std::cout<<"in_nums/height_col_/width_col_,src.channels(),  out_img_h, out_img_w: "<<
//			in_nums/height_col_/width_col_<< " "<<src.channels()<<"  "<<out_img_h<<"  "<<
//			out_img_w<<std::endl;
	const int channels_ = dst.channels();
	const int height_ = dst.height();
	const int width_ = dst.width();
	const int out_num = dst.num();

	for(int n = 0; n < out_num; n++){
			const Dtype*  src_data = src.cpu_data() + src.offset(n*height_col_*width_col_);
			Dtype*  dst_data = dst.mutable_cpu_data() + dst.offset(n);
			caffe::col2im_v2_cpu(src_data, channels_, height_,
		            width_, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
		            dst_data);
	}

	return;
}

template void ConcateSubImagesInBlobs_cpu(const Blob<float>& src,
		Blob<float>& dst,const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w, const int stride_h,
	    const int stride_w, const int out_img_h, const int out_img_w);

template void ConcateSubImagesInBlobs_cpu(const Blob<double>& src,
		Blob<double>& dst,const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w, const int stride_h,
	    const int stride_w, const int out_img_h, const int out_img_w);



template <typename Dtype>
void CropBlobs( const Blob<Dtype>&src, const int start_h,
		const int start_w, const int end_h, const int end_w, Blob<Dtype>&dst)
{
  switch (Caffe::mode()) {
	case Caffe::CPU:
	  CropBlobs_cpu(src,start_h, start_w,end_h, end_w,dst);
	  break;
	case Caffe::GPU:
		CropBlobs_gpu(src,start_h, start_w,end_h, end_w,dst);
	  break;
	default:
	  LOG(FATAL)<< "Unknown caffe mode.";
  }
}

template void  CropBlobs( const Blob<float>&src,
	const int start_h, const int start_w,
			const int end_h, const int end_w, Blob<float>&dst);

template void  CropBlobs( const Blob<double>&src,
	const int start_h, const int start_w,
			const int end_h, const int end_w, Blob<double>&dst);


template <typename Dtype>
void CropBlobs( const Blob<Dtype>&src, const int src_num_id, const int start_h,
		const int start_w, const int end_h, const int end_w, Blob<Dtype>&dst,
		const int dst_num_id,const int dst_start_h  , const int dst_start_w ){
  switch (Caffe::mode()) {
	case Caffe::CPU:
	  CropBlobs_cpu( src, src_num_id,  start_h,
				 start_w,  end_h, end_w, dst,
				 dst_num_id,dst_start_h  , dst_start_w );
	  break;
	case Caffe::GPU:
	  CropBlobs_gpu( src, src_num_id,  start_h,
					 start_w,  end_h, end_w, dst,
					 dst_num_id, dst_start_h  , dst_start_w );
	  break;
	default:
	  LOG(FATAL)<< "Unknown caffe mode.";
  }
}

template void CropBlobs( const Blob<float>&src, const int src_num_id, const int start_h,
		const int start_w, const int end_h, const int end_w, Blob<float>&dst,
		const int dst_num_id,const int dst_start_h  , const int dst_start_w );
template void CropBlobs( const Blob<double>&src, const int src_num_id, const int start_h,
		const int start_w, const int end_h, const int end_w, Blob<double>&dst,
		const int dst_num_id,const int dst_start_h  , const int dst_start_w );

template <typename Dtype>
void ResizeBlob(const Blob<Dtype>* src,
		Blob<Dtype>* dst)
{
	switch (Caffe::mode()) {
		case Caffe::CPU:
		  ResizeBlob_cpu(src,dst);
		  break;
		case Caffe::GPU:
		  ResizeBlob_gpu(src,dst);
		  break;
		default:
		  LOG(FATAL)<< "Unknown caffe mode.";
	}
}
template void ResizeBlob(const Blob<float>* src,Blob<float>* dst);
template void ResizeBlob(const Blob<double>* src,Blob<double>* dst);


template <typename Dtype>
Mat_<Dtype> Get_Affine_matrix(const Point_<Dtype>& srcCenter,
    const Point_<Dtype>& dstCenter,
    const Dtype alpha, const Dtype scale) {
  Mat_<Dtype> M(2, 3);

  M(0, 0) = scale * cos(alpha);
  M(0, 1) = scale * sin(alpha);
  M(1, 0) = -M(0, 1);
  M(1, 1) = M(0, 0);

  M(0, 2) = srcCenter.x - M(0, 0) * dstCenter.x - M(0, 1) * dstCenter.y;
  M(1, 2) = srcCenter.y - M(1, 0) * dstCenter.x - M(1, 1) * dstCenter.y;
  return M;
}

template Mat_<float> Get_Affine_matrix(const Point_<float>& srcCenter,
    const Point_<float>& dstCenter,
    const float alpha, const float scale);
template Mat_<double> Get_Affine_matrix(const Point_<double>& srcCenter,
    const Point_<double>& dstCenter,
    const double alpha, const double scale);

template <typename Dtype>
void mAffineWarp(const Mat_<Dtype>& M, const Mat& srcImg, Mat& dstImg,
    const bool fill_type, const uchar value) {
  if (dstImg.empty()) dstImg = cv::Mat(srcImg.size(), srcImg.type(), cv::Scalar::all(0));
  for (int y = 0; y < dstImg.rows; ++y) {
    for (int x = 0; x < dstImg.cols; ++x) {
      Dtype fx = M(0, 0) * x + M(0, 1) * y + M(0, 2);
      Dtype fy = M(1, 0) * x + M(1, 1) * y + M(1, 2);
      int sy = cvFloor(fy);
      int sx = cvFloor(fx);
      if (fill_type && (sy < 1 || sy > srcImg.rows - 2 || sx < 1 || sx > srcImg.cols - 2)) {
        for (int k = 0; k < srcImg.channels(); ++k) {
          dstImg.at<cv::Vec3b>(y, x)[k] = value;
        }
        continue;
      }

      fx -= sx;
      fy -= sy;

      sy = MAX(1, MIN(sy, srcImg.rows - 2));
      sx = MAX(1, MIN(sx, srcImg.cols - 2));
      Dtype w_y0 = std::abs(1.0f - fy);
      Dtype w_y1 = std::abs(fy);
      Dtype w_x0 = std::abs(1.0f - fx);
      Dtype w_x1 = std::abs(fx);
      for (int k = 0; k < srcImg.channels(); ++k) {
        dstImg.at<cv::Vec3b>(y, x)[k] = (srcImg.at<cv::Vec3b>(sy, sx)[k] * w_x0 * w_y0
            + srcImg.at<cv::Vec3b>(sy + 1, sx)[k] * w_x0 * w_y1
            + srcImg.at<cv::Vec3b>(sy, sx + 1)[k] * w_x1 * w_y0
            + srcImg.at<cv::Vec3b>(sy + 1, sx + 1)[k] * w_x1 * w_y1);
      }
    }
  }
}

template void mAffineWarp<float>(const Mat_<float>& M,
    const Mat& srcImg, Mat& dstImg,
    const bool fill_type, const uchar value);
template void mAffineWarp<double>(const Mat_<double>& M,
    const Mat& srcImg, Mat& dstImg,
    const bool fill_type, const uchar value);

float GetAffineImage_GetScale(const cv::Mat& src, cv::Mat& dst,
    const vector<float>& landmark,
    const AffineImageParameter& affine_param) {

//  CHECK_EQ(dst.cols, dst.rows) << "Only support square output";

  const float left_eye_x = landmark[0];
  const float left_eye_y = landmark[1];
  const float right_eye_x = landmark[2];
  const float right_eye_y = landmark[3];
  const float left_mouth_x = landmark[6];
  const float left_mouth_y = landmark[7];
  const float right_mouth_x = landmark[8];
  const float right_mouth_y = landmark[9];

  const float norm_standard_len = MAX(dst.rows, dst.cols) * affine_param.norm_ratio();
  float actual_len = norm_standard_len;
  switch (affine_param.norm_mode()) {
    case AVE_LE2LM_RE2RM: {

      const float deltaX1 = left_eye_x - left_mouth_x;
      const float deltaY1 = left_eye_y - left_mouth_y;

      const float deltaX2 = right_eye_x - right_mouth_x;
      const float deltaY2 = right_eye_y - right_mouth_y;

      actual_len = sqrt(deltaX1 * deltaX1 + deltaY1 * deltaY1)
          + sqrt(deltaX2 * deltaX2 + deltaY2 * deltaY2);
      actual_len /= 2;

      break;
    }

    case RECT_LE_RE_LM_RM: {
      const float left_top_x = MIN(MIN(MIN(left_eye_x, right_eye_x),
                                    left_mouth_x),
                                  right_mouth_x);
      const float right_bottom_x = MAX(MAX(MAX(left_eye_x, right_eye_x),
                                      left_mouth_x),
                                    right_mouth_x);

      const float left_top_y = MIN(MIN(MIN(left_eye_y, right_eye_y),
                                    left_mouth_y),
                                  right_mouth_y);
      const float right_bottom_y = MAX(MAX(MAX(left_eye_y, right_eye_y),
                                      left_mouth_y),
                                    right_mouth_y);

      const float deltaX = right_bottom_x - left_top_x;
      const float deltaY = right_bottom_y - left_top_y;
      actual_len = sqrt((deltaX * deltaX + deltaY * deltaY) / 2.f);

      break;
    }

    default:
      LOG(FATAL) << "Unknow Norm Mode";
  }

  const float scale = actual_len / norm_standard_len;
  return scale;
}

cv::Point_<float> GetAffineImage_GetSrcCenter(const vector<float>& landmark,
    const AffineImageParameter& affine_param) {

  cv::Point_<float> src_center;

  if (affine_param.center_ind_size() == 0 ||
      (affine_param.center_ind_size() == 1 && affine_param.center_ind(0) == -1)) {
    const float left_eye_x = landmark[0];
    const float left_eye_y = landmark[1];
    const float right_eye_x = landmark[2];
    const float right_eye_y = landmark[3];
    const float left_mouth_x = landmark[6];
    const float left_mouth_y = landmark[7];
    const float right_mouth_x = landmark[8];
    const float right_mouth_y = landmark[9];

    src_center.x = (left_eye_x + right_eye_x + left_mouth_x + right_mouth_x) / 4;
    src_center.y = (left_eye_y + right_eye_y + left_mouth_y + right_mouth_y) / 4;

  } else {
    src_center.x = 0;
    src_center.y = 0;
    for (int i = 0; i < affine_param.center_ind_size(); ++i) {
      src_center.x += landmark[affine_param.center_ind(i) * 2];
      src_center.y += landmark[affine_param.center_ind(i) * 2 + 1];
    }

    src_center.x /= affine_param.center_ind_size();
    src_center.y /= affine_param.center_ind_size();
  }

  return src_center;
}

float GetAffineImage_GetAngle(const vector<float>& landmark) {

  const float left_eye_x = landmark[0];
  const float left_eye_y = landmark[1];
  const float right_eye_x = landmark[2];
  const float right_eye_y = landmark[3];

  return atan2((right_eye_y - left_eye_y), (right_eye_x - left_eye_x));
}


void GetAffineImage(const cv::Mat& src, cv::Mat& dst,
    const vector<float>& landmark,
    const AffineImageParameter& affine_param) {

  const cv::Point_<float> src_center =
      GetAffineImage_GetSrcCenter(landmark, affine_param);

  const cv::Point_<float> dst_center(dst.cols / 2, dst.rows / 2);

  const float scale = GetAffineImage_GetScale(src, dst, landmark,
      affine_param);

  const float angle = GetAffineImage_GetAngle(landmark);

  const cv::Mat_<float> affine_mat = Get_Affine_matrix(
      src_center, dst_center, -angle, scale);

  mAffineWarp(affine_mat, src, dst, affine_param.fill_type(), affine_param.value());
}

float GetROCData(vector<pair<float, bool> >& pred_results,
    vector<vector<float> >& tp_fp_rates) {

  std::stable_sort(pred_results.begin(), pred_results.end());

  tp_fp_rates.clear();
  vector<float> point(5);

  int positve_count = 0;
  for (int i = 0; i < pred_results.size(); ++i) {
    if (pred_results[i].second) {
      positve_count += 1;
    }
  }
  const int negative_count = pred_results.size() - positve_count;

  int tp = 0;
  for (int i = 0; i < pred_results.size(); ++i) {
    if (pred_results[i].second) {
      tp += 1;
    }

    point[0] = tp;
    point[1] = point[0] / static_cast<float>(positve_count);
    point[2] = i + 1.0f - point[0];
    if (negative_count == 0) {
      point[3] = 0;
    } else {
      point[3] = point[2] / static_cast<float>(negative_count);
    }
    point[4] = pred_results[i].first;

    tp_fp_rates.push_back(point);
  }

  float auc = 0;
  if (tp_fp_rates.size() > 0) {
    auc = tp_fp_rates[0][1] * tp_fp_rates[0][3];
    for (int i = 1; i < pred_results.size(); ++i) {
      auc += tp_fp_rates[i][1] * (tp_fp_rates[i][3] - tp_fp_rates[i - 1][3]);
    }
  }

  return auc;
}

pair<float, float> GetBestAccuracy(vector<pair<float, bool> >& pred_results) {
  std::stable_sort(pred_results.begin(), pred_results.end());

  // ????????????????????????
  int total_neg_count = 0;
  for (int score_i = 0; score_i < pred_results.size(); ++score_i) {
    total_neg_count += (1 - pred_results[score_i].second);
  }
  // ??????????????????
  int best_correct_count = 0;
  int best_ind = -1;
  int tp = 0;
  int fn = 0;
  int cur_correct_count = 0;
  for (int score_i = 0; score_i < pred_results.size(); ++score_i) {
    cur_correct_count = total_neg_count - fn + tp;
    if (cur_correct_count > best_correct_count) {
      best_correct_count = cur_correct_count;
      best_ind = score_i;
    }

    if (pred_results[score_i].second) {
      ++tp;
    } else {
      ++fn;
    }
  }
  cur_correct_count = total_neg_count - fn + tp;
  if (cur_correct_count > best_correct_count) {
    best_correct_count = cur_correct_count;
    best_ind = pred_results.size();
  }

  pair<float, float> ret;
  ret.first = best_correct_count / static_cast<float>(pred_results.size());
  ret.second = (best_ind == pred_results.size() ?
      (pred_results.back().first + ELLISION) : pred_results[best_ind].first);

  return ret;
}

void ReadAnnotations(std::string sourcefile, vector<pair<string, vector<float> > > & samples ,int key_point_count)
{
	LOG(INFO) << "Opening file " << sourcefile; 
	std::ifstream infile(sourcefile.c_str());
	std::string filename;
	float lmk;
	samples.clear();
	
	
	while (infile >> filename)
	{
		std::vector< float > lmks;
		lmks.clear();
		for(int i=0;i<key_point_count*2;i++)
		{
			infile >> lmk;
			lmks.push_back(lmk);
     // LOG(INFO)<<filename<<" "<<lmk;
		}
		samples.push_back(std::make_pair(filename, lmks));
	}

	infile.close();
}

void ReadAnnotations(std::string sourcefile, std::string lmkfile, vector<pair<string, vector<float> > > & samples ,int key_point_count)
{
	LOG(INFO) << "Opening file " << sourcefile; 
	std::ifstream imagefile(sourcefile.c_str());
	std::ifstream landmarkfile(lmkfile.c_str());
	std::string filename;
	float lmk;
	samples.clear();
	
	
	while (imagefile >> filename)
	{
		std::vector< float > lmks;
		lmks.clear();
		for(int i=0;i<key_point_count;i++)
		{
			landmarkfile >> lmk;
			lmks.push_back(lmk);
		}
		samples.push_back(std::make_pair(filename, lmks));
	}

	imagefile.close();
	landmarkfile.close();
}

float GetL2Distance(const vector<float>& fea1, const vector<float>& fea2) {
  float total_diff = 0;
  for (int fea_i = 0; fea_i < fea1.size(); ++fea_i) {
    float diff = fea1[fea_i] - fea2[fea_i];
    diff *= diff;

    total_diff += diff;
  }

  return total_diff;
}

void mAffineWarp(const Mat_<float> M, const Mat& srcImg,Mat& dstImg,int interpolation)
{
    if(dstImg.empty())
        dstImg = Mat(srcImg.size(), srcImg.type());
    dstImg.setTo(0);

    for (int y=0; y<dstImg.rows; ++y)
    {
        for (int x=0; x<dstImg.cols; ++x)
        {
            float fx = M(0,0)*x + M(0,1)*y + M(0,2);
            float fy = M(1,0)*x + M(1,1)*y + M(1,2);

            int sy  = cvFloor(fy);
            int sx  = cvFloor(fx);
            fx -= sx;
            fy -= sy;

            //if(sy<1 ||sy>srcImg.rows-2 || sx<1 || sx>srcImg.cols-2)
            //    continue;
            //sy = max(1, min(sy, srcImg.rows-2)); //my modify
            //sx = max(1, min(sx, srcImg.cols-2)); //my modify

            float w_y0 = abs(1.0f - fy);
            float w_y1 = abs(fy);
            float w_x0 = abs(1.0f-fx);
            float w_x1 = abs(fx);
            if(srcImg.channels()==1)
            {
                if(interpolation ==INTER_NEAREST)
                {
                    if(sy<1 ||sy>srcImg.rows-2 || sx<1 || sx>srcImg.cols-2)
                        continue;
                    dstImg.at<uchar>(y, x) = srcImg.at<uchar>(sy, sx);
                }
                else
                {
                    sy = max(1, min(sy, srcImg.rows-2)); //my modify
                    sx = max(1, min(sx, srcImg.cols-2)); //my modify
                    dstImg.at<uchar>(y, x) = (srcImg.at<uchar>(sy, sx) * w_x0 * w_y0 + 
                            srcImg.at<uchar>(sy+1, sx) * w_x0 * w_y1 +
                            srcImg.at<uchar>(sy, sx+1) * w_x1 *w_y0 + 
                            srcImg.at<uchar>(sy+1, sx+1) * w_x1 * w_y1);
                }
            }
            else
            {
                if(interpolation ==INTER_NEAREST)
                {
                    if(sy<1 ||sy>srcImg.rows-2 || sx<1 || sx>srcImg.cols-2)
                        continue;
                    for (int k=0; k<srcImg.channels(); ++k)
                        dstImg.at<cv::Vec3b>(y, x)[k] = srcImg.at<cv::Vec3b>(sy, sx)[k];
                }
                else
                {
                    //sy = max(1, min(sy, srcImg.rows-2)); //my modify
                    //sx = max(1, min(sx, srcImg.cols-2)); //my modify
                    if(sy<1 ||sy>srcImg.rows-2 || sx<1 || sx>srcImg.cols-2)
                        continue;
                    for (int k=0; k<srcImg.channels(); ++k)
                    {
                        dstImg.at<cv::Vec3b>(y, x)[k] = (srcImg.at<cv::Vec3b>(sy, sx)[k] * w_x0 * w_y0 +
                                srcImg.at<cv::Vec3b>(sy+1, sx)[k] * w_x0 * w_y1 +
                                srcImg.at<cv::Vec3b>(sy, sx+1)[k] * w_x1 *w_y0 +
                                srcImg.at<cv::Vec3b>(sy+1, sx+1)[k] * w_x1 * w_y1);
                    }
                }
            }
        }
    }
}

//grid based disturbance

void disturb_Triangle(Mat& img,Mat& dstImg, Mat& labelImg, Mat& dstlabelImg,int blockx, int blocky, float distrub_radius_ratio,  int interpolation)
{
    if(dstImg.rows!= img.rows || dstImg.cols!= img.cols  || dstImg.channels()!= img.channels())
        dstImg.create(img.size(),img.type());
    Mat temp_labelImg;
    if(labelImg.rows!= img.rows || labelImg.cols!= img.cols )
        resize(img,temp_labelImg,img.size(),0,0,INTER_NEAREST);
    else
        temp_labelImg = labelImg;

    Mat  temp_dstlabelImg(img.size(),labelImg.type());

    int imgw = img.cols;
    int imgh = img.rows;

    int stepx = imgw / blockx;
    int stepy=imgh / blocky;

    int radius_x = stepx*distrub_radius_ratio;
    int radius_y = stepy*distrub_radius_ratio;

    /*generate the rect gride and distrubed gride*/
    vector<Point2f> src_pts;
    vector<Point2f> dst_pts;
    for(int j=0;j<=blocky;++j)
    {
        int y =  j < blocky ?  j*stepy : imgh-1;
        for(int i=0;i<=blockx;++i)
        {
            int x = i < blockx ? i * stepx : imgw-1;
            Point2f tmp_pt(x,y);
            src_pts.push_back(tmp_pt);
            if(i!=0 &&  i!=blockx && j!=0 && j!=blocky)
            {
                tmp_pt.x += (float(rand()) / RAND_MAX - 0.5)*radius_x;
                tmp_pt.y += (float(rand()) / RAND_MAX - 0.5)*radius_y;
            }
            dst_pts.push_back(tmp_pt);
        }
    }

    /*get triangles*/
    vector<Triangle> triangles;
    for(int j=0;j<blocky;++j)
    {
        for(int i =0;i<blockx;++i)
        {
            int s = j * (blockx + 1) + i;
            triangles.push_back(Triangle(s, s+blockx+1, s+blockx+1+1));
            triangles.push_back(Triangle(s, s+1, s+blockx+1+1));
        }
    }

    /*warping img*/
    for(int r=0;r<img.rows;++r)
    {
        for(int c=0;c<img.cols;++c)
        {
            /*get triangle id*/
            int grid_x = c / stepx < blockx ? c / stepx : blockx-1; 
            int grid_y = r / stepy < blocky ?  r / stepy : blocky-1;
            int tri_id = grid_y*blockx*2 + grid_x*2 + ( (c-grid_x*stepx)<= (float)stepx / float(stepy)*(r - grid_y*stepy)? 0 :1); 
            float fx,fy;
            TriWarp(c,r,
                    src_pts[triangles[tri_id].i].x,src_pts[triangles[tri_id].i].y, 
                    src_pts[triangles[tri_id].j].x,src_pts[triangles[tri_id].j].y,
                    src_pts[triangles[tri_id].k].x,src_pts[triangles[tri_id].k].y,
                    fx,fy,
                    dst_pts[triangles[tri_id].i].x,dst_pts[triangles[tri_id].i].y,
                    dst_pts[triangles[tri_id].j].x,dst_pts[triangles[tri_id].j].y,
                    dst_pts[triangles[tri_id].k].x,dst_pts[triangles[tri_id].k].y);


            int sy  = cvFloor(fy);
            int sx  = cvFloor(fx);
            fx -= sx;
            fy -= sy;

            /*if(sy<1 ||sy>img.rows-2 || sx<1 || sx>img.cols-2)
             *                          continue;*/
            sy = max(1, min(sy, img.rows-2)); /*my modify*/
            sx = max(1, min(sx, img.cols-2)); /*my modify*/

            float w_y0 = abs(1.0f - fy);
            float w_y1 = abs(fy);
            float w_x0 = abs(1.0f-fx);
            float w_x1 = abs(fx);

            /*warping img*/
            if(img.channels()==1)
            {
                if(interpolation ==INTER_NEAREST)
                    dstImg.at<uchar>(r, c) = img.at<uchar>(sy, sx);
                else
                {
                    dstImg.at<uchar>(r, c) = (img.at<uchar>(sy, sx) * w_x0 * w_y0 + 
                            img.at<uchar>(sy+1, sx) * w_x0 * w_y1 +
                            img.at<uchar>(sy, sx+1) * w_x1 *w_y0 + 
                            img.at<uchar>(sy+1, sx+1) * w_x1 * w_y1);
                }
            }
            else
            {
                if(interpolation ==INTER_NEAREST)
                {
                    for (int k=0; k<img.channels(); ++k)
                        dstImg.at<cv::Vec3b>(r, c)[k] = img.at<cv::Vec3b>(sy, sx)[k];
                }
                else
                {
                    for (int k=0; k<img.channels(); ++k)
                    {
                        dstImg.at<cv::Vec3b>(r, c)[k] = (img.at<cv::Vec3b>(sy, sx)[k] * w_x0 * w_y0 +
                                img.at<cv::Vec3b>(sy+1, sx)[k] * w_x0 * w_y1 +
                                img.at<cv::Vec3b>(sy, sx+1)[k] * w_x1 *w_y0 +
                                img.at<cv::Vec3b>(sy+1, sx+1)[k] * w_x1 * w_y1);
                    }
                }
            }

            /*warping label  img*/
            if(temp_labelImg.channels()==1)
                temp_dstlabelImg.at<uchar>(r, c) = temp_labelImg.at<uchar>(sy, sx);
            else
            {
                for (int k=0; k<temp_dstlabelImg.channels(); ++k)
                    temp_dstlabelImg.at<cv::Vec3b>(r, c)[k] = temp_labelImg.at<cv::Vec3b>(sy, sx)[k];
            }
        }
    }

    resize(temp_dstlabelImg,dstlabelImg,labelImg.size(),0,0,INTER_NEAREST);
}

cv::Mat get_outputmap(std::vector<float> data,int out_idx,int outheight,int outwidth,int channels,bool auto_scale=false)
{
  int spacedim = outheight*outwidth;
  int count = spacedim * channels;
  const float * outdata= &data[out_idx * count];
  cv::Mat result = cv::Mat(outheight,outwidth,CV_8UC1);

  float maxv=-FLT_MAX;
  int maxid=0;
  float v=0;
  
  int scale_rate=1;
  if(auto_scale)
  {
    scale_rate = 255/(channels-1);
  }
  
  for(int h=0;h<outheight;h++)
  {
    //unsigned char * pdata = result.ptr<unsigned char>(h);
    for(int w=0;w<outwidth;w++)
    {
         
        for(int c=0;c<channels;c++)
        {
          v=outdata[c*spacedim + h* outwidth + w];
          if (v > maxv)
          {
            maxv = v;
            maxid = c;
          }
        }
        if(auto_scale)
        {
            maxid = maxid * scale_rate;
        }
        result.at<unsigned char>(h, w)=(unsigned char)(maxid);
        maxv=-FLT_MAX;
        maxid=0;
    }
  }
  return result;
}

}
// namespace caffe
