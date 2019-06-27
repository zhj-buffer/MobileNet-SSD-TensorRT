#include "common.h"
#include "cudaUtility.h"
#include "mathFunctions.h"
#include "pluginImplement.h"
#include "tensorNet.h"
#include "util/loadImage.h"
#include "imageBuffer.h"
#include <chrono>
#include <thread>

#include <librealsense2/rs.hpp>
#include "motor/main.h"
#include "cv-helpers.hpp"

using namespace cv;
using namespace rs2;



const char* model  = "../../model/MobileNetSSD_deploy_iplugin.prototxt";
const char* weight = "../../model/MobileNetSSD_deploy.caffemodel";

const char* INPUT_BLOB_NAME = "data";

const char* OUTPUT_BLOB_NAME = "detection_out";
static const uint32_t BATCH_SIZE = 1;

//image buffer size = 10
//dropFrame = false
ConsumerProducerQueue<cv::Mat> *imageBuffer = new ConsumerProducerQueue<cv::Mat>(10,false);

class Timer {
public:
    void tic() {
        start_ticking_ = true;
        start_ = std::chrono::high_resolution_clock::now();
    }
    void toc() {
        if (!start_ticking_)return;
        end_ = std::chrono::high_resolution_clock::now();
        start_ticking_ = false;
        t = std::chrono::duration<double, std::milli>(end_ - start_).count();
        //std::cout << "Time: " << t << " ms" << std::endl;
    }
    double t;
private:
    bool start_ticking_ = false;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_;
};


/* *
 * @TODO: unifiedMemory is used here under -> ( cudaMallocManaged )
 * */
float* allocateMemory(DimsCHW dims, char* info)
{
    float* ptr;
    size_t size;
    std::cout << "Allocate memory: " << info << std::endl;
    size = BATCH_SIZE * dims.c() * dims.h() * dims.w();
    assert(!cudaMallocManaged( &ptr, size*sizeof(float)));
    return ptr;
}


void loadImg( cv::Mat &input, int re_width, int re_height, float *data_unifrom,const float3 mean,const float scale )
{
    int i;
    int j;
    int line_offset;
    int offset_g;
    int offset_r;
    cv::Mat dst;

    unsigned char *line = NULL;
    float *unifrom_data = data_unifrom;

    cv::resize( input, dst, cv::Size( re_width, re_height ), (0.0), (0.0), cv::INTER_LINEAR );
    offset_g = re_width * re_height;
    offset_r = re_width * re_height * 2;
    for( i = 0; i < re_height; ++i )
    {
        line = dst.ptr< unsigned char >( i );
        line_offset = i * re_width;
        for( j = 0; j < re_width; ++j )
        {
            // b
            unifrom_data[ line_offset + j  ] = (( float )(line[ j * 3 ] - mean.x) * scale);
            // g
            unifrom_data[ offset_g + line_offset + j ] = (( float )(line[ j * 3 + 1 ] - mean.y) * scale);
            // r
            unifrom_data[ offset_r + line_offset + j ] = (( float )(line[ j * 3 + 2 ] - mean.z) * scale);
        }
    }
}

//thread read video
void readPicture()
{
    cv::VideoCapture cap("../../testVideo/test.avi");
    cv::Mat image;
    while(cap.isOpened())
    {
        cap >> image;
        imageBuffer->add(image);
    }
}

const size_t inWidth      = 300;
const size_t inHeight     = 300;
const float inScaleFactor = 0.007843f;
const float meanVal       = 127.5;
const float WHRatio       = inWidth / (float)inHeight;
const char* classNames[]  = {"background",
                             "aeroplane", "bird", "person", "boat",
                             "bottle", "bus", "car", "cat", "chair",
                             "cow", "diningtable", "dog", "horse",
                             "motorbike", "bicycle", "pottedplant",
                             "sheep", "sofa", "train", "tvmonitor"};

int main(int argc, char *argv[])
{

    move_init();
    
    // Start streaming from Intel RealSense Camera
    pipeline pipe;
    auto config = pipe.start();
    auto profile = config.get_stream(RS2_STREAM_COLOR)
	    .as<video_stream_profile>();
    rs2::align align_to(RS2_STREAM_COLOR);

    Size cropSize;
    if (profile.width() / (float)profile.height() > WHRatio)
    {
	    cropSize = Size(static_cast<int>(profile.height() * WHRatio),
			    profile.height());
    }
    else
    {
	    cropSize = Size(profile.width(),
			    static_cast<int>(profile.width() / WHRatio));
    }

    Rect crop(Point((profile.width() - cropSize.width) / 2,
			    (profile.height() - cropSize.height) / 2),
		    cropSize);

    const auto window_name = "Display Image";
    namedWindow(window_name, WINDOW_AUTOSIZE);


    std::vector<std::string> output_vector = {OUTPUT_BLOB_NAME};
    TensorNet tensorNet;
    tensorNet.LoadNetwork(model,weight,INPUT_BLOB_NAME, output_vector,BATCH_SIZE);

    DimsCHW dimsData = tensorNet.getTensorDims(INPUT_BLOB_NAME);
    DimsCHW dimsOut  = tensorNet.getTensorDims(OUTPUT_BLOB_NAME);

    float* data    = allocateMemory( dimsData , (char*)"input blob");
    std::cout << "allocate data" << std::endl;
    float* output  = allocateMemory( dimsOut  , (char*)"output blob");
    std::cout << "allocate output" << std::endl;
    int height = 300;
    int width  = 300;

    cv::Mat frame,srcImg;

    void* imgCPU;
    void* imgCUDA;
    Timer timer;


#if 0

//    std::string imgFile = "../../testPic/test.jpg";
//    frame = cv::imread(imgFile);
    std::thread readTread(readPicture);
    readTread.detach();

    cv::VideoCapture capture;
    capture.open(0);//open 根据编号打开摄像头
    std::cout<<"-------------"<<std::endl;
    if (!capture.isOpened())
    {
        std::cout << "Read video Failed !" << std::endl;
        return 0;
    }

#endif
    int rw = 300;
    int rh = 300;

    const size_t rsize = rw * rh * sizeof(float3);

    const size_t size = width * height * sizeof(float3);
    if( CUDA_FAILED( cudaMalloc( &imgCUDA, rsize)) )
    {
        cout <<"Cuda Memory allocation error occured."<<endl;
        return false;
    }
    void* imgData = malloc(rsize);
    memset(imgData,0,rsize);

    void* buffers[] = { imgCUDA, output };
    vector<vector<float> > detections;

    //while(1){
    while (getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0)
    {

	    // Wait for the next set of frames
	    auto data = pipe.wait_for_frames();
	    // Make sure the frames are spatially aligned
	    data = align_to.process(data);

	    auto color_frame = data.get_color_frame();
	    auto depth_frame = data.get_depth_frame();

	    // If we only received new depth frame, 
	    // but the color did not update, continue
	    static int last_frame_number = 0;
	    if (color_frame.get_frame_number() == last_frame_number) continue;
	    last_frame_number = color_frame.get_frame_number();

	    printf("%s, %s, %d\n", __FILE__, __func__, __LINE__);
	    // Convert RealSense frame to OpenCV matrix:
	    auto color_mat = frame_to_mat(color_frame);
	    auto depth_mat = depth_frame_to_meters(pipe, depth_frame);

	    // Crop both color and depth frames
	    color_mat = color_mat(crop);
	    depth_mat = depth_mat(crop);

	    printf("%s, %s, %d, color_mat.cols: %d, color_mat.rows: %d \n", __FILE__, __func__, __LINE__, color_mat.cols, color_mat.rows);

	
	cv::imshow("test1", color_mat);
//	cv::waitKey(1);
	
	frame = color_mat.clone();
        //imageBuffer->consume(frame);
        //capture >> frame;

	//srcImg = frame.clone();
        //cv::resize(frame, frame, cv::Size(300,300));
        cv::resize(frame, frame, cv::Size(rw, rh));

  
        loadImg(frame,rh,rw,(float*)imgData,make_float3(127.5,127.5,127.5),0.007843);
        cudaMemcpyAsync(imgCUDA,imgData,size,cudaMemcpyHostToDevice);

//        void* buffers[] = { imgCUDA, output };

        timer.tic();
        tensorNet.imageInference( buffers, output_vector.size() + 1, BATCH_SIZE);
        timer.toc();
        double msTime = timer.t;

//        vector<vector<float> > detections;

        for (int k=0; k<100; k++)
        {
            if(output[7*k+1] == -1)
                break;
            float classIndex = output[7*k+1];
            float confidence = output[7*k+2];
            float xmin = output[7*k + 3];
            float ymin = output[7*k + 4];
            float xmax = output[7*k + 5];
            float ymax = output[7*k + 6];
            std::cout << classIndex << " , " << confidence << " , "  << xmin << " , " << ymin<< " , " << xmax<< " , " << ymax << std::endl;
            int x1 = static_cast<int>(xmin * color_mat.cols);
            int y1 = static_cast<int>(ymin * color_mat.rows);
            int x2 = static_cast<int>(xmax * color_mat.cols);
            int y2 = static_cast<int>(ymax * color_mat.rows);
            cv::rectangle(color_mat,cv::Rect2f(cv::Point(x1,y1),cv::Point(x2,y2)),cv::Scalar(255,0,255),1);

	    Rect object((int)x1, (int)y1,
			    (int)(x2 - x1),
			    (int)(y2 - y1));

	    object = object  & Rect(0, 0, depth_mat.cols, depth_mat.rows);

	                    // Calculate mean depth inside the detection region
                // This is a very naive way to estimate objects depth
                // but it is intended to demonstrate how one might
                // use depht data in general
                Scalar m = mean(depth_mat(object));

                std::ostringstream ss;
                ss << classNames[static_cast<int>(classIndex)] << " ";
                ss << std::setprecision(2) << m[0] << " meters away";
                //ss << object << "";
                String conf(ss.str());

		std::cout << "distance: " << ss.str() << std::endl;

		int baseLine = 0;
		Size labelSize = getTextSize(ss.str(), FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

		auto center = (object.br() + object.tl())*0.5;
                std::cout<< "center: " << center << ".x" << center.x  << std::endl;

		printf("classIdndx: %d \n", static_cast<int>(classIndex));

		int dis = 100;
		int dis_l = (color_mat.cols - 100) / 2;
		int dis_r = dis_l + dis;

		center.x = center.x - labelSize.width / 2;
		std::cout<< "center: " << center << std::endl;


#if 0
		if (classIndex == 3) {
			//if (objectClass == 5) {
			if (center.x > dis_r) {
				std::cout << "move left " << std::endl;
				move_left(50);
			}
			if (center.x < dis_l) {
				std::cout << "move right " << std::endl;
				move_right(50);
			}
			if (center.x < dis_r && center.x > dis_l) {
				std::cout << "move stop " << std::endl;
				move_stop();
			}
		} else {
			move_stop();
		}
#endif
		rectangle(color_mat, Rect(Point(center.x, center.y - labelSize.height),
					Size(labelSize.width, labelSize.height + baseLine)),
				Scalar(255, 255, 255), FILLED);
		putText(color_mat, ss.str(), center,
				FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0));


        }
        std::cout << "time: " << msTime << std::endl;
        cv::imshow("DetectNet", color_mat);
        cv::waitKey(1);
    }
    free(imgData);
    cudaFree(imgCUDA);
    cudaFreeHost(imgCPU);
    cudaFree(output);
    tensorNet.destroy();
    return 0;
}
