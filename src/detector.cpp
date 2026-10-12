#include "pcicf.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace pcicf {
struct Detector::Impl {
    cv::dnn::Net net;
    int width, height;
    Impl(const fs::path& model, int w, int h) : net(cv::dnn::readNetFromONNX(model.string())), width(w), height(h) {
        if (w <= 0 || h <= 0) throw std::runtime_error("Model dimensions must be positive");
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    std::vector<Detection> infer(const cv::Mat& image) {
        if (image.empty()) throw std::runtime_error("Empty detector image");
        const double gain=std::min(double(width)/image.cols,double(height)/image.rows);
        const int resized_w=int(std::nearbyint(image.cols*gain));
        const int resized_h=int(std::nearbyint(image.rows*gain));
        const int left=int(std::nearbyint((width-resized_w)/2.0-0.1));
        const int top=int(std::nearbyint((height-resized_h)/2.0-0.1));
        cv::Mat resized, padded;
        cv::resize(image,resized,{resized_w,resized_h},0,0,cv::INTER_LINEAR);
        cv::copyMakeBorder(resized,padded,top,height-resized_h-top,left,width-resized_w-left,
                           cv::BORDER_CONSTANT,{114,114,114});
        net.setInput(cv::dnn::blobFromImage(padded,1.0/255.0,{width,height},{0,0,0},true,false,CV_32F));
        std::vector<cv::Mat> outputs;
        net.forward(outputs,net.getUnconnectedOutLayersNames());
        if (outputs.size()!=1) throw std::runtime_error("Expected one raw YOLO11 output. Export with nms=False");
        const auto& raw=outputs[0];
        if (raw.dims!=3 || raw.size[0]!=1 || raw.depth()!=CV_32F)
            throw std::runtime_error("Expected float32 YOLO11 output of shape [1,84,N]");
        cv::Mat predictions;
        if (raw.size[1]==84) {
            cv::Mat channels(84,raw.size[2],CV_32F,const_cast<float*>(raw.ptr<float>()));
            cv::transpose(channels,predictions);
        } else if (raw.size[2]==84) {
            predictions=cv::Mat(raw.size[1],84,CV_32F,const_cast<float*>(raw.ptr<float>()));
        } else throw std::runtime_error("Expected COCO YOLO11 (80 classes, no objectness channel)");
        std::vector<Detection> candidates;
        for (int i=0;i<predictions.rows;++i) {
            const auto* row=predictions.ptr<float>(i);
            const auto* best=std::max_element(row+4,row+84);
            if (*best<=0.1f || !std::isfinite(*best) || row[2]<=0 || row[3]<=0) continue;
            candidates.push_back({{row[0]-row[2]/2,row[1]-row[3]/2,row[2],row[3]},*best,int(best-row-4)});
        }
        std::stable_sort(candidates.begin(),candidates.end(),[](const auto& a,const auto& b){return a.score>b.score;});
        if (candidates.size()>30000) candidates.resize(30000);
        std::vector<Detection> kept;
        for (const auto& d : candidates) {
            bool suppressed=false;
            for (const auto& previous : kept) {
                if (d.cls!=previous.cls) continue;
                const float w=std::max(0.0f,std::min(d.box.x+d.box.width,previous.box.x+previous.box.width)-std::max(d.box.x,previous.box.x));
                const float h=std::max(0.0f,std::min(d.box.y+d.box.height,previous.box.y+previous.box.height)-std::max(d.box.y,previous.box.y));
                const float intersection=w*h;
                if (intersection/(d.box.area()+previous.box.area()-intersection)>0.7f) {suppressed=true;break;}
            }
            if (!suppressed) kept.push_back(d);
            if (kept.size()==300) break;
        }
        // Ultralytics applies NMS in model coordinates, then scales/clips boxes.
        for (auto& d : kept) {
            const float x1=std::clamp(float((d.box.x-left)/gain),0.0f,float(image.cols));
            const float y1=std::clamp(float((d.box.y-top)/gain),0.0f,float(image.rows));
            const float x2=std::clamp(float((d.box.x+d.box.width-left)/gain),0.0f,float(image.cols));
            const float y2=std::clamp(float((d.box.y+d.box.height-top)/gain),0.0f,float(image.rows));
            d.box={x1,y1,x2-x1,y2-y1};
        }
        return kept;
    }
};
Detector::Detector(const fs::path& p,int w,int h):impl_(std::make_unique<Impl>(p,w,h)){}
Detector::~Detector()=default;
std::vector<Detection> Detector::infer(const cv::Mat& image){return impl_->infer(image);}
} // namespace pcicf
