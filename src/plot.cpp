#include "pcicf.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace pcicf {
void save_morton_plot(const fs::path& path,const std::vector<Row>& rows) {
    cv::Mat image(640,1000,CV_8UC3,cv::Scalar(255,255,255));
    const cv::Scalar black(30,30,30), gray(225,225,225), red(0,0,255), green(0,128,0);
    const int left=110,right=880,top=55,bottom=555;
    const auto text=[&](const std::string& s,cv::Point p,double scale=0.48) {
        cv::putText(image,s,p,cv::FONT_HERSHEY_SIMPLEX,scale,black,1,cv::LINE_AA);
    };
    double xmin=0,xmax=1,fmin=0,fmax=1;
    if (!rows.empty()) {
        xmin=xmax=double(rows.front().morton);
        fmin=fmax=double(rows.front().frame_id);
        for (const auto& row : rows) {
            xmin=std::min(xmin,double(row.morton)); xmax=std::max(xmax,double(row.morton));
            fmin=std::min(fmin,double(row.frame_id)); fmax=std::max(fmax,double(row.frame_id));
        }
        const double xpad=xmax>xmin?(xmax-xmin)*0.05:std::max(0.5,std::abs(xmin)*0.05);
        const double fpad=fmax>fmin?(fmax-fmin)*0.05:std::max(0.5,std::abs(fmin)*0.05);
        xmin-=xpad; xmax+=xpad; fmin-=fpad; fmax+=fpad;
    }
    const auto xpixel=[&](double value) {return left+int(std::lround((value-xmin)/(xmax-xmin)*(right-left)));};
    const auto fpixel=[&](double value) {return bottom-int(std::lround((value-fmin)/(fmax-fmin)*(bottom-top)));};
    for (int i=0;i<=5;++i) {
        const int x=left+(right-left)*i/5;
        cv::line(image,{x,top},{x,bottom},gray,1);
        const double value=xmin+(xmax-xmin)*i/5;
        std::ostringstream label;
        if (std::abs(value)>=1000) label<<std::fixed<<std::setprecision(0)<<value;
        else label<<std::setprecision(4)<<value;
        const auto sz=cv::getTextSize(label.str(),cv::FONT_HERSHEY_SIMPLEX,0.48,1,nullptr);
        text(label.str(),{x-sz.width/2,bottom+25});
    }
    for (int i=0;i<=4;++i) {
        const int y=bottom-(bottom-top)*i/4;
        cv::line(image,{left,y},{right,y},gray,1);
        std::ostringstream frequency, frame;
        frequency<<i/4.0;
        frame<<std::fixed<<std::setprecision(1)<<(fmin+(fmax-fmin)*i/4);
        text(frequency.str(),{left-42,y+5}); text(frame.str(),{right+9,y+5});
    }
    for (const auto& row : rows) {
        const int x=xpixel(double(row.morton));
        cv::line(image,{x,top},{x,bottom},red,1,cv::LINE_AA);
    }
    for (const auto& row : rows)
        cv::circle(image,{xpixel(double(row.morton)),fpixel(double(row.frame_id))},2,green,cv::FILLED,cv::LINE_AA);
    cv::rectangle(image,{left,top},{right,bottom},black,1);
    text("Morton",{(left+right)/2-32,615},0.7);
    const auto vertical=[&](const std::string& label,int x,bool clockwise) {
        cv::Mat word(34,210,CV_8UC3,cv::Scalar(255,255,255)),rotated;
        cv::putText(word,label,{6,24},cv::FONT_HERSHEY_SIMPLEX,0.7,black,1,cv::LINE_AA);
        cv::rotate(word,rotated,clockwise?cv::ROTATE_90_CLOCKWISE:cv::ROTATE_90_COUNTERCLOCKWISE);
        rotated.copyTo(image(cv::Rect(x,(top+bottom-rotated.rows)/2,rotated.cols,rotated.rows)));
    };
    vertical("Frequency",20,false); vertical("Frame Number",958,true);
    if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
    if (!cv::imwrite(path.string(),image)) throw std::runtime_error("Cannot save plot: "+path.string());
}
} // namespace pcicf
