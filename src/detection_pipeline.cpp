#include "pcicf.hpp"
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <opencv2/imgcodecs.hpp>

using namespace pcicf;
static void usage() {
    std::cout<<"Usage: detection_pipeline -d pie|MoreSMIRK -o OUTPUT [-s] [options]\n"
        "  -s, --save_yolo_result       Save bounding-box visualizations\n"
        "  --config PATH               Config JSON (default: config.json)\n"
        "  --model PATH                YOLO11x ONNX (default: yolo11x.onnx)\n"
        "  --model-width N             Model input width (default: 640)\n"
        "  --model-height N            Model input height (default: 480)\n"
        "  --gmc sparseOptFlow|none     Motion compensation (default: sparseOptFlow)\n"
        "  --tracks PATH               Replay tracked boxes from JSON\n"
        "  --detections PATH           Replay detections through native tracker\n"
        "  --dump-tracks               Save tracked boxes beside CSV files\n"
        "  -h, --help                  Print this help\n";
}
static Options parse(int argc,char** argv) {
    Options o;
    const auto value=[&](int& i) {
        if (++i>=argc) throw std::runtime_error("Missing option value");
        return std::string(argv[i]);
    };
    for (int i=1;i<argc;++i) {
        const std::string arg=argv[i];
        if (arg=="-d"||arg=="--dataset") o.dataset=value(i);
        else if (arg=="-o"||arg=="--output_path") o.output_path=value(i);
        else if (arg=="-s"||arg=="--save_yolo_result") o.save_yolo_result=true;
        else if (arg=="--config") o.config_path=value(i);
        else if (arg=="--model") o.model_path=value(i);
        else if (arg=="--model-width") o.model_width=std::stoi(value(i));
        else if (arg=="--model-height") o.model_height=std::stoi(value(i));
        else if (arg=="--tracks") o.tracks_path=value(i);
        else if (arg=="--detections") o.detections_path=value(i);
        else if (arg=="--dump-tracks") o.dump_tracks=true;
        else if (arg=="--gmc") {
            const auto mode=value(i);
            if (mode!="sparseOptFlow"&&mode!="none") throw std::runtime_error("Unsupported GMC mode");
            o.gmc=mode!="none";
        } else if (arg=="--help"||arg=="-h") {usage();std::exit(0);}
        else throw std::runtime_error("Unknown option: "+arg);
    }
    if (o.dataset!="pie"&&o.dataset!="MoreSMIRK") throw std::runtime_error("-d must be pie or MoreSMIRK");
    if (o.output_path.empty()) throw std::runtime_error("-o is required");
    if (!o.tracks_path.empty()&&!o.detections_path.empty()) throw std::runtime_error("Choose --tracks or --detections");
    if (o.model_width<=0||o.model_height<=0) throw std::runtime_error("Invalid model dimensions");
    return o;
}
static Json read_json(const fs::path& p) {
    std::ifstream file(p);
    if (!file) throw std::runtime_error("Cannot open: "+p.string());
    Json value; file>>value; return value;
}
static Json replay_frames(const fs::path& p) {
    Json normalized=Json::object();
    const auto document=read_json(p);
    for (const auto& item : document.at("frames").items()) normalized[path_key(item.key())]=item.value();
    return normalized;
}
static cv::Rect2f parse_box(const Json& row) {
    const float x1=row.at(0).get<float>(),y1=row.at(1).get<float>();
    const float x2=row.at(2).get<float>(),y2=row.at(3).get<float>();
    if (!std::isfinite(x1)||!std::isfinite(y1)||!std::isfinite(x2)||!std::isfinite(y2)||x2<x1||y2<y1)
        throw std::runtime_error("Invalid replay bounding box");
    return {x1,y1,x2-x1,y2-y1};
}
static void run(const Sequence& seq,const Options& options,const Config& config,
                Detector* detector,const Json& replay) {
    BotSort tracker(options.gmc);
    std::vector<std::vector<TrackedBox>> tracks;
    Json dumped={{"frames",Json::object()}};
    for (const auto& path : seq.frames) {
        (void)frame_id(path);
        const auto image=cv::imread(path.string());
        if (image.empty()) throw std::runtime_error("Cannot read image: "+path.string());
        std::vector<TrackedBox> current;
        if (!options.tracks_path.empty()) {
            const auto& entries=replay.at(path_key(path));
            for (const auto& row : entries) {
                if (!row.is_array()||row.size()!=7) throw std::runtime_error("Track row requires seven values");
                current.push_back({parse_box(row),row[4].get<std::int64_t>(),row[5].get<float>(),row[6].get<int>()});
            }
        } else {
            std::vector<Detection> detections;
            if (!options.detections_path.empty()) {
                for (const auto& row : replay.at(path_key(path))) {
                    if (!row.is_array()||row.size()!=6) throw std::runtime_error("Detection row requires six values");
                    detections.push_back({parse_box(row),row[4].get<float>(),row[5].get<int>()});
                }
            } else detections=detector->infer(image);
            current=tracker.update(detections,image);
        }
        if (options.dump_tracks) {
            auto values=Json::array();
            for (const auto& t : current)
                values.push_back({t.box.x,t.box.y,t.box.x+t.box.width,t.box.y+t.box.height,t.track_id,t.score,t.cls});
            dumped["frames"][path_key(path)]=std::move(values);
        }
        tracks.push_back(std::move(current));
    }
    const auto rows=compute_rows(seq,config,tracks,options);
    const fs::path base=options.output_path/seq.name;
    write_csvs(base,rows);
    save_morton_plot(base.string()+"_morton.png",rows);
    if (options.dump_tracks) {
        std::ofstream out(base.string()+"_tracks.json");
        out<<dumped.dump(2)<<'\n'; out.close();
        if (!out) throw std::runtime_error("Cannot save track replay");
    }
    std::cout<<"Written "<<rows.size()<<" rows: "<<base.string()<<"\n";
}
int main(int argc,char** argv) {
    try {
        const auto options=parse(argc,argv);
        const auto document=read_json(options.config_path);
        const auto config=load_config(document,options.dataset);
        const auto sequences=make_input_list(document,options.dataset);
        Json replay;
        if (!options.tracks_path.empty()) replay=replay_frames(options.tracks_path);
        else if (!options.detections_path.empty()) replay=replay_frames(options.detections_path);
        std::unique_ptr<Detector> detector;
        if (options.tracks_path.empty()&&options.detections_path.empty()) {
            if (!fs::is_regular_file(options.model_path)) throw std::runtime_error("Missing ONNX model: "+options.model_path.string());
            detector=std::make_unique<Detector>(options.model_path,options.model_width,options.model_height);
        }
        for (const auto& seq : sequences) {
            std::cout<<"Processing "<<seq.name<<" ("<<seq.frames.size()<<" frames)...\n";
            run(seq,options,config,detector.get(),replay);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr<<"Error: "<<error.what()<<'\n'; return 1;
    }
}
