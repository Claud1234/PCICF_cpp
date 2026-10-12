#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

namespace pcicf {
namespace fs = std::filesystem;
using Json = nlohmann::json;

struct Options {
    std::string dataset;
    fs::path output_path;
    fs::path config_path = "config.json";
    fs::path model_path = "yolo11x.onnx";
    fs::path tracks_path, detections_path;
    bool save_yolo_result = false, dump_tracks = false, gmc = true;
    int model_width = 640, model_height = 480;
};

struct Config {
    int image_width = 640, image_height = 480;
    int roi_width = 60, roi_height = 120;
    double track_threshold = 180, sfc_bounding = 0;
    std::array<cv::Rect, 6> rois;
    std::string yaml_path, smirk_input_path;
};

struct Detection {
    cv::Rect2f box;
    float score = 0;
    int cls = 0;
};
struct TrackedBox {
    cv::Rect2f box;
    std::int64_t track_id = 0;
    float score = 0;
    int cls = 0;
};
struct Row {
    std::int64_t time_stamp_ms = 0, frame_id = 0;
    std::array<unsigned, 6> cells{};
    std::uint64_t morton = 0;
};
struct Sequence {
    std::string name;
    std::vector<fs::path> frames;
};

Config load_config(const Json& config, const std::string& dataset);
std::vector<Sequence> make_input_list(const Json& config, const std::string& dataset);
std::int64_t frame_id(const fs::path& path);
std::string path_key(const fs::path& path);
std::uint64_t calculate_morton(const std::array<unsigned, 6>& cells);
std::array<unsigned, 6> occupancy(const Config& cfg, cv::Size image_size,
                                const std::vector<cv::Rect>& boxes);
std::vector<Row> compute_rows(const Sequence& sequence, const Config& cfg,
                             const std::vector<std::vector<TrackedBox>>& tracks,
                             const Options& options);
void write_csvs(const fs::path& base, const std::vector<Row>& rows);
void save_morton_plot(const fs::path& path, const std::vector<Row>& rows);

class Detector {
public:
    Detector(const fs::path& model, int width, int height);
    ~Detector();
    std::vector<Detection> infer(const cv::Mat& image);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Native port of the no-ReID configuration in Ultralytics 8.3.203.
class BotSort {
public:
    explicit BotSort(bool global_motion_compensation = true);
    ~BotSort();
    std::vector<TrackedBox> update(const std::vector<Detection>& detections,
                                   const cv::Mat& image);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace pcicf
