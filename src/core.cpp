#include "pcicf.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

namespace pcicf {
Config load_config(const Json& root, const std::string& dataset) {
    Config cfg;
    const auto& general = root.at("General");
    cfg.image_width = general.at("image_width").get<int>();
    cfg.image_height = general.at("image_height").get<int>();
    cfg.roi_width = general.at("roi_width").get<int>();
    cfg.roi_height = general.at("roi_height").get<int>();
    cfg.track_threshold = general.at("yolo_track_threshold").get<double>();
    cfg.sfc_bounding = general.at("sfc_bounding").get<double>();
    if (cfg.roi_width <= 0 || cfg.roi_height <= 0 || cfg.image_width <= 0 || cfg.image_height <= 0)
        throw std::runtime_error("Image and ROI dimensions must be positive");
    const auto& ds = root.at("Dataset").at(dataset);
    const auto& grid = ds.at("grid_left_top_coord");
    if (grid.size() != 6) throw std::runtime_error("This pipeline requires exactly six ROI cells");
    for (std::size_t i = 0; i < 6; ++i) {
        if (!grid[i].is_array() || grid[i].size() != 2)
            throw std::runtime_error("Each ROI origin must be [x, y]");
        const int x = grid[i][0].get<int>(), y = grid[i][1].get<int>();
        if (x < 0 || y < 0) throw std::runtime_error("ROI origins must be nonnegative");
        cfg.rois[i] = {x, y, cfg.roi_width, cfg.roi_height};
    }
    if (dataset == "pie") cfg.yaml_path = ds.at("yaml_path").get<std::string>();
    else cfg.smirk_input_path = ds.at("input_path").get<std::string>();
    return cfg;
}

std::string path_key(const fs::path& p) { return p.lexically_normal().generic_string(); }

std::int64_t frame_id(const fs::path& p) {
    const auto name = p.stem().string();
    std::size_t consumed = 0;
    const auto result = std::stoll(name, &consumed);
    if (consumed != name.size()) throw std::runtime_error("Frame filename must be an integer: " + p.string());
    return result;
}

std::vector<Sequence> make_input_list(const Json& config, const std::string& dataset) {
    const auto& ds = config.at("Dataset").at(dataset);
    std::vector<Sequence> result;
    if (dataset == "pie") {
        const std::string yaml_name = ds.at("yaml_path").get<std::string>();
        const auto document = YAML::LoadFile((fs::path("datasets/pie_splits") / yaml_name).string());
        if (!document.IsSequence()) throw std::runtime_error("PIE YAML must contain a sequence");
        for (const auto& item : document) {
            const auto window = item["event_window"];
            if (!window.IsSequence() || window.size() < 2) throw std::runtime_error("Invalid event_window");
            const auto id = item["id"].as<std::string>();
            const auto first = window[0].as<std::int64_t>();
            const auto last = window[window.size()-1].as<std::int64_t>();
            const auto split = yaml_name.substr(0, yaml_name.find('.'));
            Sequence seq{split + "/" + id + "_" + std::to_string(first) + "_" + std::to_string(last), {}};
            // Python range(first, last): last is deliberately exclusive.
            for (auto frame = first; frame < last; ++frame) {
                std::ostringstream name;
                name << std::setfill('0') << std::setw(3) << frame << ".png";
                seq.frames.push_back(fs::path("datasets/pie") / id / name.str());
            }
            result.push_back(std::move(seq));
        }
    } else if (dataset == "MoreSMIRK") {
        const fs::path folder = ds.at("input_path").get<std::string>();
        auto normalized = folder.lexically_normal();
        if (normalized.filename().empty()) normalized = normalized.parent_path();
        Sequence seq{"more_smirk_" + normalized.filename().string(), {}};
        if (fs::is_directory(folder)) {
            for (const auto& entry : fs::directory_iterator(folder)) {
                const auto filename = entry.path().filename().string();
                // Match glob('*.png'), including its exclusion of hidden names.
                if (entry.path().extension() == ".png" && !filename.empty() && filename[0] != '.')
                    seq.frames.push_back(entry.path());
            }
        }
        std::sort(seq.frames.begin(), seq.frames.end(), [](const auto& a, const auto& b) {
            return a.generic_string() < b.generic_string();
        });
        result.push_back(std::move(seq));
    } else throw std::runtime_error("Dataset must be pie or MoreSMIRK");
    return result;
}

std::uint64_t calculate_morton(const std::array<unsigned, 6>& cells) {
    std::uint64_t code = 0;
    for (unsigned dim = 0; dim < 6; ++dim) {
        if (cells[dim] > 1) throw std::runtime_error("Expected binary occupancy");
        // tools/sfc.py scales 0/1 to 0/10 BEFORE interleaving.
        const unsigned value = cells[dim] * 10;
        for (unsigned bit = 0; bit < 4; ++bit)
            code |= static_cast<std::uint64_t>((value >> bit) & 1U) << (bit * 6 + dim);
    }
    return code;
}

std::array<unsigned, 6> occupancy(const Config& cfg, cv::Size size, const std::vector<cv::Rect>& boxes) {
    std::array<unsigned, 6> cells{};
    const cv::Rect image_bounds(0, 0, size.width, size.height);
    for (std::size_t i = 0; i < cells.size(); ++i) {
        const cv::Rect clipped_roi = cfg.rois[i] & image_bounds;
        int pixels = 0;
        if (clipped_roi.area() > 0) {
            cv::Mat mask = cv::Mat::zeros(clipped_roi.size(), CV_8U);
            for (const auto& box : boxes) {
                auto intersection = box & clipped_roi;
                if (intersection.area() > 0) {
                    intersection.x -= clipped_roi.x;
                    intersection.y -= clipped_roi.y;
                    mask(intersection).setTo(1);
                }
            }
            pixels = cv::countNonZero(mask);
        }
        const double ratio = std::min(1.0, double(pixels) / (double(cfg.roi_width) * cfg.roi_height));
        cells[i] = ratio > cfg.sfc_bounding ? 1U : 0U;
    }
    return cells;
}

static int to_uint16(float value) {
    if (!std::isfinite(value)) throw std::runtime_error("Non-finite tracked coordinate");
    // Preserve the original truncation and modulo-65536 conversion.
    return static_cast<std::uint16_t>(static_cast<std::int64_t>(value));
}

static fs::path visual_path(const fs::path& input, const Options& options, const Config& cfg) {
    fs::path relative;
    bool found = false;
    for (const auto& part : input.lexically_normal()) {
        if (!found) { if (part == "datasets") found = true; }
        else relative /= part;
    }
    // Avoid overwriting input images when an external dataset path is used.
    if (!found) relative = input.filename();
    fs::path result = fs::path("outputs") / (options.dataset + "_visual");
    if (options.dataset == "pie") result /= cfg.yaml_path.substr(0, cfg.yaml_path.find('.'));
    return result / relative;
}

std::vector<Row> compute_rows(const Sequence& seq, const Config& cfg,
                             const std::vector<std::vector<TrackedBox>>& frames, const Options& options) {
    if (seq.frames.size() != frames.size()) throw std::runtime_error("Tracked frame count mismatch");
    struct Sample { cv::Rect box; int center_x; std::size_t frame_index; };
    std::map<std::uint16_t, std::vector<Sample>> history;
    for (std::size_t i = 0; i < frames.size(); ++i) {
        for (const auto& track : frames[i]) {
            if (track.cls != 0) continue;
            const float x1 = track.box.x, y1 = track.box.y;
            const float x2 = x1 + track.box.width, y2 = y1 + track.box.height;
            const int left = to_uint16(x1), top = to_uint16(y1);
            const int right = to_uint16(x2), bottom = to_uint16(y2);
            const int center = to_uint16((x1 + x2) / 2.0f);
            history[static_cast<std::uint16_t>(track.track_id)].push_back(
                {{left, top, std::max(0, right-left), std::max(0, bottom-top)}, center, i});
        }
    }
    std::vector<std::vector<cv::Rect>> crossing_boxes(frames.size());
    bool has_crossing = false;
    for (const auto& [id, samples] : history) {
        (void)id;
        int min_x = 65535, max_x = 0;
        for (const auto& s : samples) { min_x = std::min(min_x, s.center_x); max_x = std::max(max_x, s.center_x); }
        if (max_x-min_x > cfg.track_threshold && max_x > cfg.image_width/2.0 && min_x < cfg.image_width/2.0) {
            has_crossing = true;
            for (const auto& s : samples) crossing_boxes[s.frame_index].push_back(s.box);
        }
    }
    if (!has_crossing) return {};
    std::vector<Row> rows;
    for (std::size_t i = 0; i < seq.frames.size(); ++i) {
        cv::Mat image = cv::imread(seq.frames[i].string());
        if (image.empty()) throw std::runtime_error("Cannot read image: " + seq.frames[i].string());
        Row row;
        row.frame_id = frame_id(seq.frames[i]);
        row.time_stamp_ms = static_cast<std::int64_t>(i) * 33;
        row.cells = occupancy(cfg, image.size(), crossing_boxes[i]);
        row.morton = calculate_morton(row.cells);
        rows.push_back(row);
        if (options.save_yolo_result) {
            for (const auto& roi : cfg.rois)
                cv::rectangle(image, roi.tl(), cv::Point(roi.x+roi.width, roi.y+roi.height), {0,0,255}, 2);
            for (const auto& box : crossing_boxes[i])
                cv::rectangle(image, box.tl(), cv::Point(box.x+box.width, box.y+box.height), {0,255,0}, 2);
            const auto destination = visual_path(seq.frames[i], options, cfg);
            fs::create_directories(destination.parent_path());
            if (!cv::imwrite(destination.string(), image)) throw std::runtime_error("Cannot write " + destination.string());
        }
    }
    return rows;
}

void write_csvs(const fs::path& base, const std::vector<Row>& rows) {
    if (!base.parent_path().empty()) fs::create_directories(base.parent_path());
    std::ofstream sfc(base.string() + "_sfc_input.csv", std::ios::binary);
    std::ofstream morton(base.string() + "_morton.csv", std::ios::binary);
    if (!sfc || !morton) throw std::runtime_error("Cannot create CSV output for " + base.string());
    sfc << "time_stamp_ms;frame_id;cell_0;cell_1;cell_2;cell_3;cell_4;cell_5\n";
    morton << "time_stamp_ms;frame_id;morton\n";
    for (const auto& row : rows) {
        sfc << row.time_stamp_ms << ';' << row.frame_id;
        for (auto value : row.cells) sfc << ';' << value;
        sfc << '\n';
        morton << row.time_stamp_ms << ';' << row.frame_id << ';' << row.morton << '\n';
    }
    sfc.close(); morton.close();
    if (!sfc || !morton) throw std::runtime_error("Failed to finish CSV output");
}
} // namespace pcicf
