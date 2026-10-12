// SPDX-License-Identifier: AGPL-3.0-or-later
// Port of the no-ReID BoT-SORT configuration in Ultralytics 8.3.203.
// See THIRD_PARTY_NOTICES.md and LICENSE.
#include "pcicf.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace pcicf {
namespace {
using Vec8 = Eigen::Matrix<double, 8, 1>;
using Mat8 = Eigen::Matrix<double, 8, 8>;
using Vec4 = Eigen::Matrix<double, 4, 1>;
enum class State { Tracked, Lost, Removed };
struct Track {
    Vec8 mean = Vec8::Zero();
    Mat8 cov = Mat8::Zero();
    std::int64_t id = 0;
    int start = 0, last = 0, cls = 0;
    float score = 0;
    bool activated = false;
    State state = State::Tracked;
    cv::Rect2f box() const {
        // Ultralytics emits coordinates as float32 after KF updates.
        const float x1 = float(mean[0]-mean[2]/2), y1 = float(mean[1]-mean[3]/2);
        const float x2 = float(mean[0]+mean[2]/2), y2 = float(mean[1]+mean[3]/2);
        return {x1, y1, x2-x1, y2-y1};
    }
};
using Ptr = std::shared_ptr<Track>;
using Tracks = std::vector<Ptr>;

Vec4 measurement(const Detection& d) {
    const float x = d.box.x + d.box.width/2, y = d.box.y + d.box.height/2;
    // The Python track constructor stores a float32 top-left/width/height box.
    const float left = x-d.box.width/2, top = y-d.box.height/2;
    return {float(left+d.box.width/2), float(top+d.box.height/2), d.box.width, d.box.height};
}

void initiate(Track& t, const Detection& d) {
    t.mean.setZero();
    t.mean.head<4>() = measurement(d);
    t.cov.setZero();
    for (int i = 0; i < 8; ++i) {
        const double dim = t.mean[(i%2) + 2];
        const double sigma = dim * (i < 4 ? 2.0/20 : 10.0/160);
        t.cov(i,i) = sigma*sigma;
    }
    t.cls = d.cls; t.score = d.score;
}

void predict(Track& t) {
    if (t.state != State::Tracked) t.mean[6] = t.mean[7] = 0;
    Mat8 transition = Mat8::Identity(), noise = Mat8::Zero();
    for (int i = 0; i < 4; ++i) transition(i, i+4) = 1;
    for (int i = 0; i < 8; ++i) {
        const double sigma = t.mean[(i%2)+2] * (i < 4 ? 1.0/20 : 1.0/160);
        noise(i,i) = sigma*sigma;
    }
    t.mean = (transition*t.mean).eval();
    t.cov = (transition*t.cov*transition.transpose()+noise).eval();
}

void correct(Track& t, const Detection& d, int frame) {
    Eigen::Matrix4d projected = t.cov.topLeftCorner<4,4>();
    for (int i = 0; i < 4; ++i) {
        const double sigma = t.mean[(i%2)+2]/20;
        projected(i,i) += sigma*sigma;
    }
    const Eigen::Matrix<double,8,4> cross = t.cov.leftCols<4>();
    const Eigen::Matrix<double,8,4> gain = projected.llt().solve(cross.transpose()).transpose();
    t.mean += gain*(measurement(d)-t.mean.head<4>());
    t.cov -= gain*projected*gain.transpose();
    t.state = State::Tracked; t.activated = true; t.last = frame;
    t.score = d.score; t.cls = d.cls;
}

float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float w = std::max(0.0f, std::min(a.x+a.width,b.x+b.width)-std::max(a.x,b.x));
    const float h = std::max(0.0f, std::min(a.y+a.height,b.y+b.height)-std::max(a.y,b.y));
    const float intersection = w*h;
    return intersection/(a.width*a.height+b.width*b.height-intersection+1e-7f);
}

struct Assignment {
    std::vector<std::pair<int,int>> matches;
    std::vector<int> rows, columns;
};

// Hungarian assignment with dummy costs equal to half the match threshold.
// This solves the same limited-cost objective as lapjv(extend_cost=True).
Assignment assign(const Tracks& tracks, const std::vector<Detection>& dets, double threshold, bool fuse) {
    Assignment result;
    const int nr = int(tracks.size()), nc = int(dets.size()), n = nr+nc;
    if (!nr || !nc) {
        for (int i=0;i<nr;++i) result.rows.push_back(i);
        for (int i=0;i<nc;++i) result.columns.push_back(i);
        return result;
    }
    std::vector<std::vector<double>> cost(n+1, std::vector<double>(n+1,threshold/2));
    for (int i=0;i<nr;++i) for (int j=0;j<nc;++j) {
        float distance = 1.0f-iou(tracks[i]->box(),dets[j].box);
        if (fuse) distance = 1.0f-(1.0f-distance)*dets[j].score;
        cost[i+1][j+1] = distance;
    }
    for (int i=nr+1;i<=n;++i) for (int j=nc+1;j<=n;++j) cost[i][j] = 0;
    std::vector<double> u(n+1), v(n+1);
    std::vector<int> p(n+1), way(n+1);
    for (int i=1;i<=n;++i) {
        p[0]=i; int j0=0;
        std::vector<double> best(n+1,std::numeric_limits<double>::infinity());
        std::vector<bool> used(n+1,false);
        do {
            used[j0]=true; const int i0=p[j0];
            double delta=std::numeric_limits<double>::infinity(); int j1=0;
            for (int j=1;j<=n;++j) if (!used[j]) {
                const double current=cost[i0][j]-u[i0]-v[j];
                if (current<best[j]) { best[j]=current; way[j]=j0; }
                if (best[j]<delta) { delta=best[j]; j1=j; }
            }
            for (int j=0;j<=n;++j) {
                if (used[j]) { u[p[j]]+=delta; v[j]-=delta; }
                else best[j]-=delta;
            }
            j0=j1;
        } while (p[j0]);
        do { const int j1=way[j0]; p[j0]=p[j1]; j0=j1; } while (j0);
    }
    std::vector<int> row_to_column(nr,-1);
    std::vector<bool> matched_columns(nc,false);
    for (int j=1;j<=nc;++j) if (p[j]>0 && p[j]<=nr && cost[p[j]][j]<=threshold) {
        row_to_column[p[j]-1]=j-1; matched_columns[j-1]=true;
    }
    for (int i=0;i<nr;++i) {
        if (row_to_column[i]<0) result.rows.push_back(i);
        else result.matches.emplace_back(i,row_to_column[i]);
    }
    for (int j=0;j<nc;++j) if (!matched_columns[j]) result.columns.push_back(j);
    return result;
}

Tracks joint(const Tracks& a, const Tracks& b) {
    Tracks result=a;
    std::unordered_set<std::int64_t> ids;
    for (const auto& t : a) ids.insert(t->id);
    for (const auto& t : b) if (ids.insert(t->id).second) result.push_back(t);
    return result;
}
Tracks subtract(const Tracks& a, const Tracks& b) {
    std::unordered_set<std::int64_t> ids;
    for (const auto& t : b) ids.insert(t->id);
    Tracks result;
    for (const auto& t : a) if (!ids.count(t->id)) result.push_back(t);
    return result;
}

struct MotionCompensation {
    cv::Mat previous;
    std::vector<cv::Point2f> previous_points;
    cv::Matx23d apply(const cv::Mat& image) {
        const cv::Matx23d identity(1,0,0,0,1,0);
        cv::Mat gray, small;
        cv::cvtColor(image,gray,cv::COLOR_BGR2GRAY);
        cv::resize(gray,small,{image.cols/2,image.rows/2});
        std::vector<cv::Point2f> points;
        cv::goodFeaturesToTrack(small,points,1000,0.01,1,cv::noArray(),3,false,0.04);
        if (previous.empty() || previous_points.empty()) {
            previous=small.clone(); previous_points=points; return identity;
        }
        std::vector<cv::Point2f> next, before, after;
        std::vector<uchar> status; std::vector<float> error;
        cv::calcOpticalFlowPyrLK(previous,small,previous_points,next,status,error);
        for (std::size_t i=0;i<status.size();++i) if (status[i]) {
            before.push_back(previous_points[i]); after.push_back(next[i]);
        }
        cv::Matx23d warp=identity;
        if (before.size()>4) {
            const auto estimate=cv::estimateAffinePartial2D(before,after,cv::noArray(),cv::RANSAC);
            if (!estimate.empty()) {
                for (int i=0;i<2;++i) for (int j=0;j<3;++j) warp(i,j)=estimate.at<double>(i,j);
                warp(0,2)*=2; warp(1,2)*=2;
            }
        }
        previous=small.clone(); previous_points=std::move(points);
        return warp;
    }
};

void warp_track(Track& t, const cv::Matx23d& warp) {
    Mat8 rotation=Mat8::Zero();
    for (int k=0;k<4;++k) for (int i=0;i<2;++i) for (int j=0;j<2;++j)
        rotation(2*k+i,2*k+j)=warp(i,j);
    t.mean=(rotation*t.mean).eval();
    t.mean[0]+=warp(0,2); t.mean[1]+=warp(1,2);
    t.cov=(rotation*t.cov*rotation.transpose()).eval();
}
} // anonymous namespace

struct BotSort::Impl {
    Tracks tracked, lost, removed;
    int frame=0;
    std::int64_t next_id=1;
    bool use_gmc;
    MotionCompensation motion;
    explicit Impl(bool gmc) : use_gmc(gmc) {}

    std::vector<TrackedBox> update(const std::vector<Detection>& input, const cv::Mat& image) {
        ++frame;
        std::vector<Detection> high, low;
        for (const auto& d : input) {
            if (d.score>=0.25f) high.push_back(d);
            else if (d.score>0.1f) low.push_back(d);
        }
        Tracks confirmed, unconfirmed, activated, refound, new_lost, new_removed;
        for (const auto& t : tracked) (t->activated?confirmed:unconfirmed).push_back(t);
        auto pool=joint(confirmed,lost);
        for (auto& t : pool) predict(*t);
        if (use_gmc && !image.empty()) {
            cv::Matx23d warp(1,0,0,0,1,0);
            try { warp=motion.apply(image); } catch (const cv::Exception&) {}
            for (auto& t : pool) warp_track(*t,warp);
            for (auto& t : unconfirmed) warp_track(*t,warp);
        }
        const auto first=assign(pool,high,0.8,true);
        for (const auto& [i,j] : first.matches) {
            const bool was_tracked=pool[i]->state==State::Tracked;
            correct(*pool[i],high[j],frame);
            (was_tracked?activated:refound).push_back(pool[i]);
        }
        Tracks remaining;
        for (int i : first.rows) if (pool[i]->state==State::Tracked) remaining.push_back(pool[i]);
        const auto second=assign(remaining,low,0.5,false);
        for (const auto& [i,j] : second.matches) {
            correct(*remaining[i],low[j],frame); activated.push_back(remaining[i]);
        }
        for (int i : second.rows) { remaining[i]->state=State::Lost; new_lost.push_back(remaining[i]); }
        std::vector<Detection> unmatched_high;
        for (int i : first.columns) unmatched_high.push_back(high[i]);
        const auto pending=assign(unconfirmed,unmatched_high,0.7,true);
        for (const auto& [i,j] : pending.matches) {
            correct(*unconfirmed[i],unmatched_high[j],frame); activated.push_back(unconfirmed[i]);
        }
        for (int i : pending.rows) { unconfirmed[i]->state=State::Removed; new_removed.push_back(unconfirmed[i]); }
        for (int i : pending.columns) {
            if (unmatched_high[i].score<0.25f) continue;
            auto t=std::make_shared<Track>(); initiate(*t,unmatched_high[i]);
            t->id=next_id++; t->start=t->last=frame; t->activated=frame==1;
            activated.push_back(t);
        }
        for (auto& t : lost) if (frame-t->last>30) { t->state=State::Removed; new_removed.push_back(t); }
        Tracks still_tracked;
        for (const auto& t : tracked) if (t->state==State::Tracked) still_tracked.push_back(t);
        tracked=joint(joint(still_tracked,activated),refound);
        lost=subtract(lost,tracked);
        lost.insert(lost.end(),new_lost.begin(),new_lost.end());
        lost=subtract(lost,removed);
        std::vector<bool> drop_tracked(tracked.size()), drop_lost(lost.size());
        for (std::size_t i=0;i<tracked.size();++i) for (std::size_t j=0;j<lost.size();++j) {
            if (1.0f-iou(tracked[i]->box(),lost[j]->box())<0.15f) {
                if (tracked[i]->last-tracked[i]->start>lost[j]->last-lost[j]->start) drop_lost[j]=true;
                else drop_tracked[i]=true;
            }
        }
        Tracks unique_tracked, unique_lost;
        for (std::size_t i=0;i<tracked.size();++i) if (!drop_tracked[i]) unique_tracked.push_back(tracked[i]);
        for (std::size_t i=0;i<lost.size();++i) if (!drop_lost[i]) unique_lost.push_back(lost[i]);
        tracked=std::move(unique_tracked); lost=std::move(unique_lost);
        removed.insert(removed.end(),new_removed.begin(),new_removed.end());
        if (removed.size()>1000) removed.erase(removed.begin(),removed.end()-999);
        std::vector<TrackedBox> output;
        for (const auto& t : tracked) if (t->activated) output.push_back({t->box(),t->id,t->score,t->cls});
        return output;
    }
};
BotSort::BotSort(bool gmc) : impl_(std::make_unique<Impl>(gmc)) {}
BotSort::~BotSort() = default;
std::vector<TrackedBox> BotSort::update(const std::vector<Detection>& d, const cv::Mat& image) {
    return impl_->update(d,image);
}
} // namespace pcicf
