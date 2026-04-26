// Usage: ./batch_maker <input_folder> <overlay_folder> <output_name> [min_unique_markers]

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/img_hash.hpp>

#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <random>
#include <limits>
#include <string>
#include <iomanip>

namespace fs = std::filesystem;

// ---------- helpers ----------
static cv::Mat ensureBGR(const cv::Mat& src) {
    if (src.empty()) return src;
    if (src.channels() == 3) return src;

    cv::Mat out;
    if (src.channels() == 1) cv::cvtColor(src, out, cv::COLOR_GRAY2BGR);
    else if (src.channels() == 4) cv::cvtColor(src, out, cv::COLOR_BGRA2BGR);
    else out = src.clone();

    return out;
}

static bool hasImageExt(const fs::path& p) {
    auto ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (
        ext == ".jpg" ||
        ext == ".jpeg" ||
        ext == ".png" ||
        ext == ".bmp" ||
        ext == ".webp"
    );
}

static cv::Rect2f quadToAABB(const std::vector<cv::Point2f>& q) {
    float minx = q[0].x, maxx = q[0].x;
    float miny = q[0].y, maxy = q[0].y;

    for (int i = 1; i < (int)q.size(); ++i) {
        minx = std::min(minx, q[i].x);
        maxx = std::max(maxx, q[i].x);
        miny = std::min(miny, q[i].y);
        maxy = std::max(maxy, q[i].y);
    }

    return cv::Rect2f(minx, miny, maxx - minx, maxy - miny);
}

static cv::Rect2f clampRectToImage(const cv::Rect2f& r, const cv::Size& sz) {
    float x1 = std::max(0.f, r.x);
    float y1 = std::max(0.f, r.y);
    float x2 = std::min((float)sz.width  - 1.f, r.x + r.width);
    float y2 = std::min((float)sz.height - 1.f, r.y + r.height);

    float w = x2 - x1;
    float h = y2 - y1;

    if (w <= 0.f || h <= 0.f) return cv::Rect2f(0, 0, 0, 0);
    return cv::Rect2f(x1, y1, w, h);
}

static unsigned long long ipow_u64(unsigned long long base, unsigned long long exp) {
    const unsigned long long LIM = std::numeric_limits<unsigned long long>::max();

    unsigned long long res = 1ULL;
    for (unsigned long long i = 0; i < exp; i++) {
        if (base != 0 && res > LIM / base) return LIM;
        res *= base;
    }

    return res;
}

static cv::Mat letterboxResize(
    const cv::Mat& src,
    int target_w,
    int target_h,
    cv::Scalar pad_color = cv::Scalar(114, 114, 114)
) {
    if (src.empty()) return src;

    float scale = std::min(
        target_w / (float)src.cols,
        target_h / (float)src.rows
    );

    int new_w = std::max(1, (int)std::round(src.cols * scale));
    int new_h = std::max(1, (int)std::round(src.rows * scale));

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat out(target_h, target_w, src.type(), pad_color);

    int x_offset = (target_w - new_w) / 2;
    int y_offset = (target_h - new_h) / 2;

    resized.copyTo(out(cv::Rect(x_offset, y_offset, new_w, new_h)));
    return out;
}

// ---------- OpenCV pHash helpers ----------
static cv::Mat computeOpenCVPHash(const cv::Mat& input) {
    cv::Mat img = ensureBGR(input);
    cv::Mat hash;

    if (img.empty()) return hash;

    cv::Ptr<cv::img_hash::PHash> phasher = cv::img_hash::PHash::create();
    phasher->compute(img, hash);

    return hash;
}

static double pHashDistance(const cv::Mat& h1, const cv::Mat& h2) {
    if (h1.empty() || h2.empty()) {
        return std::numeric_limits<double>::infinity();
    }

    return cv::norm(h1, h2, cv::NORM_HAMMING);
}

static std::string hashMatToHex(const cv::Mat& hash) {
    if (hash.empty()) return "EMPTY";

    cv::Mat flat = hash.reshape(1, 1);

    std::ostringstream ss;
    ss << "0x";

    for (int i = 0; i < flat.cols; i++) {
        int v = (int)flat.at<uchar>(0, i);
        ss << std::hex
           << std::setw(2)
           << std::setfill('0')
           << v;
    }

    return ss.str();
}

static void countLabelsInto(
    const std::string& yolo_txt,
    std::vector<unsigned long long>& final_class_counts,
    int multiplier
) {
    std::istringstream iss(yolo_txt);

    int cls;
    float cx, cy, w, h;

    while (iss >> cls >> cx >> cy >> w >> h) {
        if (cls >= 0 && cls < (int)final_class_counts.size()) {
            final_class_counts[cls] += multiplier;
        }
    }
}

// Draw a filled label box + class id text
static void drawClassLabel(cv::Mat& img, int class_id, const cv::Rect2f& bbox) {
    std::string text = std::to_string(class_id);

    int baseline = 0;
    cv::Size ts = cv::getTextSize(
        text,
        cv::FONT_HERSHEY_SIMPLEX,
        0.6,
        2,
        &baseline
    );

    int x = std::max(0, (int)bbox.x);
    int y = (int)bbox.y - 6;

    if (y - ts.height - 6 < 0) {
        y = std::max(ts.height + 6, (int)bbox.y + ts.height + 6);
    }

    int rect_w = ts.width + 8;
    int rect_h = ts.height + 8;

    int rx = x;
    int ry = y - rect_h;

    if (rx + rect_w >= img.cols) {
        rx = std::max(0, img.cols - rect_w - 1);
    }

    if (ry < 0) ry = 0;

    cv::rectangle(
        img,
        cv::Rect(rx, ry, rect_w, rect_h),
        cv::Scalar(0, 255, 0),
        cv::FILLED
    );

    cv::putText(
        img,
        text,
        cv::Point(rx + 4, ry + rect_h - 4),
        cv::FONT_HERSHEY_SIMPLEX,
        0.6,
        cv::Scalar(0, 0, 0),
        2,
        cv::LINE_AA
    );
}

// Render ONE output given assignment.
// YOLO class == overlay index.
static bool renderWithAssignment(
    const cv::Mat& image_in,
    const std::vector<int>& ids,
    const std::vector<std::vector<cv::Point2f>>& corners,
    const std::unordered_map<int, int>& id_to_overlay_idx,
    const std::vector<cv::Mat>& overlays,
    cv::Mat& output_img,
    cv::Mat& boxed_img,
    std::string& yolo_labels_out
) {
    cv::Mat image = ensureBGR(image_in);
    if (image.empty()) return false;

    cv::Mat output = image.clone();
    cv::Mat output_boxed = image.clone();

    const int marker_patch_size = 320;
    const int border_px = 1;
    const int cover_px = 14;

    const float min_visible_frac = 0.15f;
    const float min_visible_area = 400.f;

    const float imgW = (float)image.cols;
    const float imgH = (float)image.rows;

    std::ostringstream label_ss;

    for (int i = 0; i < (int)ids.size(); i++) {
        int id = ids[i];

        auto it = id_to_overlay_idx.find(id);
        if (it == id_to_overlay_idx.end()) continue;

        int ov_idx = it->second;
        if (ov_idx < 0 || ov_idx >= (int)overlays.size()) continue;

        const cv::Mat& overlay = overlays[ov_idx];
        if (overlay.empty()) continue;

        std::vector<cv::Point2f> m = corners[i];
        if (m.size() != 4) continue;

        std::vector<cv::Point2f> marker_dst = {
            {0.f, 0.f},
            {(float)marker_patch_size - 1, 0.f},
            {(float)marker_patch_size - 1, (float)marker_patch_size - 1},
            {0.f, (float)marker_patch_size - 1}
        };

        cv::Mat Hm = cv::findHomography(m, marker_dst);
        if (Hm.empty()) continue;

        cv::Mat marker_patch;
        cv::warpPerspective(
            image,
            marker_patch,
            Hm,
            cv::Size(marker_patch_size, marker_patch_size),
            cv::INTER_LINEAR,
            cv::BORDER_CONSTANT,
            cv::Scalar(255, 255, 255)
        );

        marker_patch = ensureBGR(marker_patch);

        cv::Mat overlay_resized;
        {
            int targetW = marker_patch.cols;

            int newH = (int)std::round(
                overlay.rows * (targetW / (double)overlay.cols)
            );

            newH = std::max(1, newH);

            cv::resize(
                overlay,
                overlay_resized,
                cv::Size(targetW, newH),
                0,
                0,
                cv::INTER_AREA
            );
        }

        cv::Mat overlay_patch;
        cv::copyMakeBorder(
            overlay_resized,
            overlay_patch,
            cover_px,
            cover_px,
            cover_px,
            cover_px,
            cv::BORDER_REPLICATE
        );

        float x0 = (float)cover_px;
        float y0 = (float)cover_px;

        std::vector<cv::Point2f> marker_rect_overlay = {
            cv::Point2f(x0, y0),
            cv::Point2f(x0 + marker_patch.cols - 1, y0),
            cv::Point2f(x0 + marker_patch.cols - 1, y0 + marker_patch.rows - 1),
            cv::Point2f(x0, y0 + marker_patch.rows - 1)
        };

        cv::Mat Hc = cv::findHomography(marker_rect_overlay, m);
        if (Hc.empty()) continue;

        std::vector<cv::Point2f> overlay_rect = {
            cv::Point2f(0.f, 0.f),
            cv::Point2f((float)overlay_patch.cols - 1, 0.f),
            cv::Point2f((float)overlay_patch.cols - 1, (float)overlay_patch.rows - 1),
            cv::Point2f(0.f, (float)overlay_patch.rows - 1)
        };

        std::vector<cv::Point2f> overlay_quad_img;
        cv::perspectiveTransform(overlay_rect, overlay_quad_img, Hc);

        cv::Rect2f overlay_aabb = quadToAABB(overlay_quad_img);
        cv::Rect2f overlay_clip = clampRectToImage(overlay_aabb, image.size());

        float full_area = std::max(1.f, overlay_aabb.area());
        float vis_area = overlay_clip.area();
        float vis_frac = vis_area / full_area;

        if (vis_area < min_visible_area || vis_frac < min_visible_frac) continue;

        cv::Mat overlay_warp;
        cv::warpPerspective(
            overlay_patch,
            overlay_warp,
            Hc,
            image.size(),
            cv::INTER_LINEAR,
            cv::BORDER_CONSTANT,
            cv::Scalar(0, 0, 0)
        );

        cv::Mat mask_src(
            overlay_patch.rows,
            overlay_patch.cols,
            CV_8UC1,
            cv::Scalar(255)
        );

        cv::Mat mask;
        cv::warpPerspective(
            mask_src,
            mask,
            Hc,
            image.size(),
            cv::INTER_NEAREST,
            cv::BORDER_CONSTANT,
            cv::Scalar(0)
        );

        overlay_warp.copyTo(output, mask);
        overlay_warp.copyTo(output_boxed, mask);

        std::vector<cv::Point> border_poly;
        border_poly.reserve(4);

        for (const auto& p : overlay_quad_img) {
            border_poly.push_back(
                cv::Point(
                    (int)std::round(p.x),
                    (int)std::round(p.y)
                )
            );
        }

        cv::polylines(
            output,
            border_poly,
            true,
            cv::Scalar(0, 0, 0),
            border_px,
            cv::LINE_AA
        );

        cv::polylines(
            output_boxed,
            border_poly,
            true,
            cv::Scalar(0, 0, 0),
            border_px,
            cv::LINE_AA
        );

        cv::Rect2f ov_aabb = quadToAABB(overlay_quad_img);
        cv::Rect2f ov_clip = clampRectToImage(ov_aabb, image.size());

        if (ov_clip.area() < 25.f) continue;

        float cx = (ov_clip.x + ov_clip.width * 0.5f) / imgW;
        float cy = (ov_clip.y + ov_clip.height * 0.5f) / imgH;
        float w = ov_clip.width / imgW;
        float h = ov_clip.height / imgH;

        if (w <= 0.f || h <= 0.f) continue;

        cx = std::min(1.f, std::max(0.f, cx));
        cy = std::min(1.f, std::max(0.f, cy));
        w = std::min(1.f, std::max(0.f, w));
        h = std::min(1.f, std::max(0.f, h));

        label_ss << ov_idx << " "
                 << cx << " "
                 << cy << " "
                 << w << " "
                 << h << "\n";

        cv::rectangle(
            output_boxed,
            cv::Rect(
                (int)ov_clip.x,
                (int)ov_clip.y,
                (int)ov_clip.width,
                (int)ov_clip.height
            ),
            cv::Scalar(0, 255, 0),
            2,
            cv::LINE_AA
        );

        for (int k = 0; k < 4; k++) {
            cv::circle(
                output_boxed,
                overlay_quad_img[k],
                4,
                cv::Scalar(0, 0, 255),
                -1,
                cv::LINE_AA
            );
        }

        drawClassLabel(output_boxed, ov_idx, ov_clip);
    }

    output_img = output;
    boxed_img = output_boxed;
    yolo_labels_out = label_ss.str();

    return true;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <input_folder> <overlay_folder> <output_name> [min_unique_markers]\n";
        return 1;
    }

    fs::path input_folder = argv[1];
    fs::path overlay_folder = argv[2];
    std::string output_name = argv[3];

    int min_unique_markers = 0;
    bool use_min_unique = false;

    if (argc >= 5) {
        try {
            min_unique_markers = std::stoi(argv[4]);
        } catch (...) {
            std::cerr << "Error: min_unique_markers must be an integer.\n";
            return 1;
        }

        if (min_unique_markers < 0) {
            std::cerr << "Error: min_unique_markers must be >= 0.\n";
            return 1;
        }

        use_min_unique = (min_unique_markers > 0);
    }

    if (!fs::exists(input_folder) || !fs::is_directory(input_folder)) {
        std::cerr << "Input folder not found: " << input_folder << "\n";
        return 1;
    }

    if (!fs::exists(overlay_folder) || !fs::is_directory(overlay_folder)) {
        std::cerr << "Overlay folder not found: " << overlay_folder << "\n";
        return 1;
    }

    fs::path out_dir = fs::path(output_name);
    fs::create_directories(out_dir);

    fs::path images_dir = out_dir / "images";
    fs::path bbox_dir = out_dir / "labels";
    fs::path boxed_dir = out_dir / ("boxed_" + output_name);

    fs::create_directories(images_dir);
    fs::create_directories(bbox_dir);
    fs::create_directories(boxed_dir);

    std::vector<fs::path> overlay_paths;

    for (const auto& e : fs::directory_iterator(overlay_folder)) {
        if (e.is_regular_file() && hasImageExt(e.path())) {
            overlay_paths.push_back(e.path());
        }
    }

    std::sort(overlay_paths.begin(), overlay_paths.end());

    if (overlay_paths.empty()) {
        std::cerr << "No overlay images found in: " << overlay_folder << "\n";
        return 1;
    }

    std::vector<cv::Mat> overlays;
    std::vector<std::string> overlay_names;

    overlays.reserve(overlay_paths.size());
    overlay_names.reserve(overlay_paths.size());

    for (const auto& p : overlay_paths) {
        cv::Mat ov = cv::imread(p.string(), cv::IMREAD_UNCHANGED);
        ov = ensureBGR(ov);

        if (ov.empty()) {
            std::cerr << "Warning: could not read overlay: " << p << " skipping\n";
            continue;
        }

        overlays.push_back(ov);
        overlay_names.push_back(p.filename().string());
    }

    if (overlays.empty()) {
        std::cerr << "All overlay reads failed.\n";
        return 1;
    }

    // ---------- OpenCV pHash eval ----------
    const double similar_phash_threshold = 10.0;

    std::vector<cv::Mat> overlay_hashes(overlays.size());
    std::vector<double> overlay_sampling_weights(overlays.size(), 1.0);
    std::vector<int> similar_neighbor_counts(overlays.size(), 0);

    std::cout << "\n[pHash] OpenCV img_hash::PHash hashes by class:\n";

    for (size_t i = 0; i < overlays.size(); i++) {
        overlay_hashes[i] = computeOpenCVPHash(overlays[i]);

        std::cout << "  class " << i
                  << " file=" << overlay_names[i]
                  << " phash=" << hashMatToHex(overlay_hashes[i])
                  << "\n";
    }

    std::cout << "\n[pHash] similar overlay pairs, threshold <= "
              << similar_phash_threshold << ":\n";

    bool found_similar_pair = false;

    for (size_t i = 0; i < overlays.size(); i++) {
        for (size_t j = i + 1; j < overlays.size(); j++) {
            double dist = pHashDistance(overlay_hashes[i], overlay_hashes[j]);

            if (dist <= similar_phash_threshold) {
                found_similar_pair = true;
                similar_neighbor_counts[i]++;
                similar_neighbor_counts[j]++;

                std::cout << "  class " << i
                          << " <-> class " << j
                          << " distance=" << dist
                          << " files=(" << overlay_names[i]
                          << ", " << overlay_names[j] << ")\n";
            }
        }
    }

    if (!found_similar_pair) {
        std::cout << "  none\n";
    }

    for (size_t i = 0; i < overlays.size(); i++) {
        overlay_sampling_weights[i] =
            1.0 + 2.0 * std::sqrt((double)similar_neighbor_counts[i]);

    }

    std::cout << "\n[pHash] sampling weights:\n";

    for (size_t i = 0; i < overlay_sampling_weights.size(); i++) {
        std::cout << "  class " << i
          << " file=" << overlay_names[i]
          << " weight=" << overlay_sampling_weights[i]
          << " similar_neighbors=" << similar_neighbor_counts[i]
          << "\n";
    }

    {
        fs::path map_path = out_dir / "overlay_index_map.txt";
        std::ofstream mf(map_path.string());

        if (!mf.is_open()) {
            std::cerr << "Warning: could not write mapping file: " << map_path << "\n";
        } else {
            mf << "# class_id(overlay_index)\toverlay_filename\tphash\tsampling_weight\tsimilar_neighbors\n";

            for (size_t i = 0; i < overlay_names.size(); i++) {
                mf << i << "\t"
                   << overlay_names[i] << "\t"
                   << hashMatToHex(overlay_hashes[i]) << "\t"
                   << overlay_sampling_weights[i] << "\t"
                   << similar_neighbor_counts[i] << "\n";
            }

            std::cout << "[info] wrote overlay map: " << map_path << "\n";
        }
    }

    std::vector<fs::path> image_paths;

    for (const auto& e : fs::directory_iterator(input_folder)) {
        if (e.is_regular_file() && hasImageExt(e.path())) {
            image_paths.push_back(e.path());
        }
    }

    std::sort(image_paths.begin(), image_paths.end());

    if (image_paths.empty()) {
        std::cerr << "No images found in: " << input_folder << "\n";
        return 1;
    }

    auto dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

    const int outputs_per_input_image = 100;
    const int extra_outputs_if_similar = 50;
    const int duplicates_per_output = 1;

    const int OUTPUT_W = 1280;
    const int OUTPUT_H = 960;

    std::mt19937 rng(1337);

    std::discrete_distribution<int> weighted_overlay_dist(
        overlay_sampling_weights.begin(), 
        overlay_sampling_weights.end()
    );

    const int max_attempts_per_input =
        (outputs_per_input_image + extra_outputs_if_similar) * 200;

    std::vector<unsigned long long> final_class_counts(overlays.size(), 0ULL);

    for (size_t img_idx = 0; img_idx < image_paths.size(); img_idx++) {
        const auto& img_path = image_paths[img_idx];

        cv::Mat image_raw = cv::imread(img_path.string(), cv::IMREAD_UNCHANGED);

        if (image_raw.empty()) {
            std::cerr << "Warning: could not read image: " << img_path << " skipping\n";
            continue;
        }

        image_raw = ensureBGR(image_raw);

        cv::Mat image = letterboxResize(image_raw, OUTPUT_W, OUTPUT_H);

        if (image.empty()) {
            std::cerr << "Warning: letterbox failed: " << img_path << " skipping\n";
            continue;
        }

        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;

        cv::aruco::detectMarkers(gray, dictionary, corners, ids);

        if (ids.empty()) {
            std::cout << "[skip] no markers: " << img_path.filename().string() << "\n";
            continue;
        }

        std::vector<int> unique_ids = ids;

        std::sort(unique_ids.begin(), unique_ids.end());
        unique_ids.erase(
            std::unique(unique_ids.begin(), unique_ids.end()),
            unique_ids.end()
        );

        if (use_min_unique && (int)unique_ids.size() < min_unique_markers) {
            std::cout << "[skip] too few unique markers: "
                      << img_path.filename().string()
                      << " unique_ids=" << unique_ids.size()
                      << " < min_unique_markers=" << min_unique_markers
                      << "\n";
            continue;
        }

        const unsigned long long K = (unsigned long long)overlays.size();
        const unsigned long long U = (unsigned long long)unique_ids.size();

        unsigned long long theoretical_variants = ipow_u64(K, U);

        const bool enumerate_all = (unique_ids.size() == 1);

        const int target_outputs = enumerate_all
            ? (int)overlays.size()
            : outputs_per_input_image + (found_similar_pair ? extra_outputs_if_similar : 0);

        std::cout << "[info] " << img_path.filename().string()
                  << " unique_ids=" << unique_ids.size()
                  << " overlays=" << overlays.size()
                  << " theoretical_variants=";

        if (theoretical_variants == std::numeric_limits<unsigned long long>::max()) {
            std::cout << ">=2^64-1";
        } else {
            std::cout << theoretical_variants;
        }

        std::cout << " mode=" << (enumerate_all ? "ENUM_ALL(K)" : "WEIGHTED_RANDOM(N)")
                  << " target_outputs=" << target_outputs
                  << " duplicates=" << duplicates_per_output
                  << " output_res=" << OUTPUT_W << "x" << OUTPUT_H
                  << "\n";

        int produced = 0;

        if (enumerate_all) {
            const int uid = unique_ids[0];

            for (int ov = 0; ov < (int)overlays.size(); ov++) {
                std::unordered_map<int, int> assignment;
                assignment.reserve(1);
                assignment[uid] = ov;

                cv::Mat out_img;
                cv::Mat boxed_img;
                std::string yolo_txt;

                bool ok = renderWithAssignment(
                    image,
                    ids,
                    corners,
                    assignment,
                    overlays,
                    out_img,
                    boxed_img,
                    yolo_txt
                );

                if (!ok) continue;
                if (yolo_txt.empty()) continue;

                countLabelsInto(
                    yolo_txt,
                    final_class_counts,
                    duplicates_per_output
                );

                for (int d = 1; d <= duplicates_per_output; d++) {
                    std::string stem =
                        output_name + "_" +
                        std::to_string(img_idx + 1) + "_" +
                        std::to_string(produced + 1) +
                        "_dup" + std::to_string(d);

                    fs::path out_img_path = images_dir / (stem + ".jpg");
                    fs::path out_label_path = bbox_dir / (stem + ".txt");
                    fs::path boxed_img_path = boxed_dir / (stem + ".jpg");

                    cv::imwrite(out_img_path.string(), out_img);
                    cv::imwrite(boxed_img_path.string(), boxed_img);

                    std::ofstream lf(out_label_path.string());
                    if (lf.is_open()) lf << yolo_txt;
                }

                produced++;
            }

            if (produced < (int)overlays.size()) {
                std::cout << "[warn] Enumerate-all: produced "
                          << produced << " / " << overlays.size()
                          << "\n";
            }
        } else {
            int attempts = 0;

            while (produced < target_outputs && attempts < max_attempts_per_input) {
                attempts++;

                std::unordered_map<int, int> assignment;
                assignment.reserve(unique_ids.size());

                for (int uid : unique_ids) {
                    assignment[uid] = weighted_overlay_dist(rng);
                }

                cv::Mat out_img;
                cv::Mat boxed_img;
                std::string yolo_txt;

                bool ok = renderWithAssignment(
                    image,
                    ids,
                    corners,
                    assignment,
                    overlays,
                    out_img,
                    boxed_img,
                    yolo_txt
                );

                if (!ok) continue;
                if (yolo_txt.empty()) continue;

                countLabelsInto(
                    yolo_txt,
                    final_class_counts,
                    duplicates_per_output
                );

                for (int d = 1; d <= duplicates_per_output; d++) {
                    std::string stem =
                        output_name + "_" +
                        std::to_string(img_idx + 1) + "_" +
                        std::to_string(produced + 1) +
                        "_dup" + std::to_string(d);

                    fs::path out_img_path = images_dir / (stem + ".jpg");
                    fs::path out_label_path = bbox_dir / (stem + ".txt");
                    fs::path boxed_img_path = boxed_dir / (stem + ".jpg");

                    cv::imwrite(out_img_path.string(), out_img);
                    cv::imwrite(boxed_img_path.string(), boxed_img);

                    std::ofstream lf(out_label_path.string());
                    if (lf.is_open()) lf << yolo_txt;
                }

                produced++;
            }

            if (produced < target_outputs) {
                std::cout << "[warn] Weighted random mode: could not reach "
                          << target_outputs
                          << " outputs for this image.\n";
            }
        }
    }

    std::cout << "\n[final] samples per class/overlay:\n";

    for (size_t i = 0; i < final_class_counts.size(); i++) {
        std::cout << "  class " << i
                  << " file=" << overlay_names[i]
                  << " samples=" << final_class_counts[i]
                  << "\n";
    }

    std::cout << "\nDone.\n"
              << "Images: " << images_dir << "\n"
              << "BBoxes: " << bbox_dir << "\n"
              << "Boxed:  " << boxed_dir << "\n";

    return 0;
}