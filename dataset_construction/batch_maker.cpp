// Usage: ./synth <input_folder> <overlay_folder> <output_name> [min_unique_markers]

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

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
    return (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".webp");
}

static cv::Rect2f quadToAABB(const std::vector<cv::Point2f>& q) {
    float minx = q[0].x, maxx = q[0].x;
    float miny = q[0].y, maxy = q[0].y;
    for (int i = 1; i < (int)q.size(); ++i) {
        minx = std::min(minx, q[i].x); maxx = std::max(maxx, q[i].x);
        miny = std::min(miny, q[i].y); maxy = std::max(maxy, q[i].y);
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
    if (w <= 0.f || h <= 0.f) return cv::Rect2f(0,0,0,0);
    return cv::Rect2f(x1, y1, w, h);
}

static unsigned long long ipow_u64(unsigned long long base, unsigned long long exp) {
    const unsigned long long LIM = std::numeric_limits<unsigned long long>::max();
    unsigned long long res = 1ULL;
    for (unsigned long long i = 0; i < exp; i++) {
        if (base != 0 && res > LIM / base) return LIM; // saturate
        res *= base;
    }
    return res;
}

// Letterbox resize to target size (no distortion)
static cv::Mat letterboxResize(
    const cv::Mat& src,
    int target_w,
    int target_h,
    cv::Scalar pad_color = cv::Scalar(114,114,114)
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

// Draw a filled label box + class id text (for readability)
static void drawClassLabel(cv::Mat& img, int class_id, const cv::Rect2f& bbox) {
    std::string text = std::to_string(class_id);

    int baseline = 0;
    cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);

    int x = std::max(0, (int)bbox.x);
    int y = (int)bbox.y - 6; // prefer above bbox
    if (y - ts.height - 6 < 0) {
        y = std::max(ts.height + 6, (int)bbox.y + ts.height + 6);
    }

    int rect_w = ts.width + 8;
    int rect_h = ts.height + 8;

    int rx = x;
    int ry = y - rect_h;

    if (rx + rect_w >= img.cols) rx = std::max(0, img.cols - rect_w - 1);
    if (ry < 0) ry = 0;

    cv::rectangle(img, cv::Rect(rx, ry, rect_w, rect_h), cv::Scalar(0,255,0), cv::FILLED);
    cv::putText(img, text, cv::Point(rx + 4, ry + rect_h - 4),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,0), 2, cv::LINE_AA);
}

// Render ONE output given assignment.
// YOLO class == overlay index.
// Boxed image shows bbox + corner points + class id text.
static bool renderWithAssignment(
    const cv::Mat& image_in,
    const std::vector<int>& ids,
    const std::vector<std::vector<cv::Point2f>>& corners,
    const std::unordered_map<int,int>& id_to_overlay_idx,
    const std::vector<cv::Mat>& overlays,
    cv::Mat& output_img,
    cv::Mat& boxed_img,
    std::string& yolo_labels_out
) {
    cv::Mat image = ensureBGR(image_in);
    if (image.empty()) return false;

    cv::Mat output = image.clone();
    cv::Mat output_boxed = image.clone();

    // ---- Card styling knobs ----
    const int padding = 20;
    const int gap = 14;
    const int border_px = 4;
    const int marker_patch_size = 320;

    // ---- Partial-visibility knobs ----
    const float min_visible_frac = 0.15f;
    const float min_visible_area = 400.f;

    const float imgW = (float)image.cols;
    const float imgH = (float)image.rows;

    std::ostringstream label_ss;

    for (int i = 0; i < (int)ids.size(); i++) {
        int id = ids[i];

        auto it = id_to_overlay_idx.find(id);
        if (it == id_to_overlay_idx.end()) continue;

        int ov_idx = it->second; // overlay index == class id
        if (ov_idx < 0 || ov_idx >= (int)overlays.size()) continue;

        const cv::Mat& overlay = overlays[ov_idx];
        if (overlay.empty()) continue;

        std::vector<cv::Point2f> m = corners[i];
        if (m.size() != 4) continue;

        // ---- 1) Extract marker patch ----
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
            image, marker_patch, Hm,
            cv::Size(marker_patch_size, marker_patch_size),
            cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255,255,255)
        );
        marker_patch = ensureBGR(marker_patch);

        // ---- 2) Build card (marker + overlay) ----
        cv::Mat overlay_resized;
        {
            int targetW = marker_patch.cols;
            int newH = (int)std::round(overlay.rows * (targetW / (double)overlay.cols));
            newH = std::max(1, newH);
            cv::resize(overlay, overlay_resized, cv::Size(targetW, newH), 0, 0, cv::INTER_AREA);
        }

        int contentW = marker_patch.cols;
        int contentH = marker_patch.rows + gap + overlay_resized.rows;

        int cardW = contentW + 2*(padding + border_px);
        int cardH = contentH + 2*(padding + border_px);

        cv::Mat card(cardH, cardW, CV_8UC3, cv::Scalar(255,255,255));
        cv::rectangle(card, cv::Rect(0,0,cardW,cardH), cv::Scalar(0,0,0), border_px, cv::LINE_AA);

        int x0 = border_px + padding;
        int y0 = border_px + padding;
        marker_patch.copyTo(card(cv::Rect(x0, y0, marker_patch.cols, marker_patch.rows)));

        int y1 = y0 + marker_patch.rows + gap;
        overlay_resized.copyTo(card(cv::Rect(x0, y1, overlay_resized.cols, overlay_resized.rows)));

        // Pin marker region to detected marker quad
        std::vector<cv::Point2f> marker_rect_card = {
            cv::Point2f((float)x0, (float)y0),
            cv::Point2f((float)(x0 + marker_patch.cols - 1), (float)y0),
            cv::Point2f((float)(x0 + marker_patch.cols - 1), (float)(y0 + marker_patch.rows - 1)),
            cv::Point2f((float)x0, (float)(y0 + marker_patch.rows - 1))
        };

        cv::Mat Hc = cv::findHomography(marker_rect_card, m);
        if (Hc.empty()) continue;

        // partial visibility check via card AABB clipped
        std::vector<cv::Point2f> card_corners = {
            cv::Point2f(0.f, 0.f),
            cv::Point2f((float)card.cols - 1, 0.f),
            cv::Point2f((float)card.cols - 1, (float)card.rows - 1),
            cv::Point2f(0.f, (float)card.rows - 1)
        };
        std::vector<cv::Point2f> card_corners_img;
        cv::perspectiveTransform(card_corners, card_corners_img, Hc);

        cv::Rect2f card_aabb = quadToAABB(card_corners_img);
        cv::Rect2f card_clip = clampRectToImage(card_aabb, image.size());

        float full_area = std::max(1.f, card_aabb.area());
        float vis_area  = card_clip.area();
        float vis_frac  = vis_area / full_area;

        if (vis_area < min_visible_area || vis_frac < min_visible_frac) continue;

        // warp card + mask
        cv::Mat card_warp;
        cv::warpPerspective(card, card_warp, Hc, image.size(),
                            cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));

        cv::Mat mask_src(card.rows, card.cols, CV_8UC1, cv::Scalar(255));
        cv::Mat mask;
        cv::warpPerspective(mask_src, mask, Hc, image.size(),
                            cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));

        card_warp.copyTo(output, mask);
        card_warp.copyTo(output_boxed, mask);

        // YOLO bbox for overlay region
        std::vector<cv::Point2f> overlay_rect_card = {
            cv::Point2f((float)x0, (float)y1),
            cv::Point2f((float)(x0 + overlay_resized.cols - 1), (float)y1),
            cv::Point2f((float)(x0 + overlay_resized.cols - 1), (float)(y1 + overlay_resized.rows - 1)),
            cv::Point2f((float)x0, (float)(y1 + overlay_resized.rows - 1))
        };

        std::vector<cv::Point2f> overlay_quad_img;
        cv::perspectiveTransform(overlay_rect_card, overlay_quad_img, Hc);

        cv::Rect2f ov_aabb  = quadToAABB(overlay_quad_img);
        cv::Rect2f ov_clip  = clampRectToImage(ov_aabb, image.size());
        if (ov_clip.area() < 25.f) continue;

        float cx = (ov_clip.x + ov_clip.width  * 0.5f) / imgW;
        float cy = (ov_clip.y + ov_clip.height * 0.5f) / imgH;
        float w  = ov_clip.width  / imgW;
        float h  = ov_clip.height / imgH;

        if (w <= 0.f || h <= 0.f) continue;
        cx = std::min(1.f, std::max(0.f, cx));
        cy = std::min(1.f, std::max(0.f, cy));
        w  = std::min(1.f, std::max(0.f, w));
        h  = std::min(1.f, std::max(0.f, h));

        // class id is overlay index
        label_ss << ov_idx << " " << cx << " " << cy << " " << w << " " << h << "\n";

        // draw bbox + points + class id on boxed image
        cv::rectangle(output_boxed,
                      cv::Rect((int)ov_clip.x, (int)ov_clip.y, (int)ov_clip.width, (int)ov_clip.height),
                      cv::Scalar(0,255,0), 2, cv::LINE_AA);

        for (int k = 0; k < 4; k++) {
            cv::circle(output_boxed, overlay_quad_img[k], 4, cv::Scalar(0,0,255), -1, cv::LINE_AA);
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
                  << " <input_folder> <overlay_folder> <output_name> [min_unique_markers]\n"
                  << "  [min_unique_markers] (optional): if provided and > 0, skip images whose\n"
                  << "  number of UNIQUE detected marker IDs is < min_unique_markers.\n";
        return 1;
    }

    fs::path input_folder   = argv[1];
    fs::path overlay_folder = argv[2];
    std::string output_name = argv[3];

    // Optional: minimum number of UNIQUE marker IDs required
    int  min_unique_markers = 0;   // 0 => disabled
    bool use_min_unique     = false;

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
    fs::path bbox_dir   = out_dir / "bounding_box";
    fs::path boxed_dir  = out_dir / ("boxed_" + output_name);

    fs::create_directories(images_dir);
    fs::create_directories(bbox_dir);
    fs::create_directories(boxed_dir);

    // Load overlay paths (sorted => stable class ids)
    std::vector<fs::path> overlay_paths;
    for (const auto& e : fs::directory_iterator(overlay_folder)) {
        if (e.is_regular_file() && hasImageExt(e.path())) overlay_paths.push_back(e.path());
    }
    std::sort(overlay_paths.begin(), overlay_paths.end());

    if (overlay_paths.empty()) {
        std::cerr << "No overlay images found in: " << overlay_folder << "\n";
        return 1;
    }

    // Load overlays, keep filename list aligned to indices
    std::vector<cv::Mat> overlays;
    std::vector<std::string> overlay_names;
    overlays.reserve(overlay_paths.size());
    overlay_names.reserve(overlay_paths.size());

    for (const auto& p : overlay_paths) {
        cv::Mat ov = cv::imread(p.string(), cv::IMREAD_UNCHANGED);
        ov = ensureBGR(ov);
        if (ov.empty()) {
            std::cerr << "Warning: could not read overlay: " << p << " (skipping)\n";
            continue;
        }
        overlays.push_back(ov);
        overlay_names.push_back(p.filename().string());
    }

    if (overlays.empty()) {
        std::cerr << "All overlay reads failed.\n";
        return 1;
    }

    // Write overlay index map
    {
        fs::path map_path = out_dir / "overlay_index_map.txt";
        std::ofstream mf(map_path.string());
        if (!mf.is_open()) {
            std::cerr << "Warning: could not write mapping file: " << map_path << "\n";
        } else {
            mf << "# class_id(overlay_index)\toverlay_filename\n";
            for (size_t i = 0; i < overlay_names.size(); i++) {
                mf << i << "\t" << overlay_names[i] << "\n";
            }
            std::cout << "[info] wrote overlay map: " << map_path << "\n";
        }
    }

    // Collect input images
    std::vector<fs::path> image_paths;
    for (const auto& e : fs::directory_iterator(input_folder)) {
        if (e.is_regular_file() && hasImageExt(e.path())) image_paths.push_back(e.path());
    }
    std::sort(image_paths.begin(), image_paths.end());

    if (image_paths.empty()) {
        std::cerr << "No images found in: " << input_folder << "\n";
        return 1;
    }

    auto dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

    // ---- knobs ----
    const int outputs_per_input_image = 70; // used only when num_unique_markers > 1
    const int duplicates_per_output   = 1;

    // Output resolution (labels will be in this coordinate system)
    const int OUTPUT_W = 1280;
    const int OUTPUT_H = 960;

    std::mt19937 rng(1337);
    std::uniform_int_distribution<int> uni_overlay(0, (int)overlays.size() - 1);

    const int max_attempts_per_input = outputs_per_input_image * 200;

    for (size_t img_idx = 0; img_idx < image_paths.size(); img_idx++) {
        const auto& img_path = image_paths[img_idx];

        cv::Mat image_raw = cv::imread(img_path.string(), cv::IMREAD_UNCHANGED);
        if (image_raw.empty()) {
            std::cerr << "Warning: could not read image: " << img_path << " (skipping)\n";
            continue;
        }
        image_raw = ensureBGR(image_raw);

        // ===== Option A: letterbox FIRST =====
        // From this point onward, everything is done in OUTPUT_W x OUTPUT_H.
        cv::Mat image = letterboxResize(image_raw, OUTPUT_W, OUTPUT_H);
        if (image.empty()) {
            std::cerr << "Warning: letterbox failed: " << img_path << " (skipping)\n";
            continue;
        }

        // Detect markers on letterboxed image (often more robust in grayscale)
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
        unique_ids.erase(std::unique(unique_ids.begin(), unique_ids.end()), unique_ids.end());

        // optional filter
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

        // Decide target outputs:
        // - U == 1 => enumerate ALL K
        // - U >  1 => random sample N
        const bool enumerate_all = (unique_ids.size() == 1);
        const int target_outputs = enumerate_all ? (int)overlays.size() : outputs_per_input_image;

        std::cout << "[info] " << img_path.filename().string()
                  << " unique_ids=" << unique_ids.size()
                  << " overlays=" << overlays.size()
                  << " theoretical_variants=";
        if (theoretical_variants == std::numeric_limits<unsigned long long>::max()) {
            std::cout << ">=2^64-1";
        } else {
            std::cout << theoretical_variants;
        }
        std::cout << " mode=" << (enumerate_all ? "ENUM_ALL(K)" : "RANDOM(N)")
                  << " target_outputs=" << target_outputs
                  << " duplicates=" << duplicates_per_output
                  << " output_res=" << OUTPUT_W << "x" << OUTPUT_H
                  << "\n";

        int produced = 0;

        if (enumerate_all) {
            // Only one unique marker ID => assignment is fully determined by overlay choice.
            const int uid = unique_ids[0];

            for (int ov = 0; ov < (int)overlays.size(); ov++) {
                std::unordered_map<int,int> assignment;
                assignment.reserve(1);
                assignment[uid] = ov;

                cv::Mat out_img, boxed_img;
                std::string yolo_txt;

                bool ok = renderWithAssignment(image, ids, corners, assignment, overlays,
                                               out_img, boxed_img, yolo_txt);
                if (!ok) continue;
                if (yolo_txt.empty()) continue;

                // Already in OUTPUT_W x OUTPUT_H. No second letterbox.
                for (int d = 1; d <= duplicates_per_output; d++) {
                    std::string stem =
                        output_name + "_" +
                        std::to_string(img_idx + 1) + "_" +
                        std::to_string(produced + 1) +
                        "_dup" + std::to_string(d);

                    fs::path out_img_path   = images_dir / (stem + ".jpg");
                    fs::path out_label_path = bbox_dir   / (stem + ".txt");
                    fs::path boxed_img_path = boxed_dir  / (stem + ".jpg");

                    cv::imwrite(out_img_path.string(), out_img);
                    cv::imwrite(boxed_img_path.string(), boxed_img);

                    std::ofstream lf(out_label_path.string());
                    if (lf.is_open()) lf << yolo_txt;
                }

                produced++;
            }

            if (produced < (int)overlays.size()) {
                std::cout << "[warn] Enumerate-all: produced " << produced
                          << " / " << overlays.size()
                          << " (some overlays failed rendering / produced no boxes)\n";
            }

        } else {
            // Random sampling (U > 1)
            int attempts = 0;

            while (produced < target_outputs && attempts < max_attempts_per_input) {
                attempts++;

                std::unordered_map<int,int> assignment;
                assignment.reserve(unique_ids.size());
                for (int uid : unique_ids) assignment[uid] = uni_overlay(rng);

                cv::Mat out_img, boxed_img;
                std::string yolo_txt;

                bool ok = renderWithAssignment(image, ids, corners, assignment, overlays,
                                               out_img, boxed_img, yolo_txt);
                if (!ok) continue;
                if (yolo_txt.empty()) continue;

                for (int d = 1; d <= duplicates_per_output; d++) {
                    std::string stem =
                        output_name + "_" +
                        std::to_string(img_idx + 1) + "_" +
                        std::to_string(produced + 1) +
                        "_dup" + std::to_string(d);

                    fs::path out_img_path   = images_dir / (stem + ".jpg");
                    fs::path out_label_path = bbox_dir   / (stem + ".txt");
                    fs::path boxed_img_path = boxed_dir  / (stem + ".jpg");

                    cv::imwrite(out_img_path.string(), out_img);
                    cv::imwrite(boxed_img_path.string(), boxed_img);

                    std::ofstream lf(out_label_path.string());
                    if (lf.is_open()) lf << yolo_txt;
                }

                produced++;
            }

            if (produced < target_outputs) {
                std::cout << "[warn] Random mode: could not reach " << target_outputs
                          << " outputs for this image (too many failed renders).\n";
            }
        }
    }

    std::cout << "Done.\n"
              << "Images: " << images_dir << "\n"
              << "BBoxes: " << bbox_dir << "\n"
              << "Boxed:  " << boxed_dir << "\n";
    return 0;
}
