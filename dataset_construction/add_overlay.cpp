// synth.cpp
// Build:
//   g++ -std=c++17 synth.cpp `pkg-config --cflags --libs opencv4` -o synth
// Run:
//   ./synth <input_image> <overlay_image> [output_image] [yolo_label]
//
// Output:
//   - output image (default: output.jpg)
//   - debug image with bbox + quad points: boxed_<output_image>
//   - YOLO label file for OVERLAY region only (default: output.txt)

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>

static cv::Point2f normalizeVec(cv::Point2f v) {
    float len = std::sqrt(v.x * v.x + v.y * v.y);
    if (len < 1e-6f) return cv::Point2f(0.f, 0.f);
    return v * (1.0f / len);
}

// If you want to allow partial cards, you can remove this check.
static bool quadInsideImage(const std::vector<cv::Point2f>& q, const cv::Size& sz) {
    for (const auto& p : q) {
        if (p.x < 0 || p.y < 0 || p.x >= sz.width || p.y >= sz.height) return false;
    }
    return true;
}

static cv::Mat ensureBGR(const cv::Mat& src) {
    if (src.empty()) return src;
    if (src.channels() == 3) return src;
    cv::Mat out;
    cv::cvtColor(src, out, cv::COLOR_GRAY2BGR);
    return out;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr
            << "Usage: " << argv[0]
            << " <input_image> <overlay_image> [output_image] [yolo_label]\n";
        return 1;
    }

    const std::string input_path = argv[1];
    const std::string overlay_path = argv[2];

    // Optional args with defaults
    const std::string output_path = (argc >= 4) ? argv[3] : "output.jpg";
    const std::string yolo_label_path = (argc >= 5) ? argv[4] : "output.txt";

    const int class_id = 0;

    cv::Mat image = cv::imread(input_path);
    if (image.empty()) {
        std::cerr << "Could not read input image: " << input_path << "\n";
        return -1;
    }
    image = ensureBGR(image);

    cv::Mat output = image.clone();
    cv::Mat output_boxed = image.clone();

    cv::Mat overlay = cv::imread(overlay_path);
    if (overlay.empty()) {
        std::cerr << "Could not read overlay image: " << overlay_path << "\n";
        return -1;
    }
    overlay = ensureBGR(overlay);

    // ArUco detect
    cv::Ptr<cv::aruco::Dictionary> dictionary =
        cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    cv::aruco::detectMarkers(image, dictionary, corners, ids);

    if (ids.empty()) {
        std::cout << "No markers detected\n";
        return 0;
    }

    std::cout << "Detections: " << ids.size() << "\nIDs: ";
    for (int id : ids) std::cout << id << " ";
    std::cout << "\n";

    std::ofstream label_file(yolo_label_path);
    if (!label_file.is_open()) {
        std::cerr << "Could not open " << yolo_label_path << " for writing\n";
        return -1;
    }

    // ---- styling knobs ----
    const int padding = 20;
    const int gap = 14;
    const int border_px = 4;

    // marker extracted to a fixed square patch
    const int marker_patch_size = 320;

    // kept from your snippet (unused, but harmless)
    const float cardHeightFactor = 2.7f;
    const float pad_out_px = 30.0f;
    const float top_lift_px = 30.0f;
    const float bottom_pad_px = 30.0f;
    (void)cardHeightFactor;
    (void)pad_out_px;
    (void)top_lift_px;
    (void)bottom_pad_px;

    const float imgW = (float)image.cols;
    const float imgH = (float)image.rows;

    int written = 0;

    for (int i = 0; i < (int)ids.size(); i++) {
        int id = ids[i];

        // Marker quad order from ArUco is typically TL,TR,BR,BL
        std::vector<cv::Point2f> m = corners[i];

        cv::Point2f topMid = 0.5f * (m[0] + m[1]);
        cv::Point2f bottomMid = 0.5f * (m[3] + m[2]);
        cv::Point2f leftMid = 0.5f * (m[0] + m[3]);
        cv::Point2f rightMid = 0.5f * (m[1] + m[2]);

        cv::Point2f downVec = normalizeVec(bottomMid - topMid);
        cv::Point2f rightVec = normalizeVec(rightMid - leftMid);

        if (downVec == cv::Point2f(0, 0) || rightVec == cv::Point2f(0, 0))
            continue;

        // ---- 1) Extract ONLY marker as a square patch ----
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

        // ---- 2) Build card image: top=marker_patch, bottom=overlay resized ----
        cv::Mat overlay_resized;
        {
            int targetW = marker_patch.cols;
            int newH = (int)std::round(overlay.rows * (targetW / (double)overlay.cols));
            cv::resize(overlay, overlay_resized, cv::Size(targetW, newH), 0, 0, cv::INTER_AREA);
        }

        int contentW = marker_patch.cols;
        int contentH = marker_patch.rows + gap + overlay_resized.rows;

        int cardW = contentW + 2 * (padding + border_px);
        int cardH = contentH + 2 * (padding + border_px);

        cv::Mat card(cardH, cardW, CV_8UC3, cv::Scalar(255, 255, 255));

        // border
        cv::rectangle(
            card,
            cv::Rect(0, 0, cardW, cardH),
            cv::Scalar(0, 0, 0),
            border_px,
            cv::LINE_AA
        );

        // place marker + overlay in the card
        int x0 = border_px + padding;
        int y0 = border_px + padding;

        marker_patch.copyTo(card(cv::Rect(x0, y0, marker_patch.cols, marker_patch.rows)));

        int y1 = y0 + marker_patch.rows + gap;
        overlay_resized.copyTo(card(cv::Rect(x0, y1, overlay_resized.cols, overlay_resized.rows)));

        // ==========================================================
        // Compute Hc by PINNING marker region in the card to the
        // detected marker quad m in the image.
        // ==========================================================
        std::vector<cv::Point2f> marker_rect_card = {
            cv::Point2f((float)x0, (float)y0), // TL
            cv::Point2f((float)(x0 + marker_patch.cols - 1), (float)y0), // TR
            cv::Point2f((float)(x0 + marker_patch.cols - 1), (float)(y0 + marker_patch.rows - 1)), // BR
            cv::Point2f((float)x0, (float)(y0 + marker_patch.rows - 1)) // BL
        };

        cv::Mat Hc = cv::findHomography(marker_rect_card, m);
        if (Hc.empty()) continue;

        // Optional: require whole card to be fully inside the image
        std::vector<cv::Point2f> card_corners = {
            cv::Point2f(0.f, 0.f),
            cv::Point2f((float)card.cols - 1, 0.f),
            cv::Point2f((float)card.cols - 1, (float)card.rows - 1),
            cv::Point2f(0.f, (float)card.rows - 1)
        };
        std::vector<cv::Point2f> card_corners_img;
        cv::perspectiveTransform(card_corners, card_corners_img, Hc);
        // if (!quadInsideImage(card_corners_img, image.size()))
        //     continue;

        // ---- Warp whole card onto image ----
        cv::Mat card_warp;
        cv::warpPerspective(
            card,
            card_warp,
            Hc,
            image.size(),
            cv::INTER_LINEAR,
            cv::BORDER_CONSTANT,
            cv::Scalar(0, 0, 0)
        );

        // mask for pasting
        cv::Mat mask_src(card.rows, card.cols, CV_8UC1, cv::Scalar(255));
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

        card_warp.copyTo(output, mask);
        card_warp.copyTo(output_boxed, mask);

        // ==========================================================
        // YOLO BBOX FOR OVERLAY ONLY: project overlay-rect via Hc
        // ==========================================================
        std::vector<cv::Point2f> overlay_rect_card = {
            cv::Point2f((float)x0, (float)y1), // TL
            cv::Point2f((float)(x0 + overlay_resized.cols - 1), (float)y1), // TR
            cv::Point2f((float)(x0 + overlay_resized.cols - 1), (float)(y1 + overlay_resized.rows - 1)), // BR
            cv::Point2f((float)x0, (float)(y1 + overlay_resized.rows - 1)) // BL
        };

        std::vector<cv::Point2f> overlay_quad_img;
        cv::perspectiveTransform(overlay_rect_card, overlay_quad_img, Hc);

        float min_x = overlay_quad_img[0].x, max_x = overlay_quad_img[0].x;
        float min_y = overlay_quad_img[0].y, max_y = overlay_quad_img[0].y;

        for (int k = 1; k < 4; k++) {
            min_x = std::min(min_x, overlay_quad_img[k].x);
            max_x = std::max(max_x, overlay_quad_img[k].x);
            min_y = std::min(min_y, overlay_quad_img[k].y);
            max_y = std::max(max_y, overlay_quad_img[k].y);
        }

        // clamp
        min_x = std::max(0.f, min_x);
        min_y = std::max(0.f, min_y);
        max_x = std::min(imgW - 1.f, max_x);
        max_y = std::min(imgH - 1.f, max_y);

        float bw = max_x - min_x;
        float bh = max_y - min_y;

        if (bw <= 1.0f || bh <= 1.0f) continue;

        float cx = (min_x + max_x) * 0.5f / imgW;
        float cy = (min_y + max_y) * 0.5f / imgH;
        float w  = bw / imgW;
        float h  = bh / imgH;

        label_file << class_id << " " << cx << " " << cy << " " << w << " " << h << "\n";
        written++;

        // debug: draw bbox + quad points
        cv::rectangle(
            output_boxed,
            cv::Rect((int)min_x, (int)min_y, (int)bw, (int)bh),
            cv::Scalar(0, 255, 0),
            2,
            cv::LINE_AA
        );
        for (int k = 0; k < 4; k++) {
            cv::circle(output_boxed, overlay_quad_img[k], 4, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
        }

        std::cout << "Rendered card for detection " << i << " (id " << id << ")\n";
    }

    label_file.close();

    cv::imwrite(output_path, output);
    cv::imwrite("boxed_" + output_path, output_boxed);

    std::cout << "Saved " << output_path << "\n";
    std::cout << "Saved boxed_" << output_path << "\n";
    std::cout << "Saved " << yolo_label_path << " (overlay boxes: " << written << ")\n";

    return 0;
}
