// MIT License
//
// Copyright (c) 2025 Luca Lobefaro
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <src/core/Registration.hpp>
#include <src/core/Utilities.hpp>

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: cilantro_nicp source.ply target.ply [--out path] "
                     "[--time path] [--no-vis]\n";
        return -1;
    }

    std::string src_path = argv[1];
    std::string dst_path = argv[2];
    std::string out_path = "data/deformed_src.ply";
    std::string time_path;  // empty => don't save
    bool no_vis = false;

    // parse optional args
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--no-vis") {
            no_vis = true;
        } else if (a == "--out" && i + 1 < argc) {
            out_path = argv[++i];
        } else if (a == "--time" && i + 1 < argc) {
            time_path = argv[++i];
        } else {
            std::cerr << "Unknown option: " << a << "\n";
            return -1;
        }
    }

    // Load clouds
    const auto src = load_pcd(src_path);
    const auto dst = load_pcd(dst_path);

    // Visualize before deformation (optional)
    if (!no_vis) visualize_pcds(src, dst);

    // Deform source + timing
    std::cout << "DEFORMING" << std::endl;
    const auto t0 = std::chrono::steady_clock::now();
    const auto deformed_src = deform_cloud(src, dst);
    const auto t1 = std::chrono::steady_clock::now();
    const double deform_sec = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "DONE" << std::endl;
    std::cout << "Deform time (s): " << deform_sec << std::endl;

    // Ensure output directory exists (out)
    {
        std::filesystem::path op(out_path);
        if (op.has_parent_path()) {
            std::filesystem::create_directories(op.parent_path());
        }
    }

    // Save result
    save_pcd(deformed_src, out_path);
    std::cout << "Result saved in " << out_path << std::endl;

    // Save time if requested
    if (!time_path.empty()) {
        std::filesystem::path tp(time_path);
        if (tp.has_parent_path()) {
            std::filesystem::create_directories(tp.parent_path());
        }
        std::ofstream fout(time_path);
        if (!fout) {
            std::cerr << "Failed to open time file: " << time_path << std::endl;
            return -1;
        }
        fout << deform_sec << "\n";
        fout.close();
        std::cout << "Time saved in " << time_path << std::endl;
    }

    // Visualize after deformation (optional)
    if (!no_vis) visualize_pcds(deformed_src, dst);

    return 0;
}
