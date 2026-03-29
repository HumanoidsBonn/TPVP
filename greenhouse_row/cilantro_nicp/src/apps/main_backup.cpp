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
#include <iostream>

#include <src/core/Registration.hpp>
#include <src/core/Utilities.hpp>

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: make run ARGS=\"source.ply target.ply\""
                  << std::endl;
        return -1;
    }

    // Load clouds
    const auto src = load_pcd(argv[1]);
    const auto dst = load_pcd(argv[2]);

    // Visualize before deformation
    visualize_pcds(src, dst);

    // Deform source
    std::cout << "DEFORMING" << std::endl;
    const auto deformed_src = deform_cloud(src, dst);
    std::cout << "DONE" << std::endl;

    // Save result
    const std::string output_filename = "data/deformed_src.ply";
    save_pcd(deformed_src, output_filename);
    std::cout << "Result saved in " << output_filename << std::endl;

    // Visualize after deformation
    visualize_pcds(deformed_src, dst);

    return 0;
}
