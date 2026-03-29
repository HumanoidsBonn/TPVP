# A fast way to perform non-rigid ICP with Cilantro library
Please refer to [cilantro](https://github.com/kzampog/cilantro/tree/master) for the usage of the main library. Here, we just want to use the non-rigid ICP implementation from it.

## Dependencies
You need a system-wise installation of [Open3D](https://github.com/isl-org/Open3D). And, of course, compiler and company for c++.

## Installion
cd cilantro_nicp_rel/src
mkdir build
cd build
cmake ..
make

## Demo
```
./src/build/apps/cilantro_nicp ./data/room1_20250918_r0_v0_src.ply ./data/room1_20250918_r0_v0_tgt.ply --out ./log/room1_20250918_r0_v0_nricp.ply --time ./log/room1_20250918_r0_v0_nricp_time.txt --no-vis
```

## Interact with View_planning_simulator
```
python run_test_all_parallel_nricptest_greenhouse.py
```
### Example Run
Input with room1_20250918 and -1.