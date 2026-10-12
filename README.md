# PCICF detection pipeline, C++17 port

This ports `detection_pipeline.py` and its runtime dependencies in
[Claud1234/PCICF, branch devel](https://github.com/Claud1234/PCICF/tree/devel).
Source snapshot: `46dd378ca0b6137f1874e7ace9b35e9e4f3f198e`.

The executable performs YOLO11 detection, native BoT-SORT tracking, complete
crossing selection, six-cell ROI occupancy, Morton encoding, semicolon-separated
CSV export, PNG plotting, and optional bounding-box visualization.

There is no Python dependency at runtime. Exporting the original pretrained model
to ONNX is a separate, one-time operation.

## Replacements

| Original | C++ implementation |
|---|---|
| argparse | Small command-line parser |
| os/glob | `std::filesystem`, lexical sorting |
| json | nlohmann/json |
| yaml | yaml-cpp |
| pandas | Typed rows in `std::vector`, `std::ofstream` |
| NumPy boxes and Boolean arrays | C++ structs and `std::array` |
| NumPy image masks | ROI-local OpenCV masks |
| zCurve | Manual six-dimensional bit interleaving |
| matplotlib | PNG plot drawn with OpenCV |
| PyTorch YOLO inference | ONNX model executed with OpenCV DNN |
| Ultralytics BoT-SORT | Native C++ implementation with Eigen Kalman algebra |

This ports the detection pipeline. The separate `evaluation_pipeline.py`,
dataset-generation scripts, and notebooks are outside this package's scope.

## Build

Requires a C++17 compiler, CMake 3.18+, OpenCV with `core`, `imgproc`, `imgcodecs`,
`dnn`, `video`, and `calib3d`, yaml-cpp, nlohmann/json, and Eigen3.

On Debian/Ubuntu, run these commands from this directory (the one containing
`CMakeLists.txt`):

```sh
sudo apt-get install build-essential cmake libopencv-dev libyaml-cpp-dev nlohmann-json3-dev libeigen3-dev
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

If the shell is already inside `build`, use `cmake --build . --parallel` and
`ctest --output-on-failure`.

If Conda is active, CMake can mix Conda dependencies with system OpenCV. This
can add Conda's library directory to the linker search path and produce
undefined references to `curl_*@CURL_OPENSSL_4` from system GDAL or HDF5.
For a build using the Debian/Ubuntu dependencies above, reconfigure with a
fresh cache and exclude the Conda prefix (CMake 3.24+):

```sh
cmake --fresh -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_IGNORE_PREFIX_PATH="$CONDA_PREFIX"
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Run this while `$CONDA_PREFIX` identifies the active Conda environment. With
older CMake versions, deactivate Conda and configure a new build directory
instead, so cached dependency paths are not reused. If `LD_LIBRARY_PATH` or
`LIBRARY_PATH` explicitly contains Conda directories, clear those overrides in
the build shell as well.

Windows: install these same dependencies using vcpkg or another CMake-compatible
package manager, configure with that manager's toolchain file, and build with
`cmake --build build --config Release`. For a multi-configuration generator,
the executable will usually be `build/Release/detection_pipeline.exe`.

## Export the model once

Use the repository's pinned Ultralytics version, `8.3.203`, in its Python
environment. Export the same `yolo11x.pt` weights:

```python
from ultralytics import YOLO

YOLO("yolo11x.pt").export(
    format="onnx",
    imgsz=(480, 640),
    batch=1,
    dynamic=False,
    simplify=False,
    opset=12,
    nms=False,
)
```

Place `yolo11x.onnx` in the PCICF repository root, or pass its path with `--model`.
The default model input is **height 480, width 640**, matching the original
640-by-480 datasets and the rectangular preprocessing of their `.pt` model.
For a square 640-by-640 export, pass `--model-height 640`.
Export dimensions and command-line model dimensions must agree.

The real YOLO11x export requires a newer OpenCV DNN importer than Ubuntu's
OpenCV 4.6: that version can fail on shape calculations while loading the model
or running inference. OpenCV 4.10 was verified to import and execute this export.
If a newer C++ OpenCV installation is available, select its CMake package in a
separate build directory:

```sh
cmake -S . -B build-opencv410 -DCMAKE_BUILD_TYPE=Release \
    -DOpenCV_DIR=/path/to/opencv/lib/cmake/opencv4
cmake --build build-opencv410 --parallel
ctest --test-dir build-opencv410 --output-on-failure
```

Use `build-opencv410/detection_pipeline` to run that build. Installing a Python
`opencv-python` wheel does not change the OpenCV libraries used by this C++
executable.

On this workstation, OpenCV 4.10 is available in the
`pcicf_py12_torch250` Conda environment. Configure it while excluding the base
Conda prefix to keep the existing system yaml-cpp dependency:

```sh
cmake -S . -B build-opencv410 -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_IGNORE_PREFIX_PATH=/home/claude/miniconda3 \
    -DOpenCV_DIR=/home/claude/miniconda3/envs/pcicf_py12_torch250/lib/cmake/opencv4
cmake --build build-opencv410 --parallel
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
    ctest --test-dir build-opencv410 --output-on-failure
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
    ./build-opencv410/detection_pipeline -d MoreSMIRK -o outputs/smirk_cpp
```

The preload selects the system C++ runtime required by GCC 13 and system
yaml-cpp; this Conda environment supplies an older `libstdc++`. These paths are
specific to this workstation.

## Run

Run from the original PCICF repository root. Keep the original `config.json`,
`datasets/pie_splits`, and datasets there.

```sh
./pcicf_cpp/build/detection_pipeline -d pie -o outputs/pie_cpp -s
./pcicf_cpp/build/detection_pipeline -d MoreSMIRK -o outputs/smirk_cpp
```

The original `-d`, `-o`, and `-s` flags retain their meanings. `--help` describes
additional options. Inference currently targets the CPU; the package does not
include model weights or image datasets.

## Preserved behavior

- PIE end frames are exclusive, just like `range(start_frame, end_frame)`.
- MoreSMIRK frames are sorted lexicographically, not numerically.
- Track histories are filtered after the whole sequence: horizontal motion must
  exceed `yolo_track_threshold` and cross the configured image center.
- The timestamp starts at zero and increases by 33 ms per input frame.
- Only class zero (person) contributes to histories; tracking sees all classes.
- Floating box coordinates and history IDs use the original uint16 truncation
  and wrap semantics. No new clipping is applied to tracked boxes.
- ROI intersections use half-open image ranges. A pixel covered by several boxes
  is counted once. Pixels beyond the image are excluded from the numerator;
  the denominator remains the configured ROI area.
- Occupancy is strictly `coverage > sfc_bounding`.
- Binary occupancy is multiplied by **10** before Morton interleaving. Directly
  interleaving zero/one values would produce different codes.
- CSV columns, delimiter, and filenames match the original. A sequence with no
  accepted crossing produces header-only CSV files and an empty plot.
- Visualizations use the original `outputs/<dataset>_visual` location. An input
  outside a `datasets` directory is placed there by filename, avoiding the
  original string-replacement behavior that can overwrite an input image.
- The accidental global `configs` reference is replaced with the passed config.

PNG plots preserve red Morton event markers and green frame scatter points with
separate y axes; font, dimensions, and tick placement differ from matplotlib.
The label `Frequency` is retained, although the original event plot is not a
frequency histogram.

## Tracking and reproducibility

The repository pins Ultralytics 8.3.203. This port targets that version's
`botsort.yaml`: high/low/new confidence thresholds 0.25/0.1/0.25, 30-frame lost
buffer, fused-score first association at 0.8, second association at 0.5,
unconfirmed association at 0.7, XYWH Kalman filtering, sparse optical-flow motion
compensation, and ReID disabled. The detector uses confidence 0.1, class-aware
NMS IoU 0.7, and at most 300 detections.

The matching solver is a native Hungarian implementation with the same
limited-cost objective as `lapjv`. Equally optimal assignments can break ties
differently. Eigen/OpenCV calculations and ONNX inference can also produce
small numeric differences relative to SciPy/PyTorch; these can affect boxes,
track identities, or threshold decisions. This is not a claim of bit-for-bit
identity on untested real sequences.

Do not change to a newer Ultralytics default tracker when exporting reference
data: that would change the comparison target. ReID and alternative GMC methods
other than `none` are outside this port's supported configuration.

## Replay and compare

`--dump-tracks` writes `<sequence>_tracks.json` alongside the CSVs. It contains
normalized input paths and rows `[x1,y1,x2,y2,track_id,score,class]`:

```json
{
  "frames": {
    "datasets/pie/pie_003/598.png": [[110, 280, 160, 400, 1, 0.91, 0]],
    "datasets/pie/pie_003/599.png": []
  }
}
```

Supply `--tracks FILE` to reproduce SFC/CSV output from already tracked boxes
without loading an ONNX model. Each processed input path must exist as a key,
including frames with empty results. Frame images must still be available.

`--detections FILE` uses the same format with detection rows
`[x1,y1,x2,y2,score,class]`, feeding the native tracker instead of the detector.
This separates tracker comparisons from inference differences.
Use `--gmc none` only when the reference comparison also disables motion
compensation.

## Validation

See `VALIDATION.md` for the checks run for this package and their limits.

## Sources

- [PCICF source](https://github.com/Claud1234/PCICF/tree/devel)
- [Pinned Ultralytics tracker source](https://github.com/ultralytics/ultralytics/tree/v8.3.203/ultralytics/trackers)
- [OpenCV YOLO deployment documentation](https://docs.opencv.org/4.x/da/d9d/tutorial_dnn_yolo.html)
- [yaml-cpp](https://github.com/jbeder/yaml-cpp)
- [nlohmann/json](https://json.nlohmann.me/integration/cmake/)

