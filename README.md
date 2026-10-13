# PCICF detection pipeline, C++17 port

## A native C++ implementation of PCICF project. The 'detection_pipeline' is working 
### TODO: 1)make the pipeline running on GPU (currently build and run on cpu) 2)add the port for 'evaluation_pipeline.' 3) add a docker wrapper. 

This ports `detection_pipeline.py` and its runtime dependencies in
[Claud1234/PCICF, branch devel](https://github.com/Claud1234/PCICF/tree/devel).
Source snapshot: `46dd378ca0b6137f1874e7ace9b35e9e4f3f198e`.


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

T

## Build

Requires a C++17 compiler, CMake 3.18+, **OpenCV 4.10.0** with `core`, `imgproc`,
`imgcodecs`, `dnn`, `video`, and `calib3d`, yaml-cpp, nlohmann/json, and Eigen3.
CMake requires exactly 4.10.0 and prints the selected version and package path.
It uses the normal CMake package search; alternative installations are selected
explicitly with `OpenCV_DIR`.

**Before configuring or compiling, deactivate all Conda environments,
including `base`.** Repeat `conda deactivate` until `CONDA_PREFIX` is empty,
or use this Bash loop:

```sh
while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do
    conda deactivate || break
done
```

On Debian/Ubuntu, install the development dependencies and check the system
OpenCV version:

```sh
sudo apt-get install build-essential cmake git pkg-config libopencv-dev libyaml-cpp-dev nlohmann-json3-dev libeigen3-dev
pkg-config --modversion opencv4
```

If the system version is **4.10.0**, build from this repository root:

```sh
cmake -UOpenCV_DIR -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

`-UOpenCV_DIR` clears a previously selected installation from the cache so CMake
can use the system package. If the shell is already inside `build`, use
`cmake --build . --parallel` and `ctest --output-on-failure`.

### If system OpenCV is not 4.10.0

Choose one of the two approaches below. Ubuntu 24.04 supplies OpenCV 4.6.0,
which fails to import the supplied YOLO11x model; Ubuntu 26.04 supplies 4.10.0.
See the official [Ubuntu 24.04 package](https://packages.ubuntu.com/noble/libopencv-dev)
and [Ubuntu 26.04 package](https://packages.ubuntu.com/en/resolute/libopencv-dev).

`OpenCV_DIR` must name an existing directory containing `OpenCVConfig.cmake`
for version 4.10.0. Do not copy `/path/to/...` placeholders as installation
paths. Installing a Python `opencv-python` wheel does not provide this C++
development package.

#### Option 1: Use a Conda environment

With Conda already installed, create an environment containing the
[conda-forge OpenCV C++ package](https://anaconda.org/conda-forge/libopencv/files?version=4.10.0).
Activate it briefly to capture its actual path, then deactivate all environments
before compiling. Run the following from this repository root in the same shell:

```sh
conda create -n pcicf-opencv410 --override-channels -c conda-forge "libopencv=4.10.0"
conda activate pcicf-opencv410
pcicf_opencv_prefix="$CONDA_PREFIX"
pcicf_conda_base="$(conda info --base)"
while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do
    conda deactivate || break
done

cmake -S . -B build-conda -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_IGNORE_PREFIX_PATH="$pcicf_conda_base" \
    -DOpenCV_DIR="$pcicf_opencv_prefix/lib/cmake/opencv4"
cmake --build build-conda --parallel
ctest --test-dir build-conda --output-on-failure
```

To use an existing environment, skip `conda create` and replace the environment
name in `conda activate`. The compiler and other project dependencies still
come from the system; `CMAKE_IGNORE_PREFIX_PATH` excludes base Conda dependencies.
See [Conda's environment documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
Run this build using `./build-conda/detection_pipeline`.

On this workstation, the existing `pcicf_py12_torch250` environment already
contains OpenCV 4.10.0. After deactivating all environments, use these exact
commands instead of creating a new environment:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_IGNORE_PREFIX_PATH=/home/claude/miniconda3 \
    -DOpenCV_DIR=/home/claude/miniconda3/envs/pcicf_py12_torch250/lib/cmake/opencv4
cmake --build build --parallel
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
    ctest --test-dir build --output-on-failure
```

This existing environment has an older `libstdc++` than GCC 13/system yaml-cpp
requires. The preload selects the system C++ runtime and is also required when
running this workstation's executable. These paths are specific to this machine.

#### Option 2: Install OpenCV 4.10.0 natively

Build the official 4.10.0 release with the system compiler and install it
system-wide under `/opt/opencv-4.10.0`. This leaves Ubuntu's package-managed
OpenCV installed separately. Deactivate all Conda environments first, then
run these commands from this repository root:

```sh
sudo apt-get install build-essential cmake git libjpeg-dev libpng-dev zlib1g-dev
pcicf_opencv_src="$HOME/src/opencv-4.10.0"
mkdir -p "$HOME/src"
git clone --depth 1 --branch 4.10.0 https://github.com/opencv/opencv.git "$pcicf_opencv_src"
cmake -S "$pcicf_opencv_src" -B "$pcicf_opencv_src/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/opencv-4.10.0 \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DBUILD_LIST=core,imgproc,imgcodecs,dnn,video,calib3d \
    -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
    -DBUILD_opencv_apps=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF
cmake --build "$pcicf_opencv_src/build" --parallel 2
sudo cmake --install "$pcicf_opencv_src/build"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DOpenCV_DIR=/opt/opencv-4.10.0/lib/cmake/opencv4
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The source directory must be new for `git clone`; if you already downloaded
that release, reuse its source directory and start at the OpenCV configure step.
The native build uses system C++ libraries and does not need the Conda runtime
preload. See the [official Linux installation guide](https://docs.opencv.org/4.10.0/d7/d9f/tutorial_linux_install.html)
and [build options](https://docs.opencv.org/4.10.0/db/d05/tutorial_config_reference.html).


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
Use the OpenCV 4.10.0 build commands above. Installing a Python
`opencv-python` wheel does not change the OpenCV libraries used by this C++
executable.

## Run

Run from this repository root. Keep the original `config.json`,
`datasets/pie_splits`, and datasets there. On this workstation:

```sh
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
    ./build/detection_pipeline -d pie -o outputs/pie_cpp -s
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
    ./build/detection_pipeline -d MoreSMIRK -o outputs/smirk_cpp -s
```

The original `-d`, `-o`, and `-s` flags retain their meanings. `--help` describes
additional options. `--device cpu` is the default; `--device cuda` selects
NVIDIA GPU inference. The package does not include model weights or image datasets.


