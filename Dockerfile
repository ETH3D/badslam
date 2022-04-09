FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

ENV OPENCV_VERSION="3.4.6"
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update && apt-get -qq install -y --no-install-recommends \
        build-essential     \
        cmake               \
        git                 \
        wget                \
        unzip               \
        yasm                \
        pkg-config          \
        libswscale-dev      \
        libtbb2             \
        libtbb-dev          \
        libjpeg-dev         \
        libpng-dev          \
        libtiff-dev         \
        libopenjp2-7-dev    \
        libavformat-dev     \
        libssl-dev          \
        libpq-dev           \
        libeigen3-dev       \
        libgl1-mesa-dev     \
        libboost1.67-all-dev\
        libglew-dev         \
        qtbase5-dev         \
        libqt5x11extras5-dev\
        libqt5opengl5-dev   \
        libsuitesparse-dev  \
        meshlab

RUN wget https://www.python.org/ftp/python/3.9.9/Python-3.9.9.tgz
RUN tar xzvf Python-3.9.9.tgz
WORKDIR Python-3.9.9
RUN ./configure
RUN make install
WORKDIR /

RUN python3 -m pip install numpy \
    && wget -q https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip -O opencv.zip \
    && unzip -qq opencv.zip -d /opt \
    && rm -rf opencv.zip \
    && cmake \
        -D BUILD_TIFF=ON            \
        -D BUILD_opencv_java=OFF    \
        -D WITH_CUDA=OFF            \
        -D WITH_OPENGL=ON           \
        -D WITH_OPENCL=OFF          \
        -D WITH_IPP=ON              \
        -D WITH_TBB=ON              \
        -D WITH_EIGEN=ON            \
        -D WITH_V4L=ON              \
        -D BUILD_TESTS=OFF          \
        -D BUILD_PERF_TESTS=OFF     \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=$(python3.9 -c "import sys; print(sys.prefix)") \
        -D PYTHON_EXECUTABLE=$(which python3.9) \
        -D PYTHON_INCLUDE_DIR=$(python3.9 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
        -D PYTHON_PACKAGES_PATH=$(python3.9 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
        /opt/opencv-${OPENCV_VERSION} \
    && make -j$(nproc) \
    && make install \
    && rm -rf /opt/build/* \
    && rm -rf /opt/opencv-${OPENCV_VERSION} \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get -qq autoremove \
    && apt-get -qq clean

RUN git clone https://github.com/dorian3d/DLib
WORKDIR DLib
RUN git reset --hard b6c28fb
RUN mkdir build
WORKDIR build
RUN cmake -E env CXXFLAGS="-march=native" cmake .. && make -j install
WORKDIR /

RUN git clone https://github.com/laurentkneip/opengv.git
WORKDIR opengv
RUN git reset --hard 91f4b19
RUN mkdir build
WORKDIR build
RUN cmake -E env CXXFLAGS="-march=native" cmake .. && make -j install
WORKDIR /

RUN git clone https://github.com/RainerKuemmerle/g2o.git
WORKDIR g2o
RUN git reset --hard b1ba729
RUN mkdir build
WORKDIR build
RUN cmake -DBUILD_WITH_MARCH_NATIVE=ON .. && make -j install

ADD . /home/badslam
RUN mkdir build_RelWithDebInfo
WORKDIR /home/badslam/build_RelWithDebInfo

ARG CUDA_ARCH
ENV CUDA_ARCH ${CUDA_ARCH}
RUN cmake -E env CXXFLAGS='-march=native' cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CUDA_FLAGS="-arch=${CUDA_ARCH}" .. && make -j
