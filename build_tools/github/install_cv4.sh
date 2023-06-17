#!/bin/bash

echo "Installing OpenCV..."

# Specify OpenCV version
cvVersion="4.5.4"

curl -O -L https://download.qt.io/official_releases/qt/5.15/5.15.0/single/qt-everywhere-src-5.15.0.tar.xz && \
    tar -xf qt-everywhere-src-5.15.0.tar.xz && \
    cd qt-everywhere* && \
    export MAKEFLAGS=-j$(nproc) && \
    ./configure -prefix /opt/Qt5.15.0 -release -opensource -confirm-license -qtnamespace QtOpenCVPython -xcb -xcb-xlib -bundled-xcb-xinput -no-openssl -no-dbus -skip qt3d -skip qtactiveqt -skip qtcanvas3d -skip qtconnectivity -skip qtdatavis3d -skip qtdoc -skip qtgamepad -skip qtgraphicaleffects -skip qtimageformats -skip qtlocation -skip qtmultimedia -skip qtpurchasing -skip qtqa -skip qtremoteobjects -skip qtrepotools -skip qtscript -skip qtscxml -skip qtsensors -skip qtserialbus -skip qtserialport -skip qtspeech -skip qttranslations -skip qtwayland -skip qtwebchannel -skip qtwebengine -skip qtwebsockets -skip qtwebview -skip xmlpatterns -skip declarative -make libs && \
    make && \
    make install && \
    cd .. && \
    rm -rf qt-everywhere-src-5.15.0 && \
    rm qt-everywhere-src-5.15.0.tar.xz

QTDIR /opt/Qt5.15.0
export PATH="$QTDIR/bin:$PATH"

mkdir ~/ffmpeg_sources && \
    cd ~/ffmpeg_sources && \
    curl -O -L https://github.com/openssl/openssl/archive/OpenSSL_1_1_1g.tar.gz && \
    tar -xf OpenSSL_1_1_1g.tar.gz && \
    cd openssl-OpenSSL_1_1_1g && \
    ./config --prefix="$HOME/ffmpeg_build" --openssldir="$HOME/ffmpeg_build" no-pinshared shared zlib && \
    make -j$(getconf _NPROCESSORS_ONLN) && \
    # skip installing documentation
    make install_sw && \
    rm -rf ~/openssl_build

cd ~/ffmpeg_sources && \
    curl -O -L http://www.nasm.us/pub/nasm/releasebuilds/2.15.04/nasm-2.15.04.tar.bz2 && \
    tar -xf nasm-2.15.04.tar.bz2 && cd nasm-2.15.04 && ./autogen.sh && \
    ./configure --prefix="$HOME/ffmpeg_build" --bindir="$HOME/bin" && \
    make -j$(getconf _NPROCESSORS_ONLN) && \
    make install

cd ~/ffmpeg_sources && \
    curl -O -L http://www.tortall.net/projects/yasm/releases/yasm-1.3.0.tar.gz && \
    tar -xf yasm-1.3.0.tar.gz && \
    cd yasm-1.3.0 && \
    ./configure --prefix="$HOME/ffmpeg_build" --bindir="$HOME/bin" && \
    make -j$(getconf _NPROCESSORS_ONLN) && \
    make install

cd ~/ffmpeg_sources && \
    git clone --depth 1 https://chromium.googlesource.com/webm/libvpx.git && \
    cd libvpx && \
    ./configure --prefix="$HOME/ffmpeg_build" --disable-examples --disable-unit-tests --enable-vp9-highbitdepth --as=yasm --enable-pic --enable-shared && \
    make -j$(getconf _NPROCESSORS_ONLN) && \
    make install

cd ~/ffmpeg_sources && \
    curl -O -L https://ffmpeg.org/releases/ffmpeg-4.3.2.tar.bz2 && \
    tar -xf ffmpeg-4.3.2.tar.bz2 && \
    cd ffmpeg-4.3.2 && \
    PATH=~/bin:$PATH && \
    PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig" ./configure --prefix="$HOME/ffmpeg_build" --extra-cflags="-I$HOME/ffmpeg_build/include" --extra-ldflags="-L$HOME/ffmpeg_build/lib" --enable-openssl --enable-libvpx --enable-shared --enable-pic --bindir="$HOME/bin" && \
    make -j$(getconf _NPROCESSORS_ONLN) && \
    make install && \
    echo "/root/ffmpeg_build/lib/" >> /etc/ld.so.conf && \
    ldconfig && \
    rm -rf ~/ffmpeg_sources

export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/root/ffmpeg_build/lib/pkgconfig

cd ~

git clone https://github.com/opencv/opencv.git
cd opencv
git checkout "$cvVersion"
cd ..
 
git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
git checkout "$cvVersion"
cd ..

cd opencv
mkdir build
cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D INSTALL_C_EXAMPLES=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D WITH_TBB=OFF \
    -D WITH_V4L=OFF \
    -D OPENCV_SKIP_PYTHON_LOADER=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D WITH_QT=OFF \
    -D WITH_OPENGL=OFF \
    -D PYTHON_DEFAULT_EXECUTABLE=/usr/bin/python3 \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
    -D ENABLE_CXX11=ON \
    -D BUILD_EXAMPLES=OFF ..

make -j$(nproc)
make install

cd $cwd
