FROM ubuntu:20.04

RUN apt update

RUN DEBIAN_FRONTEND=noninteractive apt install -y   \ 
    git                                             \
    cmake                                           \
    ninja-build                                     \
    clang-9                                         \
    libc++-9-dev                                    \
    libc++abi-9-dev                                 \
    libpng-dev                                      \
    zlib1g-dev                                      \
    python3.9-distutils                             \
    python3-pip                                     \
    libjpeg-dev

ENV CC=clang-9
ENV CXX=clang++-9

CMD git clone --recursive https://github.com/eradiate/eradiate-kernel /sources/eradiate-kernel                         \
    && mkdir -p /build/eradiate-kernel                                                                                 \
    && cd /build/eradiate-kernel                                                                                       \
    && cmake -GNinja -DPYTHON_EXECUTABLE=$(python3 -c "import sys; print(sys.executable)") /sources/eradiate-kernel    \
    && ninja                                                                                                           \
    && tar -cvf eradiate-kernel-dist.tar ./dist
