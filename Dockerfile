FROM ubuntu:20.04

RUN apt-get update
RUN apt-get install -y ninja-build cmake libc++-9-dev libz-dev libpng-dev libjpeg-dev libxrandr-dev libxinerama-dev libxcursor-dev
COPY ./builder/build/eradiate-kernel-dist.tar /build/mitsuba/

RUN mkdir -p /build/mistuba /mitsuba && cd /build/mitsuba && tar -xf eradiate-kernel-dist.tar  \
    && mv dist /mitsuba/dist \
    && cd / && rm -rf /build

ENV MITSUBA_DIR=/mitsuba
WORKDIR /app

ENV PYTHONPATH="$MITSUBA_DIR/dist/python:$MITSUBA_DIR/$BUILD_DIR/dist/python:$PYTHONPATH"
ENV PATH="$MITSUBA_DIR/dist:$MITSUBA_DIR/$BUILD_DIR/dist:$PATH"

CMD mitsuba --help