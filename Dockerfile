##############################################################################
# VLA Navigation Benchmark — Milestone 2 Container
#
# Builds a complete simulation + VLA inference environment:
#   ROS 2 Humble  ·  Gazebo Harmonic  ·  PX4 SITL  ·  Micro-XRCE-DDS Agent
#   PyTorch  ·  HuggingFace Transformers  ·  OpenVLA
#
# Build:   docker build -t vla-sim:m2 .
# Cluster: see scripts/build_container.sh for Pyxis/Slurm workflow
#
# OpenVLA weights (~14 GB) are NOT baked into the image.
# Mount them at runtime:
#   --container-mounts="$HOME/models/openvla-7b:/models/openvla-7b"
# Then set:  -p model_id:="/models/openvla-7b"
##############################################################################

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

SHELL ["/bin/bash", "-c"]

# ── 1. Base system tools ────────────────────────────────────────────────────

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl wget git cmake build-essential pkg-config \
        python3 python3-pip python3-dev python3-venv \
        lsb-release gnupg2 software-properties-common \
        xvfb mesa-utils libgl1-mesa-glx libglu1-mesa \
        locales ca-certificates sudo \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# ── 2. ROS 2 Humble ────────────────────────────────────────────────────────

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
        http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
        > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-ros-base \
        ros-humble-ros-gz \
        python3-colcon-common-extensions \
        python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

# ── 3. Gazebo Harmonic ─────────────────────────────────────────────────────

RUN curl -sSL https://packages.osrfoundation.org/gazebo.gpg \
        -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
        http://packages.osrfoundation.org/gazebo/ubuntu-stable $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
        > /etc/apt/sources.list.d/gazebo-stable.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        gz-harmonic \
        libgz-transport13-dev \
        libgz-sim8-dev \
        libgz-sensors8-dev \
        libgz-plugin2-dev \
        libgz-msgs10-dev \
        libgz-math7-dev \
        libgz-cmake3-dev \
    && rm -rf /var/lib/apt/lists/*

# ── 4. Micro-XRCE-DDS Agent ───────────────────────────────────────────────

RUN git clone --depth 1 https://github.com/eProsima/Micro-XRCE-DDS-Agent.git /tmp/agent \
    && cd /tmp/agent && mkdir build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
    && make -j"$(nproc)" && make install \
    && ldconfig \
    && rm -rf /tmp/agent

# ── 5. PX4 Autopilot ──────────────────────────────────────────────────────

ENV GZ_DISTRO=harmonic

RUN git clone --recursive --depth 1 -b v1.15.0 \
        https://github.com/PX4/PX4-Autopilot.git /opt/PX4-Autopilot \
    && cd /opt/PX4-Autopilot \
    && bash Tools/setup/ubuntu.sh --no-nuttx \
    && DONT_RUN=1 make px4_sitl gz_x500_mono_cam \
    && rm -rf /opt/PX4-Autopilot/build/px4_sitl_default/tmp

ENV PX4_DIR=/opt/PX4-Autopilot

# ── 6. ROS 2 workspace — px4_msgs + vla_navigation ────────────────────────

RUN mkdir -p /opt/ros2_ws/src

RUN cd /opt/ros2_ws/src \
    && git clone --depth 1 -b release/1.15 https://github.com/PX4/px4_msgs.git

COPY src/vla_navigation /opt/ros2_ws/src/vla_navigation

RUN source /opt/ros/humble/setup.bash \
    && cd /opt/ros2_ws \
    && colcon build --symlink-install \
    && echo "source /opt/ros/humble/setup.bash"       >> /root/.bashrc \
    && echo "source /opt/ros2_ws/install/setup.bash"   >> /root/.bashrc

# ── 7. cv_bridge (ROS Image ↔ OpenCV/numpy) ───────────────────────────────

RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-cv-bridge \
        python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# ── 8. Python deps (PyTorch, Transformers, OpenVLA) ───────────────────────

RUN pip3 install --no-cache-dir pyyaml numpy Pillow \
    && pip3 install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --no-cache-dir \
        transformers accelerate sentencepiece

# ── 9. Entry ───────────────────────────────────────────────────────────────

WORKDIR /opt/ros2_ws
CMD ["bash"]
