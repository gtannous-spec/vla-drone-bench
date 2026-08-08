#!/usr/bin/env bash
###############################################################################
# setup_inside_container.sh — Runs INSIDE the Pyxis container during build.
# Installs ROS 2 Humble, Gazebo Harmonic, PX4 SITL, Micro-XRCE-DDS Agent,
# px4_msgs, and builds the vla_navigation package.
#
# Called by build_container.sh — do not run directly on the login node.
###############################################################################

set -euo pipefail

PROJ_DIR="${1:?Usage: setup_inside_container.sh <project_dir>}"

export DEBIAN_FRONTEND=noninteractive
export TZ=Etc/UTC

echo "===== [1/7] Base system packages ====="
apt-get update
apt-get install -y --no-install-recommends \
    curl wget git cmake build-essential pkg-config \
    python3 python3-pip python3-dev python3-venv \
    lsb-release gnupg2 software-properties-common \
    xvfb mesa-utils libgl1-mesa-glx libglu1-mesa \
    locales ca-certificates sudo
locale-gen en_US.UTF-8

echo "===== [2/7] ROS 2 Humble ====="
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu jammy main" \
    > /etc/apt/sources.list.d/ros2.list
apt-get update
apt-get install -y --no-install-recommends \
    ros-humble-ros-base \
    ros-humble-ros-gz \
    python3-colcon-common-extensions

echo "===== [3/7] Gazebo Harmonic (runtime + dev) ====="
curl -sSL https://packages.osrfoundation.org/gazebo.gpg \
    -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
    http://packages.osrfoundation.org/gazebo/ubuntu-stable jammy main" \
    > /etc/apt/sources.list.d/gazebo-stable.list
apt-get update
apt-get install -y gz-harmonic \
    libgz-transport13-dev \
    libgz-sim8-dev \
    libgz-sensors8-dev \
    libgz-plugin2-dev \
    libgz-msgs10-dev \
    libgz-math7-dev \
    libgz-cmake3-dev

echo "===== [4/7] Micro-XRCE-DDS Agent ====="
cd /
git clone --depth 1 https://github.com/eProsima/Micro-XRCE-DDS-Agent.git /tmp/agent
cd /tmp/agent && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)"
make install
ldconfig
cd /
rm -rf /tmp/agent

echo "===== [5/7] PX4 Autopilot (v1.15) ====="
cd /
git clone --recursive --depth 1 -b v1.15.0 \
    https://github.com/PX4/PX4-Autopilot.git /opt/PX4-Autopilot
cd /opt/PX4-Autopilot
bash Tools/setup/ubuntu.sh --no-nuttx
export GZ_DISTRO=harmonic
make px4_sitl_default

echo "===== [6/7] px4_msgs ====="
mkdir -p /opt/ros2_ws/src
cd /opt/ros2_ws/src
git clone --depth 1 -b release/1.15 https://github.com/PX4/px4_msgs.git

echo "===== [7/7] Build ROS 2 workspace ====="
cp -r "$PROJ_DIR/src/vla_navigation" /opt/ros2_ws/src/vla_navigation

# ROS 2 setup.bash uses variables that may be unset — relax strict mode
set +u
source /opt/ros/humble/setup.bash
set -u

cd /opt/ros2_ws
colcon build --symlink-install

pip3 install --no-cache-dir pyyaml numpy

# Persist environment setup (bashrc doesn't use strict mode, so this is fine)
cat >> /root/.bashrc << 'RCEOF'
source /opt/ros/humble/setup.bash
source /opt/ros2_ws/install/setup.bash
export PX4_DIR=/opt/PX4-Autopilot
export GZ_DISTRO=harmonic
RCEOF

echo ""
echo "===== Container setup complete! ====="
set +u
source /opt/ros/humble/setup.bash
set -u
echo "  ROS 2:    $(ros2 --version 2>/dev/null || echo 'installed')"
echo "  Gazebo:   $(gz sim --version 2>/dev/null | head -1 || echo 'installed')"
echo "  PX4:      /opt/PX4-Autopilot (v1.15.0)"
echo "  Workspace: /opt/ros2_ws"
