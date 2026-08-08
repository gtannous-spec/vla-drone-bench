"""
urban_drone.launch.py — Boots the full Milestone 1 simulation stack:

  1. Gazebo Harmonic with the urban_city world
  2. PX4 SITL (gz_x500_mono_cam airframe)
  3. Micro-XRCE-DDS Agent (PX4 ↔ ROS 2 bridge)
  4. classical_planner waypoint controller
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    TimerAction,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    pkg_share = get_package_share_directory('vla_navigation')
    world_file = os.path.join(pkg_share, 'worlds', 'urban_city.sdf')
    config_file = os.path.join(pkg_share, 'config', 'benchmark_config.yaml')

    # ── Launch arguments ─────────────────────────────────────────────────────

    task_id_arg = DeclareLaunchArgument(
        'task_id', default_value='1',
        description='ID of the navigation task to execute (1-10)',
    )

    px4_dir_arg = DeclareLaunchArgument(
        'px4_dir', default_value=os.path.expanduser('~/PX4-Autopilot'),
        description='Absolute path to PX4-Autopilot source tree',
    )

    headless_arg = DeclareLaunchArgument(
        'headless', default_value='false',
        description='Run Gazebo without the GUI',
    )

    # ── 1. Gazebo Harmonic ───────────────────────────────────────────────────

    gz_server = ExecuteProcess(
        cmd=[
            'gz', 'sim', '-r', '-s',  # run, server-only (GUI separate)
            world_file,
        ],
        output='screen',
    )

    gz_gui = ExecuteProcess(
        cmd=['gz', 'sim', '-g'],      # GUI client
        output='screen',
        # Skipped when headless — handled by condition below (kept simple here)
    )

    # ── 2. PX4 SITL ─────────────────────────────────────────────────────────
    # The gz_x500_mono_cam airframe gives us an x500 quad with an RGB camera.

    px4_sitl = ExecuteProcess(
        cmd=[
            'bash', '-c',
            'cd $PX4_DIR && make px4_sitl gz_x500_mono_cam',
        ],
        additional_env={
            'PX4_DIR': LaunchConfiguration('px4_dir'),
            'PX4_GZ_WORLD': '',       # empty so PX4 doesn't load its own world
            'PX4_GZ_MODEL_POSE': '0,0,0,0,0,0',
        },
        output='screen',
    )

    # ── 3. Micro-XRCE-DDS Agent ─────────────────────────────────────────────

    dds_agent = ExecuteProcess(
        cmd=[
            'MicroXRCEAgent', 'udp4', '-p', '8888',
        ],
        output='screen',
    )

    # ── 4. Classical planner node (delayed to let PX4 boot) ──────────────────

    planner_node = TimerAction(
        period=15.0,   # wait 15 s for PX4+Gazebo to initialise
        actions=[
            Node(
                package='vla_navigation',
                executable='classical_planner',
                name='classical_planner',
                output='screen',
                parameters=[{
                    'task_id': LaunchConfiguration('task_id'),
                    'config_file': config_file,
                }],
            ),
        ],
    )

    # ── Assemble ─────────────────────────────────────────────────────────────

    return LaunchDescription([
        task_id_arg,
        px4_dir_arg,
        headless_arg,
        gz_server,
        gz_gui,
        px4_sitl,
        dds_agent,
        planner_node,
    ])
