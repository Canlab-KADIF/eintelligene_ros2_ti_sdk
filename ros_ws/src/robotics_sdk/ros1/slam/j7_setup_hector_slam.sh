#!/bin/bash
cd $SLAM_ROOT
rm -rf hector_slam
git clone https://github.com/tu-darmstadt-ros-pkg/hector_slam.git
cd  hector_slam
git checkout tags/0.5.2 -b j7_hector_slam
git apply $SLAM_ROOT/patch/j7_docker_hector_slam.patch
git add --all
git commit -m 'Enabled Hector SLAM on Jacinto7'
