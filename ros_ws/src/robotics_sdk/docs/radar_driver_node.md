mmWave Radar ROS Node
========================

This is a radar ROS node for TI's mmWave devices. The ROS node sets up a ROS service as an interface to configure the mmWave Evaluation Module and publishes a ROS PointCloud2 message with the objects detected when the sensor is activated.

```{note}
This ROS node, running on SK-TDA4VM, currently is tested only with TI's [IWR6843ISK](https://www.ti.com/tool/IWR6843ISK).
```

## Setup

### Re-flash the the TI mmWave EVM with Out-of-Box firmware

If you are using a brand new TI mmWave EVM, it may come pre-flashed with an older version of the Out-of-Box Demo. The EVM should be re-flashed with the Out-of-Box demo from the compatible version of the mmWave SDK.

1. On a PC, download [Uniflash](http://processors.wiki.ti.com/index.php/Category:CCS_UniFlash), TI’s flashing software

2. Install libraries required for Uniflash to work:
    ```
    sudo apt install libusb-0.1-4 libgconf-2-4
    ```

3. Make sure the file `/lib/x86_64-linux-gnu/libudev.so.0` exists. If it does not exist execute the following command to generate a symbolic link necessary for Uniflash to work:
    ```
    sudo ln -sf /lib/x86_64-linux-gnu/libudev.so.1 /lib/x86_64-linux-gnu/libudev.so.0
    ```

4. Navigate to the download directory, make the file executable and run the Uniflash installer:
    ```
    cd ~/Downloads
    ```
    ```
    chmod +x uniflash_sl.7.1.0.3796.run
    ```
    ```
    ./uniflash_sl.7.1.0.3796.run
    ```

5. Download the linux version of the [mmWave SDK](https://www.ti.com/tool/mmwave-sdk). MmWave SDK version:
    - Compatible with TI mmWave SDK v3.5 out-of-box demo when using IWR6843AOPEVM
    - Compatible with TI mmWave SDK v3.5 out-of-box demo when using IWR6843 ES2.0 EVM

6. Install the following mmWave SDK dependency:
    ```
    sudo apt-get install libc6:i386
    ```

7. Navigate to the download directory, make the file executable and run the mmWave SDK installer (change `<ver>` to the appropriate version):
    ```
    cd ~/Downloads
    ```
    ```
    chmod +x mmwave_sdk_<ver>-Linux-x86-Install.bin
    ```
    ```
    ./mmwave_sdk_<ver>-Linux-x86-Install.bin
    ```

8. Follow the [EVM Operational Modes Setup Guide](https://dev.ti.com/tirex/explore/node?node=AMLjV2I4hAEEwpYRfP2okA__VLyFKFf__LATEST) to configure the device for `flashing` mode. Once configured for flashing mode, power cycle the device by toggling the NRST switch or unplugging then reconnecting the micro usb cable.

9. Follow the [Using Uniflash with mmWave Guide](https://dev.ti.com/tirex/explore/node?node=AGuT13c2cuwxqZKfpyurBQ__VLyFKFf__LATEST) to flash the device with the apropriate binary.

    Ensure the device is connected to the PC, then serial port names can be found with:
    ```
    ls /dev | grep 'ttyUSB'  # for standalone EVMs

    ls /dev | grep 'ttyACM'  # for EVM + ICBOOST carrier board
    ```

    Two serial ports should appear e.g.: `ttyUSB0` and `ttyUSB1`, in this example, `/dev/ttyUSB0` should be entered as the COM port in the `Settings & Utilities` tab in Uniflash.

    The .bin file can be found at `<MMWAVE_SDK_INSTALL_DIR>/packages/ti/demo/xwr68xx/mmw/xwr68xx_mmw_demo.bin`

10. Follow the [EVM Operational Modes Setup Guide](https://dev.ti.com/tirex/explore/node?node=AMLjV2I4hAEEwpYRfP2okA__VLyFKFf__LATEST) to configure the device for `functional` mode. Once configured for functional mode, power cycle the device by toggling the NRST switch or unplugging then reconnecting the micro usb cable.

### Select Chirp Configuration

This ROS node will, by default, use a chirp configuration that optimizes range resolution. The configuration file is located at `/cfg/6843_3d.cfg`.
If you wish to use the default chirp configuration, no further action is required.

If you wish to use a custom chirp configuration, the [mmWave Sensing Estimator](https://dev.ti.com/gallery/view/mmwave/mmWaveSensingEstimator/ver/1.4.0/) can be used to estimate the chirp configuration based on scene parameters. For more information on how to select the right chirp parameters in a FMCW Radar device, please see [Programming Chirp Parameters in TI Radar Devices](https://www.ti.com/lit/an/swra553a/swra553a.pdf). To modify the parameters, edit `6843_3d.cfg` and save the file using a text editor or specify an alternate configuration file in the mmWaveQuickConfig node section of `6843_3d_TDA4VM.launch`.

```{note}
This ROS node currently is tested only with the chirp configuration provided. Modifying the chirp configuration may have an adverse effect on the performance.
```

### Setup Symbolic Links

This section is necessary for those using multiple sensors. Users with a single sensor can skip this step.

When the standalone sensors are connected to the SK board, the device appears as `/dev/ttyUSBx` (Command Port) and `/dev/ttyUSBy` (Data Port). These names are then used in the launch files to communicate with the sensor. When using multiple sensors and hardcoding these ttyUSB names, the devices must then be connected to the SK board in a specific order. Furthermore, other devices may also appear as `/dev/ttyUSBx`. Therefore, creating a symbolic link will be extremely useful. Below are the steps to setting up symbolic links.

```{note}
The following steps are for standalone EVMs and will not work with the ICBOOST carrier board.
```

1. On the target host linux, create a file named `99-usb-serial.rules` in the directory `/etc/udev/rules.d/` with super user:
    ```
    root@am6x-sk:~$ sudo vi /etc/udev/rules.d/99-usb-serial.rules
    ```

2. Press 'i', then type or paste the following:
    ```
    SUBSYSTEM=="tty", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea70", SYMLINK+="mmWave_%s{serial}_%E{ID_USB_INTERFACE_NUM}"
    ```

3. Close and save the file by hitting 'Esc', typing ':', then 'wq', and finally pressing 'Enter'.

4. Plug in the first sensor to the SK board and type the following in the command line:
    ```
    root@am6x-sk:~$ ls -1 /dev/ | grep mmWave
    ```

5. Two symbolic links should appear. Example: "mmWave_00CE0FCA_00", typically the command port, and "mmWave_00E0FCA_01", typically the data port.

6. Repeat steps 4-5 for any additional sensors that are being used. Be sure to note the names of each serial port as you go.

7. In the file, `/launch/6843_quad_3d_TDA4VM.launch`, update the `com_user` and `com_data` parameters for each sensor to match the new serial port names.

### Setup Static Transform

This section is necessary for those using multiple sensors. Users with a single sensor can skip this step.

The launch files for the radar driver also create a static transform. If the position or angle of the sensors on the robot is different, the arguments in the static transform must be changed. Find more information on static transforms here: [http://wiki.ros.org/tf#static_transform_publisher](http://wiki.ros.org/tf#static_transform_publisher)

For example, one can use the center of the robot as a baseline. In the file, `/launch/6843_quad_3d_TDA4VM.launch`, update `tf_args` (X Y Z Yaw Pitch Roll) for each sensor to describe the position of the sensor relative to the center of the robot.

<!-- ### Synchronization

For details regarding time synchronization of frames from multiple mmWave devices click [here](multi_sensor_time_synchronization.md). -->

## Usage

1. Build the mmWave radar ROS node
    ```
    cd $ROS_WS
    # ROS 1
    catkin_make --source /opt/robotics_sdk/ros1 --force-cmake
    source devel/setup.bash
    ```

2. Launch the mmWave radar ROS node
    ```
    # ROS 1
    roslaunch ti_mmwave_rospkg 6843_3d_TDA4VM.launch # for single sensor
    roslaunch ti_mmwave_rospkg 6843_quad_3d_TDA4VM.launch # for multiple sensors
    ```

## Visualization on Remote Ubuntu PC

In the PC Docker container,
```
roslaunch ti_mmwave_rospkg viz_remote_pc.launch # for single sensor

roslaunch ti_mmwave_rospkg viz_remote_pc_quad.launch # for multiple sensor
```

## Launch File Parameters

 Parameter                       | Description                                                                           | Value
---------------------------------|---------------------------------------------------------------------------------------|-----------
 max_allowed_elevation_angle_deg | Maximum allowed elevation angle in degrees for detected object data [0 > value >= 90] | int
 max_allowed_azimuth_angle_deg   | Maximum allowed azimuth angle in degrees for detected object data [0 > value >= 90]   | int
 device_num                      | The device number for this sensor                                                     | int
 com_user                        | Serial port used for configuring mmWave radar device                                  | string
 com_data                        | Serial port used for receiving data from mmWave radar device                          | string
 tf_args                         | Values to be used for tf transformation                                               | string

