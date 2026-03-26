# Panorama PTZ Viewer

A small Python GUI application for viewing a live **360° equirectangular video feed** and an interactive **PTZ view** derived from it in real time.
The GUI is built with **[Dear ImGui Bundle](https://github.com/pthom/imgui_bundle)**.

## Features

- Live **360 panorama** window
- Live interactive **PTZ** window
- Small **panorama thumbnail** inside the PTZ window
- **ROS 2 raw image** input from a `sensor_msgs/msg/Image` topic
- Optional **video file** input
- Optional **live THETA camera input through GStreamer** using `thetauvcsrc`

## Supported input modes

The application can work with:

- a **ROS 2 raw** `sensor_msgs/msg/Image` topic
- a **360 video file**
- a **RICOH THETA live stream** captured through **GStreamer** with `thetauvcsrc`

### ROS 2 mode

This mode is intended for workflows where a camera node publishes compressed
`ffmpeg_image_transport_msgs/msg/FFMPEGPacket` messages and the stream is decoded and republished as a raw `sensor_msgs/msg/Image` topic through `image_transport`.

Example decoded topic:

```bash
/camera/image_decoded
```

## Requirements

- Ubuntu
- Python 3.10+
- ROS 2 Jazzy (or compatible ROS 2 installation)
- OpenGL-capable system
- GStreamer with `gstthetauvc` installed if using THETA live camera mode

Python packages are listed in:

```bash
requirements.txt
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you use ROS 2:

```bash
source /opt/ros/jazzy/setup.bash
```

## Run

### 1) ROS 2 decoded image topic

```bash
python3 viewer.py --ros-topic /camera/image_decoded
```

### 2) 360 video file

```bash
python3 viewer.py my_video.mp4 --loop-video
```

### 3) Live THETA camera through GStreamer

```bash
python3 viewer.py --camera-mode 4K
```

Or specify the THETA serial explicitly:

```bash
python3 viewer.py --camera-mode 2K --theta-serial YOUR_SERIAL
```

You can also provide a custom GStreamer pipeline:

```bash
python3 viewer.py --gst-pipeline "thetauvcsrc mode=4K ! queue ! h264parse ! decodebin ! queue ! videoconvert ! video/x-raw,format=BGR ! appsink name=theta_appsink emit-signals=false sync=false max-buffers=1 drop=true"
```

## Example ROS 2 republish workflow

If your camera publishes `ffmpeg` transport packets, first republish them as raw images:

```bash
ros2 run image_transport republish --ros-args \
  -p in_transport:=ffmpeg \
  -p out_transport:=raw \
  --remap in/ffmpeg:=/camera/image_h264/ffmpeg \
  --remap out:=/camera/image_decoded
```

Then launch the viewer:

```bash
python3 viewer.py --ros-topic /camera/image_decoded
```

## Controls

- **Panorama window**
  - click / drag to move the PTZ target

- **PTZ window**
  - drag to pan / tilt
  - mouse wheel to zoom

- **Keyboard**
  - `W A S D` or arrow keys: pan / tilt
  - `+` / `-`: zoom
  - `R`: reset PTZ
  - `Q` or `Esc`: quit