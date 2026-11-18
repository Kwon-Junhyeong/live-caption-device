# Live Caption Device

A tabletop device for deaf and hard-of-hearing people that puts matching speaker numbers on faces and on live captions. No one wears a microphone, and everything runs offline on a Jetson Orin Nano inside the device. Built for the 24th Embedded Software Contest (2026).

![System](docs/system.png)

## From four microphones to captions labeled by speaker

Four ROS 2 nodes run on the device:

1. **Direction and separation.** One streaming network takes the audio of the four microphones in 128 ms chunks and returns a direction and a separated voice for each speaker.
2. **Camera.** Faces are detected, and each voice direction is matched to a face within 20 degrees.
3. **Speech recognition.** One faster-whisper session per voice, in Korean.
4. **Screen.** The camera image with numbered faces, and the captions with the same numbers.

## Training data for this housing

The microphones sit on the housing, and the housing scatters the sound before it reaches them. Every training voice is rendered through the housing's response computed with the boundary element method, so the network learns this housing.

## One network for direction and separation

Mamba-3 layers carry their state from one 128 ms chunk to the next, so the network runs on a live stream. It has 1.98 M parameters and improves SI-SDR by 11.67 dB.

## Timing and memory on the device

| | Median |
|---|---|
| Direction after the end of each 128 ms chunk | 0.22 s |
| Direction matched to a face | 0.29 s |
| Caption after the speaker stops | 2 s |

The network, the recognition model and the camera processing share the 8 GB Jetson, with 2.9 GB of memory to spare.

## Build and run

On the Jetson, with ROS 2 Humble, PyTorch with CUDA, `faster-whisper`, `sounddevice`, `pyrealsense2`, `mediapipe` and `mamba-ssm` installed, and the network checkpoint in `ros2_ws/src/doa_separation_node/weights/`:

```sh
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
ros2 launch ui_bridge_node captioning.launch.py
```
