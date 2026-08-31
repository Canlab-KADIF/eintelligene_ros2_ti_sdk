# ros_sdk
TI 보드에서 ROS 2 기반 로봇 및 비전 기능 개발을 위한 SDK 프로젝트입니다.

본 저장소는 ROS 2 Workspace 구조를 기반으로 구성되어 있으며,  
카메라 영상 처리, CNN 기반 객체 인식, 로봇 제어 노드 및 관련 인터페이스를 포함합니다.

## 1. Project Overview

Robotics SDK는 다음 기능을 제공하는 것을 목적으로 합니다.

- ROS 2 기반 로봇 애플리케이션 개발
- 카메라 이미지 송수신 및 영상 처리
- CNN 기반 객체 인식 및 추론
- 로봇 센서 데이터 처리
- ROS 2 Topic, Service 및 Action 인터페이스 제공
- 신규 알고리즘과 하드웨어 인터페이스 확장 지원

> 프로젝트의 실제 구현 범위에 맞게 위 항목을 수정해 주세요.

---

## 2. Repository Structure

```text
.
├── .vscode/
│   └── VS Code 개발 환경 설정
│
├── ros_ws/
│   └── src/
│       └── robotics_sdk/
│           ├── include/
│           │   └── 헤더 파일
│           ├── src/
│           │   └── 소스 파일
│           ├── launch/
│           │   └── ROS 2 Launch 파일
│           ├── config/
│           │   └── 설정 파일
│           ├── package.xml
│           └── CMakeLists.txt
│
├── .gitignore
└── README.md
```

현재 저장소 구조와 다른 폴더가 있다면 실제 폴더 구성에 맞게 수정해 주세요.

---

## 3. Development Environment

### Operating System

- Ubuntu 22.04 LTS

### ROS Version

- ROS 2 Humble Hawksbill

### Development Tools

- GCC / G++
- CMake
- Colcon
- Visual Studio Code
- Git

### Main Dependencies

- ROS 2
- OpenCV
- cv_bridge
- image_transport
- sensor_msgs
- rclcpp

CNN 추론 프레임워크를 사용하는 경우 아래 항목을 추가할 수 있습니다.

- ONNX Runtime
- TensorRT
- PyTorch
- OpenVINO

> 실제 개발환경이 Ubuntu 20.04 및 ROS 2 Foxy라면 해당 버전으로 변경해 주세요.

---

## 4. Prerequisites

ROS 2 환경이 설치되어 있어야 합니다.
ROS 2 환경을 활성화합니다.

```bash
source /opt/ros/humble/setup.bash
```

필요한 기본 패키지를 설치합니다.

```bash
sudo apt update
sudo apt install -y \
  python3-colcon-common-extensions \
  python3-rosdep \
  ros-humble-cv-bridge \
  ros-humble-image-transport \
  ros-humble-sensor-msgs
```

OpenCV가 필요한 경우 다음 명령어로 설치합니다.

```bash
sudo apt install -y libopencv-dev
```

---

## 5. Clone Repository

저장소를 복제합니다.

```bash
git clone <repository-url>
cd <repository-name>
```

예시:
```bash
git clone https://<git-server>/<group>/<repository>.git
cd <repository>
```

> `<repository-url>`과 `<repository-name>`을 실제 저장소 정보로 변경해 주세요.

---

## 6. Install Dependencies

ROS 2 Workspace의 의존성을 설치합니다.
```bash
cd ros_ws
rosdep update
rosdep install --from-paths src --ignore-src -r -y
```

---

## 7. Build
Workspace 디렉터리에서 다음 명령어를 실행합니다.

```bash
cd ros_ws
colcon build --symlink-install
```

특정 패키지만 빌드하려면 다음과 같이 실행합니다.

```bash
colcon build \
  --symlink-install \
  --packages-select robotics_sdk
```

빌드가 완료되면 Workspace 환경을 활성화합니다.

```bash
source install/setup.bash
```

### Clean Build

기존 빌드 결과를 삭제하고 다시 빌드하려면 다음 명령어를 사용합니다.

```bash
cd ros_ws
rm -rf build install log
colcon build --symlink-install
source install/setup.bash
```

---

## 8. Run

### Run ROS 2 Node

다음 명령어로 노드를 실행합니다.

```bash
ros2 run robotics_sdk <executable-name>
```

예시:

```bash
ros2 run robotics_sdk vision_cnn_node
```

### Run Launch File

Launch 파일이 제공되는 경우 다음과 같이 실행합니다.

```bash
ros2 launch robotics_sdk <launch-file-name>.launch.py
```

예시:

```bash
ros2 launch robotics_sdk vision_cnn.launch.py
```

> 실제 ROS 2 패키지명, 실행 파일명 및 Launch 파일명으로 변경해 주세요.

---

## 9. Vision CNN Node

`vision_cnn_node`는 카메라 또는 ROS 2 이미지 Topic을 입력받아 CNN 기반 추론을 수행하는 비전 노드입니다.

### Main Functions

- 카메라 영상 수신
- ROS 2 Image 메시지를 OpenCV 이미지로 변환
- 이미지 전처리
- CNN 모델 추론
- 추론 결과 후처리
- 인식 결과 Topic 발행
- 결과 이미지 시각화

### Input Topic

| Topic | Message Type | Description |
|---|---|---|
| `/camera/image_raw` | `sensor_msgs/msg/Image` | 카메라 원본 이미지 |

### Output Topic

| Topic | Message Type | Description |
|---|---|---|
| `/vision/detections` | 프로젝트 정의 메시지 | 객체 인식 결과 |
| `/vision/debug_image` | `sensor_msgs/msg/Image` | 인식 결과가 표시된 이미지 |

### Parameters

| Parameter | Type | Default | Description |
|---|---|---:|---|
| `model_path` | string | `""` | CNN 모델 파일 경로 |
| `input_topic` | string | `/camera/image_raw` | 입력 이미지 Topic |
| `confidence_threshold` | double | `0.5` | 인식 신뢰도 기준 |
| `device` | string | `cpu` | 추론 장치 |
| `publish_debug_image` | bool | `true` | 결과 이미지 발행 여부 |

> 위 Topic과 Parameter는 예시입니다. 실제 소스 코드에 정의된 이름으로 반드시 수정해 주세요.

---

## 10. Configuration

노드 설정은 다음 위치의 YAML 파일에서 관리할 수 있습니다.

```text
ros_ws/src/robotics_sdk/config/
```

설정 파일 예시:

```yaml
vision_cnn_node:
  ros__parameters:
    model_path: "/path/to/model.onnx"
    input_topic: "/camera/image_raw"
    confidence_threshold: 0.5
    device: "cpu"
    publish_debug_image: true
```

실행 전에 모델 파일 경로와 카메라 Topic을 실제 환경에 맞게 설정해 주세요.

---

## 11. Verify ROS 2 Communication

실행 중인 노드를 확인합니다.

```bash
ros2 node list
```

사용 가능한 Topic을 확인합니다.

```bash
ros2 topic list
```

입력 이미지 Topic 정보를 확인합니다.

```bash
ros2 topic info /camera/image_raw
```

이미지 데이터 발행 주기를 확인합니다.

```bash
ros2 topic hz /camera/image_raw
```

노드의 Parameter를 확인합니다.

```bash
ros2 param list /vision_cnn_node
```

---

## 12. Troubleshooting

### Package Not Found

다음과 같은 오류가 발생하는 경우:

```text
Package 'robotics_sdk' not found
```

Workspace 환경이 활성화되었는지 확인합니다.

```bash
cd ros_ws
source install/setup.bash
```

### ROS 2 Environment Not Loaded

ROS 2 명령어를 찾을 수 없는 경우 다음 명령어를 실행합니다.

```bash
source /opt/ros/humble/setup.bash
```

### Image Topic Not Received

카메라 Topic이 정상적으로 발행되는지 확인합니다.

```bash
ros2 topic list
ros2 topic hz /camera/image_raw
```

입력 Topic 이름이 다르면 설정 파일의 `input_topic` 값을 변경합니다.

### Model File Not Found

다음 항목을 확인합니다.

1. 모델 파일이 실제 경로에 존재하는지 확인합니다.
2. 설정 파일의 `model_path`가 올바른지 확인합니다.
3. 모델 파일을 읽을 수 있는 권한이 있는지 확인합니다.

```bash
ls -al /path/to/model.onnx
```

### Build Failure

의존성 설치 후 다시 빌드합니다.

```bash
cd ros_ws
rosdep install --from-paths src --ignore-src -r -y
rm -rf build install log
colcon build --symlink-install
```

---

## 13. Development Guide

### Branch Strategy

권장 Branch 구성은 다음과 같습니다.

| Branch | Purpose |
|---|---|
| `main` | 검증 및 배포 가능한 안정 버전 |
| `dev` | 기능 통합 및 개발 버전 |
| `feature/*` | 개별 기능 개발 |
| `fix/*` | 오류 수정 |
| `release/*` | 배포 준비 |

Feature Branch 예시:

```bash
git checkout dev
git pull origin dev
git checkout -b feature/add-vision-node
```

### Commit Message Convention

Commit 메시지는 변경 내용을 명확하게 작성합니다.

```text
feat: add CNN inference node
fix: correct image preprocessing
docs: update build instructions
refactor: reorganize vision module
test: add image callback test
chore: update development configuration
```

권장 형식:

```text
<type>: <summary>
```

주요 Type:

| Type | Description |
|---|---|
| `feat` | 신규 기능 |
| `fix` | 오류 수정 |
| `docs` | 문서 변경 |
| `refactor` | 기능 변화가 없는 코드 구조 개선 |
| `test` | 테스트 코드 추가 및 수정 |
| `chore` | 설정, 빌드 및 기타 작업 |

---

## 14. Merge Request Guide

Merge Request를 생성하기 전에 다음 항목을 확인합니다.

- [ ] 로컬 빌드가 정상적으로 완료되었는가?
- [ ] 신규 기능이 정상적으로 동작하는가?
- [ ] 기존 기능에 영향을 주지 않는가?
- [ ] 코드에 불필요한 로그가 남아 있지 않은가?
- [ ] 관련 설정 파일이 업데이트되었는가?
- [ ] README 또는 기술문서가 업데이트되었는가?
- [ ] 모델 및 대용량 파일이 Git에 포함되지 않았는가?
- [ ] Commit 메시지가 규칙에 맞게 작성되었는가?

Merge Request 내용에는 다음 정보를 포함합니다.

```markdown
## Summary

변경 목적과 주요 내용을 작성합니다.

## Changes

- 변경 항목 1
- 변경 항목 2
- 변경 항목 3

## Test Result

실행한 시험과 결과를 작성합니다.

## Related Issue

관련 Issue 번호 또는 링크를 작성합니다.

## Additional Notes

리뷰 시 확인이 필요한 사항을 작성합니다.
```

---

## 15. Coding Convention

- C++ 표준은 프로젝트 설정에 따릅니다.
- Class 이름은 `PascalCase`를 사용합니다.
- Function 및 Variable 이름은 `snake_case`를 사용합니다.
- 상수는 `UPPER_SNAKE_CASE`를 사용합니다.
- ROS 2 Topic과 Parameter 이름은 소문자와 underscore를 사용합니다.
- Header와 Source 파일의 역할을 분리합니다.
- Public API에는 필요한 설명을 작성합니다.
- 디버그 메시지는 ROS 2 Logging API를 사용합니다.

예시:

```cpp
RCLCPP_INFO(this->get_logger(), "Vision CNN node started");
RCLCPP_WARN(this->get_logger(), "Image topic is not available");
RCLCPP_ERROR(this->get_logger(), "Failed to load model: %s", model_path.c_str());
```

---

## 16. Known Issues

현재 확인된 제한사항 또는 미해결 문제를 작성합니다.

- 특정 카메라 해상도에서 성능 확인 필요
- GPU 추론 설정 검증 필요
- 모델 경로 설정의 환경별 관리 필요
- 실시간 추론 성능 최적화 필요

문제가 해결되면 해당 항목을 삭제하거나 Issue 번호를 추가해 주세요.

---

## 17. Roadmap

- [ ] ROS 2 이미지 입력 기능
- [ ] CNN 모델 로딩 및 추론 기능
- [ ] 객체 인식 결과 Topic 발행
- [ ] Launch 및 YAML 설정 지원
- [ ] CPU 및 GPU 추론 지원
- [ ] 단위 테스트 추가
- [ ] 통합 테스트 추가
- [ ] 성능 측정 결과 문서화
- [ ] Docker 개발환경 지원

---

## 18. Maintainer

| Role | Name | Contact |
|---|---|---|
| Project Owner | `<name>` | `<email>` |
| Maintainer | `<name>` | `<email>` |
| Developer | `<name>` | `<email>` |

프로젝트 관련 문의는 Issue 또는 담당자 이메일을 이용해 주세요.

---

## 19. License

이 프로젝트의 소유권 및 사용 권한은 조직의 내부 소프트웨어 관리 정책을 따릅니다.

외부 공개 프로젝트인 경우 실제 라이선스에 맞게 아래와 같이 변경합니다.

```text
Apache License 2.0
```

또는

```text
MIT License
```

---

## 20. Change History

| Version | Date | Description |
|---|---|---|
| `0.1.0` | YYYY-MM-DD | Initial version |
| `0.2.0` | YYYY-MM-DD | Added vision CNN node |
| `1.0.0` | YYYY-MM-DD | First stable release |

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
본 연구는 과학기술정보통신부 및 정보통신기획평가원의 자율주행기술개발혁신사업의 지원을 받아 수행된 연구임(RS-2023-00232046, 비정상 주행 데이터 전송을 통한 클라우드 기반 원인 분석 기술 개발).
 
This work was partly supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.RS-2023-00232046, Development of cloud-based cause analysis technology by transmission of abnormal driving data)
