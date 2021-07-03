# Build Instructions for Linux

## Required Libraries
*   Python2
*   TensorFlow
*   Git
*   Bazel-3.7.2


## Installation Instructions

#### For Python2
In the console, run:
```bash
sudo apt install python
```

#### For TensorFlow
In the console, run:
```bash
sudo apt install python3-pip
pip install tensorflow
```

#### For Git
In the console, run:
```bash
sudo apt-get update
sudo apt-get install git
```

#### For Bazel
Download bazel binary installer from [Bazel release page on GitHub](https://github.com/bazelbuild/bazel/releases/tag/3.7.2). 
In the console, run:
```bash
chmod +x bazel-3.7.2-installer-linux-x86_64.sh
./bazel-3.7.2-installer-linux-x86_64.sh --user
export PATH="$PATH:$HOME/bin"
```
