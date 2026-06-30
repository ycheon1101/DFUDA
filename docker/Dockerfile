# Use an official NVIDIA CUDA base image
FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04


# Set the maintainer label
LABEL maintainer="##TODO: add your email here##"

# Set environment variables to non-interactive (this prevents some prompts)
ENV DEBIAN_FRONTEND=non-interactive


# Run package updates and install packages
RUN apt-get update \
   && apt-get install -y --no-install-recommends \
   python3 \
   python3-pip \
   python3-dev \
   build-essential \
   cmake \
   wget \
   unzip \
   tmux \
   git \
   libglib2.0-0 \
   libgl1-mesa-glx \
   curl \
   && rm -rf /var/lib/apt/lists/*


# Ubuntu 20.04 provides Python 3.8.10. Fail early if the base image changes.
RUN python3 -c "import sys; assert sys.version_info[:3] == (3, 8, 10), sys.version"


# Install Python packages pinned by the repository.
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt \
   -f https://download.pytorch.org/whl/torch_stable.html




# Install MMCV separately because it is not listed in requirements.txt.
RUN pip3 install mmcv-full==1.3.7 \
   -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.7.1/index.html



# Set the working directory
WORKDIR /home/DFUDA/


# Copy the current directory contents into the container at /app
#COPY . /app


# Run your script when the container launches (uncomment the next line and add your script)
# CMD ["python3", "your_script.py"]



