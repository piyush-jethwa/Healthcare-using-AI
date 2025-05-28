
## 🔍 Problem Statement

Many individuals in semi-urban and rural areas face barriers in accessing timely medical consultations due to limited infrastructure, lack of healthcare professionals, and language challenges. There is a need for a virtual system that can understand patients' input through various mediums and provide basic diagnostics and support in multiple languages.

## 💡 Approach & Solution

**Advance HealthCare** is a virtual health assistant that utilizes AI/ML to offer preliminary diagnosis through natural input methods: **voice**, **text**, and **image**. It consists of four modular phases including diagnosis logic, speech recognition and synthesis, and a web UI interface using **Gradio**.

The system allows users to interact in their preferred language (Hindi, English, Marathi), making it highly accessible and user-friendly.

## ✨ Features

- 🎙 Voice input for patient symptoms
- 📄 Text input for faster interaction
- 🖼 Image input (for basic visual symptom detection)
- 🧠 AI-powered diagnosis engine
- 🗣 Multilingual output via Text-to-Speech (gTTS)
- 💬 Gradio UI for web-based interaction
- 🧾 Modular architecture (4 distinct components)

## 🛠 Tech Stack

- **Programming Language:** Python
- **Libraries/Frameworks:** Gradio, gTTS, SpeechRecognition, OpenCV, PIL
- **Tools:** FFmpeg, PortAudio
- **OS Compatibility:** Windows, macOS, Linux
- **Python Version:** 3.11+


# Project Setup Guide

This guide provides step-by-step instructions to set up your project environment, including the installation of FFmpeg and PortAudio across macOS, Linux, and Windows, as well as setting up a Python virtual environment using Pipenv, pip, or conda.

## Table of Contents

1. [Installing FFmpeg and PortAudio](#installing-ffmpeg-and-portaudio)
   - [macOS](#macos)
   - [Linux](#linux)
   - [Windows](#windows)
2. [Setting Up a Python Virtual Environment](#setting-up-a-python-virtual-environment)
   - [Using Pipenv](#using-pipenv)
   - [Using pip and venv](#using-pip-and-venv)
   - [Using Conda](#using-conda)
3. [Running the application](#project-phases-and-python-commands)

## Installing FFmpeg and PortAudio

### macOS

1. **Install Homebrew** (if not already installed):

   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install FFmpeg and PortAudio:**

   ```bash
   brew install ffmpeg portaudio
   ```


### Linux
For Debian-based distributions (e.g., Ubuntu):

1. **Update the package list**

```
sudo apt update
```

2. **Install FFmpeg and PortAudio:**
```
sudo apt install ffmpeg portaudio19-dev
```

### Windows

#### Download FFmpeg:
1. Visit the official FFmpeg download page: [FFmpeg Downloads](https://ffmpeg.org/download.html)
2. Navigate to the Windows builds section and download the latest static build.

#### Extract and Set Up FFmpeg:
1. Extract the downloaded ZIP file to a folder (e.g., `C:\ffmpeg`).
2. Add the `bin` directory to your system's PATH:
   - Search for "Environment Variables" in the Start menu.
   - Click on "Edit the system environment variables."
   - In the System Properties window, click on "Environment Variables."
   - Under "System variables," select the "Path" variable and click "Edit."
   - Click "New" and add the path to the `bin` directory (e.g., `C:\ffmpeg\bin`).
   - Click "OK" to apply the changes.

#### Install PortAudio:
1. Download the PortAudio binaries from the official website: [PortAudio Downloads](http://www.portaudio.com/download.html)
2. Follow the installation instructions provided on the website.

---

## Setting Up a Python Virtual Environment

### Using Pipenv
1. **Install Pipenv (if not already installed):**  
```
pip install pipenv
```

2. **Install Dependencies with Pipenv:** 

```
pipenv install
```

3. **Activate the Virtual Environment:** 

```
pipenv shell
```

---

### Using `pip` and `venv`
#### Create a Virtual Environment:
```
python -m venv venv
```

#### Activate the Virtual Environment:
**macOS/Linux:**
```
source venv/bin/activate
```

**Windows:**
```
venv\Scripts\activate
```

#### Install Dependencies:
```
pip install -r requirements.txt
```

---

### Using Conda
#### Create a Conda Environment:
```
conda create --name myenv python=3.11
```

#### Activate the Conda Environment:
```
conda activate myenv
```

#### Install Dependencies:
```
pip install -r requirements.txt
```


# Project Phases and Python Commands

## Phase 1: Brain of the doctor
```
python brain_of_the_doctor.py
```

## Phase 2: Voice of the patient
```
python voice_of_the_patient.py
```

## Phase 3: Voice of the doctor
```
python voice_of_the_doctor.py
```

## Phase 4: Setup Gradio UI
```
ai_doctor_fully_fixed.py
```
## 🖼 Screenshots

![Screenshot 2025-05-29 001049](https://github.com/user-attachments/assets/e15b152f-5609-40bb-a259-87a224dfbc41)
![Screenshot 2025-05-28 233232](https://github.com/user-attachments/assets/3348a6cd-917b-43a5-8ab8-60b3135f7811)
![Screenshot 2025-05-28 233311](https://github.com/user-attachments/assets/e1cb8310-5c9c-40e4-b9ad-07b14dc8e74e)
![Screenshot 2025-05-28 233400](https://github.com/user-attachments/assets/40af1cde-4e0f-4dfb-bed3-019aa96fadfa)





