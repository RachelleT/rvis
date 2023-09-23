FROM python:3.11.5-bullseye

#libraries for graphics
RUN apt-get update 
RUN apt install -y libgl1-mesa-glx
RUN apt install -y libx11-6 libxext-dev libxrender-dev libxinerama-dev libxi-dev libxrandr-dev libxcursor-dev libxtst-dev
RUN apt install -y libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev 
RUN apt install -y libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0
RUN apt install -y libxcb-randr0-dev libxcb-xtest0-dev libxcb-xinerama0-dev libxcb-shape0-dev libxcb-xkb-dev
RUN apt install -y libxkbcommon-x11-0
RUN apt install -y libdbus-1-3

#install python libraries
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

#copy source code
COPY . .

#run app
CMD ["python", "./main.py"]
