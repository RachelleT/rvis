# rvis

## Running with Docker

### prerequisites
Install X server which is required to run GUI applications with docker.

For Windows:
* Install Xming (https://sourceforge.net/projects/xming/)
* Run executable
  ```
  Xming.exe -ac
  ```

For Mac:
* Install XQuartz and open it
  ```
  brew install --cask xquartz
  open -a XQuartz
  ```
* Allow connections from network clients (Settings --> security --> allow connections from network clients)
* Reboot
* Check XQuartz is running
  ```
  ps aux | grep Xquartz
  ```
* Allow X11 forwarding
  ```
  xhost +
  ```
  
### Run container

* Build image
  ```
  docker build -t rvis .
  ```
* Run with Windows
  ```
  set IP=<your-local-ip>
  docker run -e DISPLAY=$IP:0 rvis
  ```
* Run with Mac
  ```
  IP=$(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}')
  docker run -e DISPLAY=$IP:0 rvis
  ```
  



