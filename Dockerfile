FROM python:3.11.5-bullseye
WORKDIR /Users/rachelletrotman/Documents/TUM/rvis
COPY . .
RUN pip3.11 install SimpleITK vtk pyqt5 numpy scipy SimpleITKUtilities
CMD ["python", "./main.py"]