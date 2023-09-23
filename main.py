#to avoid error 'could not load Qt platform plugin when running with docker'
import PyQt5
import os
dirname = os.path.dirname(PyQt5.__file__)
plugin_path = os.path.join(dirname, 'Qt5','plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

# This is a sample Python script.
import os
import sys
import SimpleITK as sitk
from SimpleITK.utilities.vtk import sitk2vtk, vtk2sitk
import vtk
from PyQt5.QtWidgets import QApplication, QMainWindow, QMdiArea, QMdiSubWindow, \
    QLabel, QPushButton, QDockWidget, QGridLayout, QLineEdit, QWidget, \
    QFrame, QScrollBar, QMessageBox, QListWidget, QAbstractItemView, QComboBox
from PyQt5 import QtCore, QtGui, QtWidgets
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import vtkLookupTable
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D, vtkStripper
from vtkmodules.vtkIOImage import vtkJPEGWriter, vtkTIFFWriter, vtkPNGWriter, vtkNIFTIImageWriter
from vtkmodules.vtkImagingCore import vtkImageMapToColors
from vtkmodules.vtkRenderingCore import vtkPolyDataMapper, vtkActor, vtkImageActor, vtkCamera
from utils.eval_utils import compute_dice, compute_hd95, binary_image, overlayMask


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

class AppWindow(QMainWindow):
    count = 0
    feature_count = 0
    checkboxes = []
    allfiles = []
    filepaths = []

    def __init__(self):
        super().__init__()
        self.setWindowIcon(QtGui.QIcon('icons/schwi_icon.png'))
        self.setWindowTitle("RVis")

        self.setAcceptDrops(True)  # for drag and drop

        # for feature panel
        self.fixedImage = QComboBox()
        self.movingImage = QComboBox()
        self.masks = QComboBox()
        self.binaryFlag = False

        # for adjusting the right panel via scrollbars
        self.axlFlag = False
        self.corFlag = False
        self.sagFlag = False

        self.mdi = QMdiArea()
        self.setCentralWidget(self.mdi)
        self.resize(2000, 800)

        self.menu_bar()
        self.tool_bar()
        self.docker_widget()
        self.docker_widgetFiles()
        self.show()

    def menu_bar(self):
        bar = self.menuBar()
        file = bar.addMenu('File')
        file_directory = self.create_action('Open Directory', 'icons/directory_icon.png', 'Ctrl+D', self.file_open_dir)
        file_image = self.create_action('Open Image', 'icons/upload_icon.png', 'Ctrl+N', self.file_open_img)
        save_image = self.create_action('Save Image', 'icons/save_icon.png', 'Ctrl+S', self.file_save_img)
        self.add_action(file, (file_directory, file_image, save_image))

        view = bar.addMenu('View')
        view_mode = self.create_action('Light Mode', 'icons/cube_icon.png', 'F2', self.docker_widget)
        view_tiled = self.create_action('Tiled Mode', 'icons/tile_icon.png', 'Ctrl+T', self.show_tiled)
        self.add_action(view, (view_mode, view_tiled))

    def tool_bar(self):
        navToolBar = self.addToolBar("Navigation")
        newAction = self.create_action('New', 'icons/plus_icon.png', 'Ctrl+D', self.file_open_dir)
        tileAction = self.create_action('Tiled Mode', 'icons/tile_icon.png', 'Ctrl+T', self.show_tiled)

        self.add_action(navToolBar, (newAction, tileAction))
        navToolBar.setFloatable(False)

    def create_action(self, text, icon=None, shortcut=None, implement=None, signal='triggered'):
        action = QtWidgets.QAction(text, self)
        if icon is not None:
            action.setIcon(QtGui.QIcon(icon))
        if shortcut is not None:
            action.setShortcut(shortcut)
        if implement is not None:
            getattr(action, signal).connect(implement)
        return action

    def add_action(self, dest, actions):
        for action in actions:
            if action is None:
                dest.addSeperator()
            else:
                dest.addAction(action)

    def dragEnterEvent(self, event):  # needed for drag event
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):  # needed for drop event
        if event.mimeData().hasImage:  # checks if dropped item is an image
            event.setDropAction(QtCore.Qt.CopyAction)
            file_path = event.mimeData().urls()[0].toLocalFile()  # retrieve file path of image
            AppWindow.filepaths.append(file_path)
            file = self.readImage(file_path)  # read an image
            AppWindow.allfiles.append(file)
            if self.binaryImageCheck(file):  # checks if dropped image is a mask
                self.binaryFlag = True
            self.add_dataset(file_path)  # add drop file to file manager
            if AppWindow.count == 1:
                self.vtk(file, -1)  # load panels and show image
            else:
                self.reloadWindows()
                self.vtk(file, -1)
            event.accept()
        else:
            event.ignore()

    def binaryImageCheck(self, imageFile):
        """ checks if an image is a binary image

        Parameters:
        imageFile: vtk object (self.reader.GetOutput())

        Returns: bool

        """
        inputImg = vtkImageData()
        inputImg.ShallowCopy(imageFile)
        npImage = sitk.GetArrayFromImage(vtk2sitk(inputImg))
        return binary_image(npImage)

    def readImage(self, filename):
        """ determine the appropriate vtk reader for an image

        Parameters:
        filename: string (file path)

        Returns: vtk object (self.reader.GetOutput())

        """
        if len(filename.split(".")) == 2: # gets file extension
            base, ext = filename.split(".")
        else:
            base, ext, fmt = filename.split(".")

        if ext in ('jpg', 'jpeg'):
            self.reader = vtk.vtkJPEGReader()
        elif ext in ('tif', 'tiff', "lsm"):
            self.reader = vtk.vtkTIFFReader()
        elif ext in ('png',):
            self.reader = vtk.vtkPNGReader()
        elif ext in ('nii', 'gz'):
            self.reader = vtk.vtkNIFTIImageReader()
        elif ext in ('bin'):
            # example for reading raw file with known header size and dims
            self.reader = vtk.vtkImageReader2()
            self.reader.SetHeaderSize(8394)
            self.reader.SetFileDimensionality(2)
            self.reader.SetDataScalarTypeToUnsignedShort()
            self.reader.SetDataByteOrderToLittleEndian()
            self.reader.SetDataExtent(0, 107, 0, 251, 0, 0)
            self.reader.SetDataSpacing(2.0, 1.0, 1.0)
        else:
            sys.stderr.write("cannot read file type %s\n" % (ext,))
            sys.exit(1)

        self.reader.SetFileName(filename)
        self.reader.Update()
        return self.reader.GetOutput()

    def sliderEvent(self):  # controls the sliding through slices of the volume
        self.sliderPosition = self.vScrollBar.sliderPosition()

        # update volume
        self.axial.SetDisplayExtent(0, 255, 0, 255, self.sliderPosition, self.sliderPosition)
        self.sagittal.SetDisplayExtent(self.sliderPosition, self.sliderPosition, 0, 255, 0, 223)

        if self.sliderPosition > 191:
            self.coronal.SetDisplayExtent(0, 223, 191, 191, 0, 223)
        else:
            self.coronal.SetDisplayExtent(0, 223, self.sliderPosition, self.sliderPosition, 0, 223)

        # update images
        if self.axlFlag:
            self.viewerCor.SetSlice(self.sliderPosition)
            self.viewerSag.SetSlice(self.sliderPosition)

            self.vScrollBarCor.setSliderPosition(self.sliderPosition)
            self.vScrollBarSag.setSliderPosition(self.sliderPosition)
        elif self.corFlag:
            self.viewer.SetSlice(self.sliderPosition)
            self.viewerSag.SetSlice(self.sliderPosition)

            self.vScrollBarAxl.setSliderPosition(self.sliderPosition)
            self.viewerSag.SetSlice(self.sliderPosition)
        elif self.sagFlag:
            self.viewer.SetSlice(self.sliderPosition)
            self.viewerCor.SetSlice(self.sliderPosition)

            self.vScrollBarAxl.setSliderPosition(self.sliderPosition)
            self.vScrollBarCor.setSliderPosition(self.sliderPosition)
        else:
            self.viewer.SetSlice(self.sliderPosition)
            self.viewerCor.SetSlice(self.sliderPosition)
            self.viewerSag.SetSlice(self.sliderPosition)

            # update sliders
            self.vScrollBarAxl.setSliderPosition(self.sliderPosition)
            self.vScrollBarCor.setSliderPosition(self.sliderPosition)
            self.vScrollBarSag.setSliderPosition(self.sliderPosition)

        # update widgets
        self.vtkWidgetAxl.update()
        self.vtkWidgetSag.update()
        self.vtkWidgetCor.update()
        self.vtkWidgetVol.update()

    def sliderEventPanels(self, p):  # controls the sliders for individual panels
        if p == 1:  # axial slider
            pos = self.vScrollBarAxl.sliderPosition()
            self.viewer.SetSlice(pos)
            self.axial.SetDisplayExtent(0, 255, 0, 255, pos, pos)
        elif p == 2:  # coronal slider
            pos = self.vScrollBarCor.sliderPosition()
            self.viewerCor.SetSlice(pos)
            if pos > 191:  # coronal slice only goes up to 191?
                self.coronal.SetDisplayExtent(0, 223, 191, 191, 0, 223)
            else:
                self.coronal.SetDisplayExtent(0, 223, pos, pos, 0, 223)
        else: # sagittal slider
            pos = self.vScrollBarSag.sliderPosition()
            self.viewerSag.SetSlice(pos)
            self.sagittal.SetDisplayExtent(pos, pos, 0, 255, 0, 223)

        self.vtkWidgetVol.update()

    def reloadWindows(self): # reload panels
        self.mdi.removeSubWindow(self.subSag)
        self.mdi.removeSubWindow(self.subVol)
        self.mdi.removeSubWindow(self.subAxl)
        self.mdi.removeSubWindow(self.subCor)
        self.ren.EraseOn()
        self.renVol.EraseOn()
        self.vtkWidgetVol.update()

    def show_tiled(self):
        self.mdi.tileSubWindows()

    def file_open_dir(self):  # load directory from menu bar
        AppWindow.count = AppWindow.count + 1
        self.filename = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder')

        if self.filename:
            self.reader = vtk.vtkDICOMImageReader()
            self.reader.SetDirectoryName(self.filename)
            self.reader.Update()
            self.add_dataset(self.filename)
            self.vtk(self.reader.GetOutput(), -1)

    def file_open_img(self):  # load file from menu bar
        self.filename = QtWidgets.QFileDialog.getOpenFileName(self, "Select File", "Images (*.jpeg *.jpg "
                                                                                   "*nii *gz)")

        if self.filename[0] != "":
            AppWindow.filepaths.append(self.filename[0])
            file = self.readImage(self.filename[0])  # self.filename is a tuple
            AppWindow.allfiles.append(file)
            self.add_dataset(self.filename[0])
            if AppWindow.count == 1:
                self.vtk(file, -1)
            else:
                self.reloadWindows()
                self.vtk(file, -1)

    def file_save_img(self):  # saving a file
        # selecting file path
        filePath, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Image", "",
                                                            "PNG(*.png);;JPEG(*.jpg *.jpeg);;NII(*.nii);;All Files(*.*) ")

        if filePath:
            # Select the writer to use.
            path, ext = os.path.splitext(filePath)
            ext = ext.lower()

            if ext in ('.jpg', '.jpeg'):
                writer = vtkJPEGWriter()
            elif ext in ('.tif', '.tiff', ".lsm"):
                writer = vtkTIFFWriter()
            elif ext in ('.png',):
                writer = vtkPNGWriter()
            elif ext in ('.nii', '.gz'):
                writer = vtkNIFTIImageWriter()
                # copy most information directory from the header
                writer.SetNIFTIHeader(self.reader.GetNIFTIHeader())
            else:
                print("to be updated")

            fileRow = self.filesListWidget.currentRow()

            writer.SetInputData(AppWindow.allfiles[fileRow])  # self.reader to be updated
            writer.SetFileName(filePath)
            # this information will override the reader's header
            writer.SetQFac(self.reader.GetQFac())
            writer.SetTimeDimension(self.reader.GetTimeDimension())
            writer.SetQFormMatrix(self.reader.GetQFormMatrix())
            writer.SetSFormMatrix(self.reader.GetSFormMatrix())
            writer.Write()

    def vtk(self, filename, volIndex):
        """ display image in the panels

            Parameters:
            filename: vtk object (self.reader.GetOutput())
            volIndex: int

        """

        if volIndex > 0:  # updates volume after applying a feature
            self.reader = vtk.vtkNIFTIImageReader()
            self.reader.SetFileName(AppWindow.filepaths[volIndex])
            self.reader.Update()

        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(0.2, 0.2, 0.2)
        self.subAxl = QMdiSubWindow()
        self.subAxl.setWindowTitle("Axial")  # Set Titlebar
        self.subAxl.setFixedSize(419, 367)
        self.frameAxl = QFrame()

        self.vtkWidgetAxl = QVTKRenderWindowInteractor(self.frameAxl)
        self.vtkWidgetAxl.setFixedSize(419, 367)
        self.subAxl.setWidget(self.vtkWidgetAxl)
        self.vtkWidgetAxl.GetRenderWindow().AddRenderer(self.ren)
        self.irenAxl = self.vtkWidgetAxl.GetRenderWindow().GetInteractor()

        self.viewer = vtk.vtkResliceImageViewer()
        self.viewer.SetInputData(filename)
        self.viewer.SetRenderWindow(self.vtkWidgetAxl.GetRenderWindow())

        style = vtk.vtkInteractorStyleImage()
        style.SetInteractionModeToImageSlicing()
        self.irenAxl.SetInteractorStyle(style)

        # cam = self.viewer.GetRenderer().GetActiveCamera()
        # cam.SetPosition(0, 0, -1)
        # cam.SetViewUp(0, -1, 0)

        # calculate index of middle slice in the dicom image
        maxSlice = self.viewer.GetSliceMax()
        midSlice = maxSlice / 2

        # set up reslice view properties
        self.viewer.SetSlice(int(midSlice))
        self.viewer.SetSliceOrientationToXY()
        self.viewer.GetRenderer().ResetCamera()
        self.viewer.Render()

        self.vScrollBarAxl = QScrollBar(self.subAxl)
        self.vScrollBarAxl.setEnabled(True)
        self.vScrollBarAxl.setOrientation(QtCore.Qt.Vertical)
        self.vScrollBarAxl.setGeometry(409, 30, 10, 367)
        self.vScrollBarAxl.setMinimum(self.viewer.GetSliceMin())
        self.vScrollBarAxl.setMaximum(maxSlice)
        self.vScrollBarAxl.setSliderPosition(int(midSlice))

        self.vtkWidgetAxl.update()

        # coronal

        self.subCor = QMdiSubWindow()
        self.subCor.setWindowTitle("Coronal")  # Set Titlebar
        self.subCor.setFixedSize(419, 367)
        self.frameCor = QFrame()

        self.vtkWidgetCor = QVTKRenderWindowInteractor(self.frameCor)
        self.vtkWidgetCor.setFixedSize(419, 367)
        self.subCor.setWidget(self.vtkWidgetCor)
        self.vtkWidgetCor.GetRenderWindow().AddRenderer(self.ren)
        self.irenCor = self.vtkWidgetCor.GetRenderWindow().GetInteractor()

        self.viewerCor = vtk.vtkResliceImageViewer()
        self.viewerCor.SetInputData(filename)
        self.viewerCor.SetRenderWindow(self.vtkWidgetCor.GetRenderWindow())

        styleCor = vtk.vtkInteractorStyleImage()
        styleCor.SetInteractionModeToImageSlicing()
        self.irenCor.SetInteractorStyle(styleCor)

        # set up reslice view properties
        self.viewerCor.SetSlice(int(midSlice))
        self.viewerCor.SetSliceOrientationToXZ()
        self.viewerCor.GetRenderer().ResetCamera()
        self.viewerCor.Render()

        self.vScrollBarCor = QScrollBar(self.subCor)
        self.vScrollBarCor.setEnabled(True)
        self.vScrollBarCor.setOrientation(QtCore.Qt.Vertical)
        self.vScrollBarCor.setGeometry(409, 30, 10, 367)
        self.vScrollBarCor.setMinimum(self.viewerCor.GetSliceMin())
        self.vScrollBarCor.setMaximum(maxSlice)
        self.vScrollBarCor.setSliderPosition(int(midSlice))

        self.vtkWidgetCor.update()

        # sagittal

        self.subSag = QMdiSubWindow()
        self.subSag.setWindowTitle("Sagittal")  # Set Titlebar
        self.subSag.setFixedSize(419, 367)
        self.frameSag = QFrame()

        self.vtkWidgetSag = QVTKRenderWindowInteractor(self.frameSag)
        self.vtkWidgetSag.setFixedSize(419, 367)
        self.subSag.setWidget(self.vtkWidgetSag)
        self.vtkWidgetSag.GetRenderWindow().AddRenderer(self.ren)
        self.irenSag = self.vtkWidgetSag.GetRenderWindow().GetInteractor()

        self.viewerSag = vtk.vtkResliceImageViewer()
        self.viewerSag.SetInputData(filename)
        self.viewerSag.SetRenderWindow(self.vtkWidgetSag.GetRenderWindow())

        styleSag = vtk.vtkInteractorStyleImage()
        styleSag.SetInteractionModeToImageSlicing()
        self.irenSag.SetInteractorStyle(styleSag)

        # set up reslice view properties
        self.viewerSag.SetSlice(int(midSlice))
        self.viewerSag.SetSliceOrientationToYZ()
        self.viewerSag.GetRenderer().ResetCamera()
        self.viewerSag.Render()

        self.vScrollBarSag = QScrollBar(self.subSag)
        self.vScrollBarSag.setEnabled(True)
        self.vScrollBarSag.setOrientation(QtCore.Qt.Vertical)
        self.vScrollBarSag.setGeometry(409, 30, 10, 367)
        self.vScrollBarSag.setMinimum(self.viewerSag.GetSliceMin())
        self.vScrollBarSag.setMaximum(maxSlice)
        self.vScrollBarSag.setSliderPosition(int(midSlice))

        self.vtkWidgetSag.update()

        # add event to scrollbars
        self.vScrollBarAxl.sliderMoved.connect(lambda: self.sliderEventPanels(1))
        self.vScrollBarCor.sliderMoved.connect(lambda: self.sliderEventPanels(2))
        self.vScrollBarSag.sliderMoved.connect(lambda: self.sliderEventPanels(3))

        # volume

        self.subVol = QMdiSubWindow()
        self.subVol.setWindowTitle("3D Volume")  # Set Titlebar
        self.subVol.setFixedSize(419, 367)
        self.frameVol = QFrame()

        self.vtkWidgetVol = QVTKRenderWindowInteractor(self.frameVol)
        self.subVol.setWidget(self.vtkWidgetVol)
        self.renVol = vtk.vtkRenderer()
        self.renVol.SetBackground(0.2, 0.2, 0.2)
        self.vtkWidgetVol.GetRenderWindow().AddRenderer(self.renVol)
        self.irenVol = self.vtkWidgetVol.GetRenderWindow().GetInteractor()

        self.imageDataVol = vtk.vtkImageData()
        self.imageDataVol.ShallowCopy(self.reader.GetOutput())

        self.volumeMapper = vtk.vtkSmartVolumeMapper()
        self.volumeProperty = vtk.vtkVolumeProperty()
        self.gradientOpacity = vtk.vtkPiecewiseFunction()
        self.scalarOpacity = vtk.vtkPiecewiseFunction()
        self.color = vtk.vtkColorTransferFunction()
        self.volume = vtk.vtkVolume()

        self.volumeMapper.SetBlendModeToComposite()
        self.volumeMapper.SetRequestedRenderModeToGPU()
        self.volumeMapper.SetInputData(self.imageDataVol)
        self.volumeProperty.ShadeOn()
        self.volumeProperty.SetInterpolationTypeToLinear()
        self.volumeProperty.SetAmbient(0.1)
        self.volumeProperty.SetDiffuse(0.9)
        self.volumeProperty.SetSpecular(0.2)
        self.volumeProperty.SetSpecularPower(10.0)
        self.gradientOpacity.AddPoint(0.0, 0.0)
        self.gradientOpacity.AddPoint(2000.0, 1.0)
        self.volumeProperty.SetGradientOpacity(self.gradientOpacity)
        self.scalarOpacity.AddPoint(-800.0, 0.0)
        self.scalarOpacity.AddPoint(-750.0, 1.0)
        self.scalarOpacity.AddPoint(-350.0, 1.0)
        self.scalarOpacity.AddPoint(-300.0, 0.0)
        self.scalarOpacity.AddPoint(-200.0, 0.0)
        self.scalarOpacity.AddPoint(-100.0, 1.0)
        self.scalarOpacity.AddPoint(1000.0, 0.0)
        self.scalarOpacity.AddPoint(2750.0, 0.0)
        self.scalarOpacity.AddPoint(2976.0, 1.0)
        self.scalarOpacity.AddPoint(3000.0, 0.0)
        self.volumeProperty.SetScalarOpacity(self.scalarOpacity)
        self.color.AddRGBPoint(-750.0, 0.08, 0.05, 0.03)
        self.color.AddRGBPoint(-350.0, 0.39, 0.25, 0.16)
        self.color.AddRGBPoint(-200.0, 0.80, 0.80, 0.80)
        self.color.AddRGBPoint(2750.0, 0.70, 0.70, 0.70)
        self.color.AddRGBPoint(3000.0, 0.35, 0.35, 0.35)
        self.volumeProperty.SetColor(self.color)
        self.volume.SetMapper(self.volumeMapper)
        self.volume.SetProperty(self.volumeProperty)

        self.renVol.AddVolume(self.volume)

        # skin

        self.colors = vtkNamedColors()

        skin_extractor = vtkFlyingEdges3D()
        skin_extractor.SetInputConnection(self.reader.GetOutputPort())
        skin_extractor.SetValue(0, 500)
        skin_extractor.Update()

        skin_stripper = vtkStripper()
        skin_stripper.SetInputConnection(skin_extractor.GetOutputPort())
        skin_stripper.Update()

        skin_mapper = vtkPolyDataMapper()
        skin_mapper.SetInputConnection(skin_stripper.GetOutputPort())
        skin_mapper.ScalarVisibilityOff()

        skin = vtkActor()
        skin.SetMapper(skin_mapper)
        skin.GetProperty().SetDiffuseColor(self.colors.GetColor3d('SkinColor'))
        skin.GetProperty().SetSpecular(0.3)
        skin.GetProperty().SetSpecularPower(20)

        # bone

        bone_extractor = vtkFlyingEdges3D()
        bone_extractor.SetInputConnection(self.reader.GetOutputPort())
        bone_extractor.SetValue(0, 1150)

        bone_stripper = vtkStripper()
        bone_stripper.SetInputConnection(bone_extractor.GetOutputPort())

        bone_mapper = vtkPolyDataMapper()
        bone_mapper.SetInputConnection(bone_stripper.GetOutputPort())
        bone_mapper.ScalarVisibilityOff()

        bone = vtkActor()
        bone.SetMapper(bone_mapper)
        bone.GetProperty().SetDiffuseColor(self.colors.GetColor3d('Ivory'))

        # Start by creating a black/white lookup table.
        bw_lut = vtkLookupTable()
        bw_lut.SetTableRange(0, 2000)
        bw_lut.SetSaturationRange(0, 0)
        bw_lut.SetHueRange(0, 0)
        bw_lut.SetValueRange(0, 1)
        bw_lut.Build()  # effective built

        # Now create a lookup table that consists of the full hue circle
        # (from HSV).
        hue_lut = vtkLookupTable()
        hue_lut.SetTableRange(0, 2000)
        hue_lut.SetHueRange(0, 1)
        hue_lut.SetSaturationRange(1, 1)
        hue_lut.SetValueRange(1, 1)
        hue_lut.Build()  # effective built

        # Finally, create a lookup table with a single hue but having a range
        # in the saturation of the hue.
        sat_lut = vtkLookupTable()
        sat_lut.SetTableRange(0, 2000)
        sat_lut.SetHueRange(0.6, 0.6)
        sat_lut.SetSaturationRange(0, 1)
        sat_lut.SetValueRange(1, 1)
        sat_lut.Build()  # effective built

        # get slices
        # calculate index of middle slice in data
        self.viewerVol = vtk.vtkResliceImageViewer()
        self.viewerVol.SetInputData(self.reader.GetOutput())
        self.minSlice = self.viewerVol.GetSliceMin()
        self.maxSlice = self.viewerVol.GetSliceMax()
        self.midSlice = self.maxSlice / 2

        sagittal_colors = vtkImageMapToColors()
        sagittal_colors.SetInputConnection(self.reader.GetOutputPort())
        sagittal_colors.SetLookupTable(bw_lut)
        sagittal_colors.Update()

        self.sagittal = vtkImageActor()
        self.sagittal.GetMapper().SetInputConnection(sagittal_colors.GetOutputPort())
        self.sagittal.SetDisplayExtent(int(self.midSlice), int(self.midSlice), 0, 255, 0, 223)
        self.sagittal.ForceOpaqueOn()

        # Create the second (axial) plane of the three planes. We use the
        # same approach as before except that the extent differs.
        axial_colors = vtkImageMapToColors()
        axial_colors.SetInputConnection(self.reader.GetOutputPort())
        axial_colors.SetLookupTable(hue_lut)
        axial_colors.Update()

        self.axial = vtkImageActor()
        self.axial.GetMapper().SetInputConnection(axial_colors.GetOutputPort())
        self.axial.SetDisplayExtent(0, 255, 0, 255, int(self.midSlice), int(self.midSlice))
        self.axial.ForceOpaqueOn()

        # Create the third (coronal) plane of the three planes. We use
        # the same approach as before except that the extent differs.
        coronal_colors = vtkImageMapToColors()
        coronal_colors.SetInputConnection(self.reader.GetOutputPort())
        coronal_colors.SetLookupTable(sat_lut)
        coronal_colors.Update()

        self.coronal = vtkImageActor()
        self.coronal.GetMapper().SetInputConnection(coronal_colors.GetOutputPort())
        self.coronal.SetDisplayExtent(0, 223, int(self.midSlice), int(self.midSlice), 0, 223)
        self.coronal.ForceOpaqueOn()

        styleVol = vtk.vtkInteractorStyle3D()
        styleVol.SetInteractor(self.irenVol)

        a_camera = vtkCamera()
        a_camera.SetViewUp(0, 0, -1)
        a_camera.SetPosition(0, -1, 0)
        a_camera.SetFocalPoint(0, 0, 0)
        a_camera.ComputeViewPlaneNormal()
        a_camera.Azimuth(30.0)
        a_camera.Elevation(30.0)

        self.renVol.SetActiveCamera(a_camera)
        self.renVol.ResetCamera()
        self.renVol.AddActor(self.sagittal)
        self.renVol.AddActor(self.axial)
        self.renVol.AddActor(self.coronal)
        self.renVol.AddActor(bone)
        self.renVol.AddActor(skin)

        self.vScrollBar = QScrollBar(self.subVol)
        self.vScrollBar.setEnabled(True)
        self.vScrollBar.setOrientation(QtCore.Qt.Vertical)
        self.vScrollBar.setGeometry(409, 30, 10, 367)
        self.vScrollBar.setMinimum(self.minSlice)
        self.vScrollBar.setMaximum(self.maxSlice)
        self.vScrollBar.setSliderPosition(int(self.midSlice))
        self.vScrollBar.sliderMoved.connect(self.sliderEvent)

        self.vtkWidgetVol.update()

        # add windows to central widget

        self.mdi.addSubWindow(self.subAxl)
        self.subAxl.show()
        self.irenAxl.Initialize()
        self.irenAxl.Start()

        self.mdi.addSubWindow(self.subCor)
        self.subCor.show()
        self.irenCor.Initialize()
        self.irenCor.Start()

        self.mdi.addSubWindow(self.subSag)
        self.subSag.show()
        self.irenSag.Initialize()
        self.irenSag.Start()

        self.mdi.addSubWindow(self.subVol)
        self.subVol.show()
        self.irenVol.Initialize()
        self.irenVol.Start()

    def checkerboardFeature(self):  # set up checkerboard layout
        fixedImageCheckerboard = QLabel('Fixed Image:')
        movingImageCheckerboard = QLabel('Moving Image:')

        tileNumberCheckerboard = QLabel('Number of tiles:')
        self.tileNumberInput = QLineEdit(self)
        self.tileNumberInput.setFixedWidth(60)

        checkerboardButton = QPushButton('Apply', self)

        checkerboardInput = QWidget()
        gridCheckerboard = QGridLayout()
        gridCheckerboard.addWidget(fixedImageCheckerboard, 1, 0)
        gridCheckerboard.addWidget(self.fixedImage, 1, 1)
        gridCheckerboard.addWidget(movingImageCheckerboard, 2, 0)
        gridCheckerboard.addWidget(self.movingImage, 2, 1)
        gridCheckerboard.addWidget(tileNumberCheckerboard, 3, 0)
        gridCheckerboard.addWidget(self.tileNumberInput, 3, 1)
        gridCheckerboard.addWidget(checkerboardButton, 4, 0)

        checkerboardInput.setLayout(gridCheckerboard)
        self.dockWidPanel.setWidget(checkerboardInput)

        checkerboardButton.clicked.connect(self.showCheckerboard)

    def showCheckerboard(self):  # applies checkerboard feature
        fixed = self.fixedImage.currentIndex()
        moving = self.movingImage.currentIndex()

        if self.tileNumberInput.text() == "":
            QMessageBox.about(self, "Oops!", "Enter number of tiles.")
        else:
            size = int(self.tileNumberInput.text())
            if size <= 0:
                QMessageBox.about(self, "Oops!", "Tile number ( > 0 ) is needed for this feature.")
            else:
                self.reloadWindows()

                fixedimg = vtkImageData()
                fixedimg.ShallowCopy(AppWindow.allfiles[fixed])

                movingimg = vtkImageData()
                movingimg.ShallowCopy(AppWindow.allfiles[moving])

                checkerboard = sitk.CheckerBoard(vtk2sitk(fixedimg), vtk2sitk(movingimg),
                                                 (size, size, size))

                self.saveFeature(sitk2vtk(checkerboard), fixed, 'c')

    def saveFeature(self, file, volumeImage, flag):
        """ display image in the panels after applying a feature

            Parameters:
            file: vtk object
            volumeImage: int (index position of image to be rendered in the volume
            flag: string (determines which feature was applied

        """
        AppWindow.feature_count += 1
        self.vIndex = volumeImage
        AppWindow.allfiles.append(file)

        # add new image with feature applied to the file manager

        if flag == 'c':
            AppWindow.filepaths.append(
                '    checkerboard-t' + self.tileNumberInput.text() + '-' + str(AppWindow.feature_count))
            self.filesListWidget.addItem(
                '    checkerboard-t' + self.tileNumberInput.text() + '-' + str(AppWindow.feature_count))
            self.fixedImage.addItem(
                '    checkerboard-t' + self.tileNumberInput.text() + '-' + str(AppWindow.feature_count))
            self.movingImage.addItem(
                '    checkerboard-t' + self.tileNumberInput.text() + '-' + str(AppWindow.feature_count))
        elif flag == 'o':
            AppWindow.filepaths.append(
                '    mask-overlay-a' + self.alphaBlendNumber.text() + '-' + str(AppWindow.feature_count))
            self.filesListWidget.addItem(
                '    mask-overlay-a' + self.alphaBlendNumber.text() + '-' + str(AppWindow.feature_count))
            self.fixedImage.addItem(
                '    mask-overlay-a' + self.alphaBlendNumber.text() + '-' + str(AppWindow.feature_count))
            self.masks.addItem('    mask-overlay-a' + self.alphaBlendNumber.text() + '-' + str(AppWindow.feature_count))
        elif flag == 'd':
            AppWindow.filepaths.append('    difference-image-' + str(AppWindow.feature_count))
            self.filesListWidget.addItem('    difference-image-' + str(AppWindow.feature_count))
            self.fixedImage.addItem('    difference-image-' + str(AppWindow.feature_count))
            self.movingImage.addItem('    difference-image-' + str(AppWindow.feature_count))

        self.vtk(file, self.vIndex)

        AppWindow.count = AppWindow.count + 1

    def overlayFeature(self):  # set up mask overlay layout

        baseImage = QLabel('Base:')
        maskImage = QLabel('Mask:')
        colorMap = QLabel('Mask Colormap: ')
        alphaBlend = QLabel('Alpha Blend: ')

        self.maskColormap = QComboBox()
        self.maskColormap.addItems(['Hot', 'Jet'])

        self.alphaBlendNumber = QLineEdit()

        overlayButton = QPushButton('Apply', self)

        overlayInput = QWidget()
        gridOverlay = QGridLayout()

        gridOverlay.addWidget(baseImage, 1, 0)
        gridOverlay.addWidget(self.fixedImage, 1, 1)
        gridOverlay.addWidget(maskImage, 2, 0)
        gridOverlay.addWidget(self.masks, 2, 1)
        gridOverlay.addWidget(colorMap, 3, 0)
        gridOverlay.addWidget(self.maskColormap, 3, 1)
        gridOverlay.addWidget(alphaBlend, 4, 0)
        gridOverlay.addWidget(self.alphaBlendNumber, 4, 1)
        gridOverlay.addWidget(overlayButton, 5, 0)

        overlayInput.setLayout(gridOverlay)
        self.dockWidPanel.setWidget(overlayInput)

        overlayButton.clicked.connect(self.showOverlay)

    def showOverlay(self):  # applies mask overlay feature
        fixedBase = self.fixedImage.currentIndex()

        for x in range(self.filesListWidget.count()):
            if self.filesListWidget.item(x).text() == self.masks.currentText():
                selectedMask = x
                break

        baseimg = vtkImageData()
        baseimg.ShallowCopy(AppWindow.allfiles[fixedBase])

        maskimg = vtkImageData()
        maskimg.ShallowCopy(AppWindow.allfiles[selectedMask])

        full = vtk2sitk(baseimg)
        mask = vtk2sitk(maskimg)

        sitkOverlay = overlayMask(full, mask, self.maskColormap.currentText(), float(self.alphaBlendNumber.text()))

        self.reloadWindows()

        self.saveFeature(sitk2vtk(sitkOverlay), fixedBase, 'o')

    def differenceFeature(self):  # set up difference image layout
        fixedDifferenceImage = QLabel('Fixed Image:')
        movingDifferenceImage = QLabel('Moving Image:')

        differenceImageButton = QPushButton('Apply', self)

        differenceImageInput = QWidget()
        gridDifferenceImage = QGridLayout()

        gridDifferenceImage.addWidget(fixedDifferenceImage, 1, 0)
        gridDifferenceImage.addWidget(self.fixedImage, 1, 1)
        gridDifferenceImage.addWidget(movingDifferenceImage, 2, 0)
        gridDifferenceImage.addWidget(self.movingImage, 2, 1)
        gridDifferenceImage.addWidget(differenceImageButton, 3, 0)

        differenceImageInput.setLayout(gridDifferenceImage)
        self.dockWidPanel.setWidget(differenceImageInput)

        differenceImageButton.clicked.connect(self.showDifferenceImage)

    def showDifferenceImage(self):  # applies difference image feature
        fixedDI = self.fixedImage.currentIndex()
        movingDI = self.movingImage.currentIndex()

        self.reloadWindows()

        fixedImgDI = vtkImageData()
        fixedImgDI.ShallowCopy(AppWindow.allfiles[fixedDI])

        movingImgDI = vtkImageData()
        movingImgDI.ShallowCopy(AppWindow.allfiles[movingDI])

        imageMath = vtk.vtkImageMathematics()
        imageMath.SetOperationToSubtract()
        imageMath.SetInput1Data(fixedImgDI)
        imageMath.SetInput2Data(movingImgDI)
        imageMath.Update()

        self.saveFeature(imageMath.GetOutput(), fixedDI, 'd')

    def plugin_handler(self):  # handles the features available
        item = str(self.pluginsListWidget.selectedItems()[0].text())
        if item == 'Checkerboard':
            self.checkerboardFeature()
        elif item == 'Mask Overlay':
            self.overlayFeature()
        elif item == 'Difference Image':
            self.differenceFeature()

    def docker_widget(self):  # right panel

        # first panel
        dockWid = QDockWidget('Features', self)
        dockWid.setFixedWidth(300)
        dockWid.setFixedHeight(245)
        dockWid.setFeatures(QDockWidget.NoDockWidgetFeatures)

        self.pluginsListWidget = QListWidget()
        self.pluginsListWidget.addItem('Checkerboard')
        self.pluginsListWidget.addItem('Mask Overlay')
        self.pluginsListWidget.addItem('Difference Image')
        self.pluginsListWidget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.pluginsListWidget.itemClicked.connect(self.plugin_handler)

        dockWid.setWidget(self.pluginsListWidget)
        dockWid.setFloating(False)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dockWid)

        # second panel
        dockWidEval = QDockWidget('Metrics', self)
        dockWidEval.setFixedWidth(300)
        dockWidEval.setFixedHeight(245)
        dockWidEval.setFeatures(QDockWidget.NoDockWidgetFeatures)

        self.evalMetricsListWidget = QListWidget()
        self.evalMetricsListWidget.addItem('Dice')
        self.evalMetricsListWidget.addItem('Hausdorff')
        self.evalMetricsListWidget.itemClicked.connect(self.metric_handler)

        dockWidEval.setWidget(self.evalMetricsListWidget)
        dockWidEval.setFloating(False)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dockWidEval)

        # third panel
        self.dockWidPanel = QDockWidget('Feature Panel', self)
        self.dockWidPanel.setFixedWidth(300)
        self.dockWidPanel.setFixedHeight(245)
        self.dockWidPanel.setFeatures(QDockWidget.NoDockWidgetFeatures)

        self.dockWidPanel.setFloating(False)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.dockWidPanel)

    def docker_widgetFiles(self):  # left panel

        filesDock = QDockWidget('Data Manager', self)
        filesDock.setFixedWidth(300)
        filesDock.setFeatures(QDockWidget.NoDockWidgetFeatures)

        self.filesListWidget = QListWidget()
        self.filesListWidget.itemClicked.connect(self.listWidgetClicked)

        filesDock.setWidget(self.filesListWidget)
        filesDock.setFloating(False)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, filesDock)

    def listWidgetClicked(self, item):  # handles switching between images in the data manager
        selectedRow = self.filesListWidget.currentRow()
        itemName = item.text()

        if '.' not in itemName:
            self.reloadWindows()
            self.vtk(AppWindow.allfiles[selectedRow], self.vIndex)
        else:
            self.reloadWindows()
            self.vtk(AppWindow.allfiles[selectedRow], selectedRow)

    def metric_handler(self):  # handles the metrics available
        item = str(self.evalMetricsListWidget.selectedItems()[0].text())
        if item == 'Dice':
            self.diceFlag = True
            self.metrics()
        elif item == 'Hausdorff':
            self.hdFlag = True
            self.metrics()

    def metrics(self):  # set up the metric layout
        fixedS = QLabel('Fixed:')
        movingS = QLabel('Moving:')
        resultMetric = QLabel('Score:')

        self.masksFixed = QComboBox()
        self.masksMoving = QComboBox()

        for x in range(self.masks.count()):
            self.masksFixed.addItem(self.masks.itemText(x))
            self.masksMoving.addItem(self.masks.itemText(x))

        self.resultScore = QLabel()

        metricButton = QPushButton('Apply', self)

        metricInput = QWidget()
        gridMetric = QGridLayout()

        gridMetric.addWidget(fixedS, 1, 0)
        gridMetric.addWidget(self.masksFixed, 1, 1)
        gridMetric.addWidget(movingS, 2, 0)
        gridMetric.addWidget(self.masksMoving, 2, 1)
        gridMetric.addWidget(resultMetric, 3, 0)
        gridMetric.addWidget(self.resultScore, 3, 1)
        gridMetric.addWidget(metricButton, 4, 0)

        metricInput.setLayout(gridMetric)
        self.dockWidPanel.setWidget(metricInput)

        metricButton.clicked.connect(self.showMetric)

    def showMetric(self):  # applying a metric
        setFixed = False
        setMoving = False

        for x in AppWindow.filepaths:
            if self.masksFixed.currentText() in x:
                mFixed = self.readImage(x)
                setFixed = True
            if self.masksMoving.currentText() in x:
                mMoving = self.readImage(x)
                setMoving = True
            if setFixed and setMoving:
                break

        if self.diceFlag:
            dice_score = compute_dice(sitk.GetArrayFromImage(vtk2sitk(mFixed)),
                                      sitk.GetArrayFromImage(vtk2sitk(mMoving)), [1])
            self.resultScore.setText(str(dice_score[0]))
        else:
            hd_score = compute_hd95(sitk.GetArrayFromImage(vtk2sitk(mFixed)),
                                    sitk.GetArrayFromImage(vtk2sitk(mMoving)), [1])
            self.resultScore.setText(str(hd_score[0]))

    def add_dataset(self, filename):
        """ manages files loaded including adding them to the data manager

            Parameters:
            filename: string (file location)

        """

        name = filename.split("/")[-1]
        self.filesListWidget.addItem(name)
        if self.binaryFlag:
            self.masks.addItem(name)
        else:
            self.fixedImage.addItem(name)
            self.movingImage.addItem(name)

        self.binaryFlag = False
        AppWindow.count = AppWindow.count + 1


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app_win = AppWindow()
    sys.exit(app.exec())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
