# This is a sample Python script.
import os
import sys
import SimpleITK as sitk
from SimpleITK.utilities import sitk2vtk, vtk2sitk
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


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

class AppWindow(QMainWindow):
    count = 0
    checkboxes = []
    allfiles = []
    filepaths = []

    def __init__(self):
        super().__init__()
        self.setWindowIcon(QtGui.QIcon('icons/schwi_icon.png'))
        self.setWindowTitle("RVis")

        self.setAcceptDrops(True)
        self.fixedImage = QComboBox()
        self.movingImage = QComboBox()
        self.maskImage = QComboBox()

        self.mdi = QMdiArea()
        self.setCentralWidget(self.mdi)
        self.resize(2000, 800)

        self.menu_bar()
        self.tool_bar()
        self.docker_widget()
        self.docker_widgetR()
        self.show()

    def menu_bar(self):
        bar = self.menuBar()
        file = bar.addMenu('File')
        file_directory = self.create_action('Open Directory', 'icons/directory_icon.png', 'Ctrl+D', self.file_open_dir)
        file_image = self.create_action('Open Image', 'icons/upload_icon.png', 'Ctrl+N', self.file_open_img)
        save_image = self.create_action('Save Image', 'icons/save_icon.png', 'Ctrl+S', self.file_save_img)
        # file_exit = self.create_action('Exit', 'icons/exit_icon.png', 'Ctrl+Q', self.close)
        self.add_action(file, (file_directory, file_image, save_image))

        view = bar.addMenu('View')
        view_shortcut = self.create_action('Show navigation', 'icons/navi_icon.png', 'F4', self.tool_bar)
        view_variable = self.create_action('Show tool widget', 'icons/tool_icon.png', 'F2', self.docker_widget)
        # view_restore = self.create_action('Restore', 'icons/restore_icon.png', 'Ctrl+Y', self.file_open)
        view_tiled = self.create_action('Tiled Mode', 'icons/tile_icon.png', 'Ctrl+T', self.show_tiled)
        # self.add_action(view, (view_shortcut, view_variable, view_restore, view_tiled))
        self.add_action(view, (view_shortcut, view_variable, view_tiled))

    def tool_bar(self):
        navToolBar = self.addToolBar("Navigation")
        newAction = self.create_action('New', 'icons/plus_icon.png', 'Ctrl+D', self.file_open_dir)
        tileAction = self.create_action('Tiled Mode', 'icons/tile_icon.png', 'Ctrl+T', self.show_tiled)
        toolAction = self.create_action('Show tool widget', 'icons/tool_icon.png', 'F2', self.docker_widget)

        self.add_action(navToolBar, (newAction, tileAction, toolAction))
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

    def dragEnterEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasImage:
            event.setDropAction(QtCore.Qt.CopyAction)
            file_path = event.mimeData().urls()[0].toLocalFile()
            AppWindow.filepaths.append(file_path) # review this with the overlap function
            file = self.readImage(file_path)
            AppWindow.allfiles.append(file)
            self.add_dataset(file_path)
            if AppWindow.count == 1:
                self.vtk(file, "f")
            if AppWindow.count == 2:
                self.mdi.removeSubWindow(self.subSag)
                self.mdi.removeSubWindow(self.subVol)
                self.mdi.removeSubWindow(self.subAxl)
                self.mdi.removeSubWindow(self.subCor)
                self.ren.EraseOn()
                self.renVol.EraseOn()
                self.vtkWidgetVol.update()
                self.vtk(file, "f")
            event.accept()
        else:
            event.ignore()

    def readImage(self, filename):

        if len(filename.split(".")) == 2:
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

    def showStateFileButton(self):  # file button
        print("to do")

    def sliderEvent(self):
        self.sliderPosition = self.vScrollBar.sliderPosition()
        zmax = self.sliderPosition * 2
        if zmax > self.maxSlice:
            zmax = self.maxSlice
        zmid = self.sliderPosition
        zmin = zmid - zmax
        if zmin < 0:
            zmin = 0

        # update volume
        self.axial.SetDisplayExtent(0, 255, 0, 255, zmid, zmid)
        self.sagittal.SetDisplayExtent(128, 128, 0, 255, zmin, zmax)
        self.coronal.SetDisplayExtent(0, 255, 128, 128, zmin, zmax)

        # update images
        self.viewer.SetSlice(self.sliderPosition)
        self.viewerSag.SetSlice(self.sliderPosition)
        self.viewerCor.SetSlice(self.sliderPosition)

        # update widgets
        self.vtkWidgetAxl.update()
        self.vtkWidgetSag.update()
        self.vtkWidgetCor.update()
        self.vtkWidgetVol.update()

    def reloadWindows(self):
        self.mdi.removeSubWindow(self.subSag)
        self.mdi.removeSubWindow(self.subVol)
        self.mdi.removeSubWindow(self.subAxl)
        self.mdi.removeSubWindow(self.subCor)
        self.ren.EraseOn()
        self.renVol.EraseOn()
        self.vtkWidgetVol.update()

    def show_tiled(self):
        self.mdi.tileSubWindows()

    def file_open_dir(self):
        AppWindow.count = AppWindow.count + 1
        self.filename = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder')

        if self.filename:
            self.reader = vtk.vtkDICOMImageReader()
            self.reader.SetDirectoryName(self.filename)
            self.reader.Update()
            self.add_dataset(self.filename)
            self.vtk(self.reader.GetOutput(), "d")

    def file_open_img(self):
        self.filename = QtWidgets.QFileDialog.getOpenFileName(self, "Select File", "Images (*.jpeg *.jpg "
                                                                                   "*nii *gz)")

        if self.filename[0] != "":
            file = self.readImage(self.filename[0])  # self.filename is a tuple
            self.add_dataset(self.filename[0])
            self.vtk(file, "f")

    def file_save_img(self):
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

            writer.SetInputConnection(self.reader.GetOutputPort()) # self.reader to be updated
            writer.SetFileName(filePath)
            # this information will override the reader's header
            writer.SetQFac(self.reader.GetQFac())
            writer.SetTimeDimension(self.reader.GetTimeDimension())
            writer.SetQFormMatrix(self.reader.GetQFormMatrix())
            writer.SetSFormMatrix(self.reader.GetSFormMatrix())
            writer.Write()

    def vtk(self, filename, flag):

        if flag == 's':
            self.reader = vtk.vtkNIFTIImageReader()
            self.reader.SetFileName(AppWindow.filepaths[0])
            self.reader.Update()

        # Start by creating a black/white lookup table.
        bw_lut = vtkLookupTable()
        bw_lut.SetTableRange(0, 2000)
        bw_lut.SetSaturationRange(0, 0)
        bw_lut.SetHueRange(0, 0)
        bw_lut.SetValueRange(0, 1)
        bw_lut.Build()  # effective built

        self.ren = vtk.vtkRenderer()

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
        self.viewer.SetLookupTable(bw_lut)
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
        self.viewerCor.SetLookupTable(bw_lut)
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
        self.viewerSag.SetLookupTable(bw_lut)
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

        self.vtkWidgetSag.update()

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
        self.sagittal.SetDisplayExtent(128, 128, 0, 255, 0, 223)
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
        self.coronal.SetDisplayExtent(0, 255, 128, 128, 0, 233)
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

        self.mdi.addSubWindow(self.subVol)
        self.subVol.show()
        self.irenVol.Initialize()
        self.irenVol.Start()

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

    def checkerboardFeature(self):
        fixedImageCheckerboard = QLabel('Fixed Image:')
        movingImageCheckerboard = QLabel('Moving Image:')

        tileSizeCheckerboard = QLabel('Tile Size:')
        self.tileSizeInput = QLineEdit(self)
        self.tileSizeInput.setFixedWidth(60)

        checkerboardButton = QPushButton('Apply', self)

        checkerboardInput = QWidget()
        gridCheckerboard = QGridLayout()
        gridCheckerboard.addWidget(fixedImageCheckerboard, 1, 0)
        gridCheckerboard.addWidget(self.fixedImage, 1, 1)
        gridCheckerboard.addWidget(movingImageCheckerboard, 2, 0)
        gridCheckerboard.addWidget(self.movingImage, 2, 1)
        gridCheckerboard.addWidget(tileSizeCheckerboard, 3, 0)
        gridCheckerboard.addWidget(self.tileSizeInput, 3, 1)
        gridCheckerboard.addWidget(checkerboardButton, 4, 0)

        checkerboardInput.setLayout(gridCheckerboard)
        self.dockWidPanel.setWidget(checkerboardInput)

        checkerboardButton.clicked.connect(self.showCheckerboard)


    def showCheckerboard(self):
        fixed = self.fixedImage.currentIndex()
        moving = self.movingImage.currentIndex()

        size = int(self.tileSizeInput.text())

        if size <= 0:
            QMessageBox.about(self, "Oops!", "Tile size ( > 0 ) is needed for this feature.")
        else:
            self.reloadWindows()

            fixedimg = vtkImageData()
            fixedimg.ShallowCopy(AppWindow.allfiles[fixed])

            movingimg = vtkImageData()
            movingimg.ShallowCopy(AppWindow.allfiles[moving])

            checkerboard = sitk.CheckerBoard(vtk2sitk(fixedimg), vtk2sitk(movingimg),
                                             (size, size, size))

            self.vtk(sitk2vtk(checkerboard), "c")

    def overlapFeature(self):

        baseImage = QLabel('Base:')
        maskImage = QLabel('Mask:')
        overlapButton = QPushButton('Apply', self)

        overlapInput = QWidget()
        gridOverlap = QGridLayout()

        gridOverlap.addWidget(baseImage, 1, 0)
        gridOverlap.addWidget(self.fixedImage, 1, 1)
        gridOverlap.addWidget(maskImage, 2, 0)
        gridOverlap.addWidget(self.maskImage, 2, 1)
        gridOverlap.addWidget(overlapButton, 3, 0)

        overlapInput.setLayout(gridOverlap)
        self.dockWidPanel.setWidget(overlapInput)

        overlapButton.clicked.connect(self.showOverlap)


    def showOverlap(self):
        fixedBase = self.fixedImage.currentIndex()
        selectedMask = self.maskImage.currentIndex()

        self.reloadWindows()

        baseimg = vtkImageData()
        baseimg.ShallowCopy(AppWindow.allfiles[fixedBase])

        maskimg = vtkImageData()
        maskimg.ShallowCopy(AppWindow.allfiles[selectedMask])

        full = vtk2sitk(baseimg)
        mask = vtk2sitk(maskimg)

        green = [0, 255, 0]

        background = sitk.LabelOverlay(image=sitk.Cast(full, sitk.sitkUInt16),
                                           labelImage=sitk.Cast(mask, sitk.sitkUInt8),
                                           opacity=0.8, backgroundValue=0)

        mask = sitk.LabelOverlay(image=sitk.Cast(full, sitk.sitkUInt16),
                                     labelImage=sitk.Cast(mask, sitk.sitkUInt8),
                                     opacity=0.8, backgroundValue=-1.0, colormap=green)

        # dice_score = compute_dice_coefficient(sitk.GetArrayFromImage(full),sitk.GetArrayFromImage(mask))
        # print(dice_score)

        image_blender = vtk.vtkImageBlend()
        image_blender.SetBlendModeToCompound()
        image_blender.SetCompoundAlpha(True)
        image_blender.AddInputData(sitk2vtk(background))
        image_blender.AddInputData(sitk2vtk(mask))
        image_blender.SetOpacity(0, 0.5)
        image_blender.SetOpacity(1, 0.5)
        image_blender.Update()

        self.vtk(image_blender.GetOutput(), "s")

    def plugin_handler(self):
        item = str(self.pluginsListWidget.selectedItems()[0].text())
        if item == 'Checkerboard':
            self.checkerboardFeature()
        elif item == 'Label Overlap':
            self.overlapFeature()

    def docker_widget(self):

        # first panel
        dockWid = QDockWidget('Features', self)
        dockWid.setFixedWidth(300)
        dockWid.setFixedHeight(245)
        dockWid.setFeatures(QDockWidget.NoDockWidgetFeatures)

        self.pluginsListWidget = QListWidget()
        self.pluginsListWidget.addItem('Checkerboard')
        self.pluginsListWidget.addItem('Label Overlap')
        self.pluginsListWidget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.pluginsListWidget.itemClicked.connect(self.plugin_handler)

        #seg_btn = QPushButton('Label Overlap', self)
        #seg_btn.setFlat(True)
        #seg_btn.clicked.connect(self.overlap)

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
        self.evalMetricsListWidget.addItem('Average Distance')
        self.evalMetricsListWidget.addItem('Hausdorff')

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

    def docker_widgetR(self):

        filesDock = QDockWidget('Data Manager', self)
        filesDock.setFixedWidth(300)
        filesDock.setFeatures(QDockWidget.NoDockWidgetFeatures)

        self.filesListWidget = QListWidget()

        filesDock.setWidget(self.filesListWidget)
        filesDock.setFloating(False)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, filesDock)

    def add_dataset(self, filename):

        name = filename.split("/")[-1]
        self.filesListWidget.addItem(name)
        self.fixedImage.addItem(name)
        self.movingImage.addItem(name)
        self.maskImage.addItem(name)

        AppWindow.count = AppWindow.count + 1

    def zoom_in(self):
        self.ren.GetActiveCamera().Zoom(2.2)

    def zoom_out(self):
        self.ren.GetActiveCamera().Zoom(0.8)

    def thrDaxis(self):
        axesActor = vtk.vtkAxesActor()
        self.axes = vtk.vtkOrientationMarkerWidget()
        self.axes.SetOrientationMarker(axesActor)
        self.axes.SetInteractor(self.iren)
        self.axes.EnabledOn()
        self.axes.InteractiveOn()
        self.ren.ResetCamera()
        # self.frame.setLayout(self.vl)
        # self.setCentralWidget(self.frame)
        # self.show()
        # self.iren.Initialize()

    def thrBox(self):
        outline = vtk.vtkOutlineFilter()
        outline.SetInputConnection(self.reader.GetOutputPort())

        outlineMapper = vtk.vtkPolyDataMapper()
        outlineMapper.SetInputConnection(outline.GetOutputPort())

        self.outlineActor = vtk.vtkActor()
        self.outlineActor.SetMapper(outlineMapper)
        self.outlineActor.GetProperty().SetColor(1, 1, 1)
        self.renVol.AddActor(self.outlineActor)
        self.renVol.ResetCamera()

    def scaleXYZ(self):
        x = int(self.scaleX.text())
        y = int(self.scaleY.text())
        z = int(self.scaleZ.text())
        self.ren.Render()
        self.ren.EraseOff()
        self.outlineActor.SetScale(x, y, z)
        self.volume.SetScale(x, y, z)
        self.ren.Render()
        self.ren.EraseOn()

    def rotateXYZ(self):
        x = int(self.rotateX.text())
        y = int(self.rotateY.text())
        z = int(self.rotateZ.text())
        self.outlineActor.SetOrientation(x, y, z)
        self.volume.SetOrientation(x, y, z)
        self.ren.Render()
        self.ren.EraseOff()

        self.volume.RotateX(x)
        self.volume.RotateY(y)
        self.volume.RotateZ(z)
        self.outlineActor.RotateX(x)
        self.outlineActor.RotateY(y)
        self.outlineActor.RotateZ(z)

        self.ren.Render()
        self.ren.EraseOn()

    def translateXYZ(self):
        x = int(self.translateX.text())
        y = int(self.translateY.text())
        z = int(self.translateZ.text())
        # self.volume.SetOrientation(0, 0, 0)
        self.ren.Render()
        self.ren.EraseOff()
        self.outlineActor.SetPosition(x, y, z)
        self.volume.SetPosition(x, y, z)
        self.ren.Render()
        self.ren.EraseOn()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app_win = AppWindow()
    sys.exit(app.exec())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
