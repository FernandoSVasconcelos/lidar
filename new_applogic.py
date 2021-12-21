import paraview.simple as smp
import time

PATH = '/home/ubuntu/Downloads/new_csv/cap'

def capture():
    print("Starting Capture")
    path_jetson = '/mnt/SSD/temp/lidar/cap' 
    
    vv.saveCSVCurrentFrame("/home/ubuntu/Downloads/teste.csv")
    for i in range(10):
        vv.saveCSVCurrentFrame(PATH + str(i) + '.csv')

def openSensor():
    calibrationFile = "/home/ubuntu/Downloads/VeloView-4.1.3-Linux-64bit/share/VLP-16.xml"
    LidarPort = 2368
    GPSPort = 8308
    LIDARForwardingPort = 2368
    GPSForwardingPort = 8308
    isForwarding = 0
    ipAddressForwarding = "192.168.1.1"
    
    vv.close()
    vv.app.grid = vv.createGrid()

    sensor = vv.smp.LidarStream(guiName='Data', CalibrationFile=calibrationFile)

    if "velarray" in calibrationFile.lower():
        sensor.Interpreter = 'Velodyne Special Velarray Interpreter'
    else :
        sensor.Interpreter = 'Velodyne Meta Interpreter'

    sensor.Interpreter.UseIntraFiringAdjustment = vv.app.actions['actionIntraFiringAdjust'].isChecked()

    sensor.ListeningPort = LidarPort
    sensor.ForwardedPort = LIDARForwardingPort
    sensor.IsForwarding = isForwarding
    sensor.ForwardedIpAddress = ipAddressForwarding

    sensor.Interpreter.IgnoreZeroDistances = vv.app.actions['actionIgnoreZeroDistances'].isChecked()
    sensor.Interpreter.HideDropPoints = vv.app.actions['actionHideDropPoints'].isChecked()
    sensor.Interpreter.IgnoreEmptyFrames = vv.app.actions['actionIgnoreEmptyFrames'].isChecked()
    sensor.UpdatePipeline()
    sensor.Start()

    vv.app.sensor = sensor
    vv.app.trailingFramesSpinBox.enabled = False
    vv.app.colorByInitialized = False
    vv.app.filenameLabel.setText('Live sensor stream (Port:'+str(LidarPort)+')' )
    vv.app.positionPacketInfoLabel.setText('')
    vv.enableSaveActions()

    vv.onCropReturns(False) # Dont show the dialog just restore settings
    vv.restoreLaserSelection()

    rep = vv.smp.Show(sensor)
#    rep.InterpolateScalarsBeforeMvv.apping = 0
#    if vv.app.sensor.GetClientSideObject().GetNumberOfChannels() == 128:
#        rep.Representation = 'Point Cloud'
#        rep.ColorArrayName = 'intensity'

    vv.smp.Render()

    vv.showSourceInSpreadSheet(vv.app.trailingFrame)

    vv.app.actions['actionShowRPM'].enabled = True
    vv.app.actions['actionCorrectIntensityValues'].enabled = True
    vv.app.actions['actionFastRenderer'].enabled = True

    #Auto adjustment of the grid size with the distance resolution
    vv.app.DistanceResolutionM = sensor.Interpreter.GetClientSideObject().GetDistanceResolutionM()
    vv.app.actions['actionMeasurement_Grid'].setChecked(True)
    vv.showMeasurementGrid()

    # Always enable dual return mode selection. A warning will be raised if
    # there's no dual return on the current frame later on
    vv.app.actions['actionDualReturnModeDual'].enabled = True
    vv.app.actions['actionDualReturnDistanceNear'].enabled = True
    vv.app.actions['actionDualReturnDistanceFar'].enabled = True
    vv.app.actions['actionDualReturnIntensityHigh'].enabled = True
    vv.app.actions['actionDualReturnIntensityLow'].enabled = True

    # Hide Position Orientation Stream by default and select lidarStream as active source
    vv.smp.SetActiveSource(sensor)

    vv.updateUIwithNewLidar()
    vv.smp.Render()

    # In OpenSensor we don't have access to the futur available arrays
    nChannels = sensor.Interpreter.GetProperty("NumberOfChannelsInformation")[0]
    print(nChannels)
    print("nChannels")
    arrayName = "intensity"
    if nChannels == 128:
        arrayName = "reflectivity"
    vv.colorByArrayName(sensor, arrayName)
    
    
def main():
    openSensor()
    capture()
main()