import sys, os, glob
import cv2 #openCV
import exiftool
import numpy as np
import pyzbar.pyzbar as pyzbar
import matplotlib.pyplot as plt
import mapboxgl
from micasense.image import Image
from micasense.panel import Panel
import micasense.plotutils as plotutils
import micasense.utils as msutils
import micasense.metadata as metadata

sys.path.append('C:/Users/Isaac Miller/Documents/GitHub/imageprocessing')

exiftoolPath = None
if os.name == 'nt':
    exiftoolPath = 'C:/exiftool/exiftool.exe'

#  get calibration
panelPath = os.path.join('.', 'data','0000SET', '000')
panelName = glob.glob(os.path.join(panelPath, 'IMG_0000_1.tif'))[0]

panelRaw = plt.imread(panelName)
panelMeta = metadata.Metadata(panelName, exiftoolPath=exiftoolPath)
radianceImage, L, V, R = msutils.raw_image_to_radiance(panelMeta, panelRaw)
plotutils.plotwithcolorbar(V,'Vignette Factor')
plotutils.plotwithcolorbar(R,'Row Gradient Factor')
plotutils.plotwithcolorbar(V*R,'Combined Corrections')
plotutils.plotwithcolorbar(L,'Vignette and row gradient corrected raw values')
plotutils.plotwithcolorbar(radianceImage,'All factors applied and scaled to radiance')
markedImg = radianceImage.copy()
ulx = 660  # upper left column (x coordinate) of panel area
uly = 490  # upper left row (y coordinate) of panel area
lrx = 840  # lower right column (x coordinate) of panel area
lry = 670  # lower right row (y coordinate) of panel area
cv2.rectangle(markedImg,(ulx,uly),(lrx,lry),(0,255,0),3)

# Our panel calibration by band (from MicaSense for our specific panel)
panelCalibration = {
    "Blue": 0.67,
    "Green": 0.69,
    "Red": 0.68,
    "Red edge": 0.67,
    "NIR": 0.61
}

# Select panel region from radiance image
panelRegion = radianceImage[uly:lry, ulx:lrx]
plotutils.plotwithcolorbar(markedImg, 'Panel region in radiance image')
meanRadiance = panelRegion.mean()
print('Mean Radiance in panel region: {:1.3f} W/m^2/nm/sr'.format(meanRadiance))
panelReflectance = panelCalibration[panelMeta.get_item('XMP:BandName')]
radianceToReflectance = panelReflectance / meanRadiance
print('Radiance to reflectance conversion factor: {:1.3f}'.format(radianceToReflectance))

reflectanceImage = radianceImage * radianceToReflectance
plotutils.plotwithcolorbar(reflectanceImage, 'Converted Reflectane Image');
panelRegionRaw = panelRaw[uly:lry, ulx:lrx]
panelRegionRefl = reflectanceImage[uly:lry, ulx:lrx]
panelRegionReflBlur = cv2.GaussianBlur(panelRegionRefl,(55,55),5)
plotutils.plotwithcolorbar(panelRegionReflBlur, 'Smoothed panel region in reflectance image')
print('Min Reflectance in panel region: {:1.2f}'.format(panelRegionRefl.min()))
print('Max Reflectance in panel region: {:1.2f}'.format(panelRegionRefl.max()))
print('Mean Reflectance in panel region: {:1.2f}'.format(panelRegionRefl.mean()))
print('Standard deviation in region: {:1.4f}'.format(panelRegionRefl.std()))



imagePath = os.path.join('images', 'test','0003SET', '000')
imageName = glob.glob(os.path.join(imagePath, 'IMG_0020_1.tif'))[0]
#image = Image(imageName)
meta = metadata.Metadata(imageName, exiftoolPath=exiftoolPath)

flightImageRaw=plt.imread(imageName)
plotutils.plotwithcolorbar(flightImageRaw, 'Raw Image')

flightRadianceImage, _, _, _ = msutils.raw_image_to_radiance(meta, flightImageRaw)
flightReflectanceImage = flightRadianceImage * radianceToReflectance
flightUndistortedReflectance = msutils.correct_lens_distortion(meta, flightReflectanceImage)
plotutils.plotwithcolorbar(flightUndistortedReflectance, 'Reflectance converted and undistorted image');