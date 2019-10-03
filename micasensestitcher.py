import sys, os, glob
import micasense.imageset as imageset
import datetime
import subprocess
from mapboxgl.viz import *
from mapboxgl.utils import df_to_geojson, create_radius_stops, scale_between
from mapboxgl.utils import create_color_stops
import jenkspy
import pandas as pd
from osgeo import *
import rasterio
import tifffile

panelNames = None
useDLS = True
warp_matrices = None

sys.path.append('C:/Users/Isaac Miller/Documents/GitHub/imageprocessing')
MERGEPATH = 'C:/Users/Isaac Miller/AppData/Local/Programs/Python/Python37/Lib/site-packages/osgeo/scripts'

def decdeg2dms(dd):
    # this function is from https://github.com/micasense/imageprocessing/blob/master/Alignment.ipynb
   is_positive = dd >= 0
   dd = abs(dd)
   minutes,seconds = divmod(dd*3600,60)
   degrees,minutes = divmod(minutes,60)
   degrees = degrees if is_positive else -degrees
   return (degrees,minutes,seconds)


def alignAndSave(dir):
    # this function is adapted from https://github.com/micasense/imageprocessing/blob/master/Alignment.ipynb
    import micasense.capture as capture
    imagePath = dir
    panelNames = glob.glob(os.path.join(imagePath,'000','IMG_0000_*.tif'))
    panelCap = capture.Capture.from_filelist(panelNames)

    outputPath = os.path.join(imagePath, '..', 'stacks')
    thumbnailPath = os.path.join(outputPath, '..', 'thumbnails')
    print(imagePath,panelNames, outputPath, thumbnailPath)
    overwrite = False # usefult to set to false to continue interrupted processing
    generateThumbnails = False

    # Allow this code to align both radiance and reflectance images; bu excluding
    # a definition for panelNames above, radiance images will be used
    # For panel images, efforts will be made to automatically extract the panel information
    # but if the panel/firmware is before Altum 1.3.5, RedEdge 5.1.7 the panel reflectance
    # will need to be set in the panel_reflectance_by_band variable.
    # Note: radiance images will not be used to properly create NDVI/NDRE images below.
    if panelNames is not None:
        panelCap = capture.Capture.from_filelist(panelNames)
    else:
        panelCap = None

    if panelCap is not None:
        if panelCap.panel_albedo() is not None:
            panel_reflectance_by_band = panelCap.panel_albedo()
        else:
            panel_reflectance_by_band = [0.67, 0.69, 0.68, 0.61, 0.67] #RedEdge band_index order
        panel_irradiance = panelCap.panel_irradiance(panel_reflectance_by_band)
        img_type = "reflectance"
    else:
        if useDLS:
            img_type='reflectance'
        else:
            img_type = "radiance"
    imgset = imageset.ImageSet.from_directory(imagePath)
    data, columns = imgset.as_nested_lists()
    df = pd.DataFrame.from_records(data, index='timestamp', columns=columns)

    #Insert your mapbox token here
    token = 'pk.eyJ1IjoiaXNhYWNqbWlsbGVyIiwiYSI6ImNrMTZ6NnBqdjFiM3czcHRrb3ZtbTZsajYifQ.3tv9y_9KCHST0M5NaDj4Zg'

    color_property = 'dls-yaw'
    num_color_classes = 8

    min_val = df[color_property].min()
    max_val = df[color_property].max()


    breaks = jenkspy.jenks_breaks(df[color_property], nb_class=num_color_classes)

    color_stops = create_color_stops(breaks,colors='YlOrRd')
    geojson_data = df_to_geojson(df,columns[3:],lat='latitude',lon='longitude')

    viz = CircleViz(geojson_data, access_token=token, color_property=color_property,
                    color_stops=color_stops,
                    center=[df['longitude'].median(),df['latitude'].median()],
                    zoom=16, height='600px',
                    style='mapbox://styles/mapbox/satellite-streets-v9')
    viz.show()

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    if generateThumbnails and not os.path.exists(thumbnailPath):
        os.makedirs(thumbnailPath)

    # Save out geojson data so we can open the image capture locations in our GIS
    with open(os.path.join(outputPath, 'imageSet.json'), 'w') as f:
        f.write(str(geojson_data))

    try:
        irradiance = panel_irradiance + [0]
    except NameError:
        irradiance = None

    start = datetime.datetime.now()
    for i, capture in enumerate(imgset.captures):
        outputFilename = capture.uuid + '.tif'
        thumbnailFilename = capture.uuid + '.jpg'
        fullOutputPath = os.path.join(outputPath, outputFilename)
        fullThumbnailPath = os.path.join(thumbnailPath, thumbnailFilename)
        if (not os.path.exists(fullOutputPath)) or overwrite:
            if (len(capture.images) == len(imgset.captures[0].images)):
                capture.create_aligned_capture(irradiance_list=irradiance, warp_matrices=warp_matrices)
                capture.save_capture_as_stack(fullOutputPath)
                if generateThumbnails:
                    capture.save_capture_as_rgb(fullThumbnailPath)
        capture.clear_image_data()

    end = datetime.datetime.now()

    print("Saving time: {}".format(end - start))
    print("Alignment+Saving rate: {:.2f} images per second".format(
        float(len(imgset.captures)) / float((end - start).total_seconds())))


    header = "SourceFile,\
    GPSDateStamp,GPSTimeStamp,\
    GPSLatitude,GpsLatitudeRef,\
    GPSLongitude,GPSLongitudeRef,\
    GPSAltitude,GPSAltitudeRef,\
    FocalLength,\
    XResolution,YResolution,ResolutionUnits\n"

    lines = [header]
    for capture in imgset.captures:
        #get lat,lon,alt,time
        outputFilename = capture.uuid+'.tif'
        fullOutputPath = os.path.join(outputPath, outputFilename)
        lat,lon,alt = capture.location()
        #write to csv in format:
        # IMG_0199_1.tif,"33 deg 32' 9.73"" N","111 deg 51' 1.41"" W",526 m Above Sea Level
        latdeg, latmin, latsec = decdeg2dms(lat)
        londeg, lonmin, lonsec = decdeg2dms(lon)
        latdir = 'North'
        if latdeg < 0:
            latdeg = -latdeg
            latdir = 'South'
        londir = 'East'
        if londeg < 0:
            londeg = -londeg
            londir = 'West'
        resolution = capture.images[0].focal_plane_resolution_px_per_mm

        linestr = '"{}",'.format(fullOutputPath)
        linestr += capture.utc_time().strftime("%Y:%m:%d,%H:%M:%S,")
        linestr += '"{:d} deg {:d}\' {:.2f}"" {}",{},'.format(int(latdeg),int(latmin),latsec,latdir[0],latdir)
        linestr += '"{:d} deg {:d}\' {:.2f}"" {}",{},{:.1f} m Above Sea Level,Above Sea Level,'.format(int(londeg),int(lonmin),lonsec,londir[0],londir,alt)
        linestr += '{}'.format(capture.images[0].focal_length)
        linestr += '{},{},mm'.format(resolution,resolution)
        linestr += '\n' # when writing in text mode, the write command will convert to os.linesep
        lines.append(linestr)
        with tifffile.TiffFile(fullOutputPath) as t:
            data = t.asarray()
            mdata = {}

    fullCsvPath = os.path.join(outputPath,'log.csv')
    with open(fullCsvPath, 'w') as csvfile: #create CSV
        csvfile.writelines(lines)

    old_dir = os.getcwd()
    os.chdir(outputPath)
    cmd = 'exiftool -csv="{}" -overwrite_original .'.format(fullCsvPath)
    print(cmd)
    print('writing metadata to csv')
    try:
        subprocess.check_call(cmd)
    finally:
        os.chdir(old_dir)

    return outputPath


# stitch with gdal
def stitch(stacksPath, outputPath):
    print(stacksPath)
    files = glob.glob(os.path.join(stacksPath, '*.tif'))
    #files = ['"'+f+'"' for f in files]
    cmd = ['python.exe', os.path.join(MERGEPATH,'gdal_merge.py'), '-o', outputPath, '-of', 'gtiff', *files]
    print(cmd)
    subprocess.call(cmd)


if __name__ == '__main__':
    print('Ready...')
    imagePath = os.path.expanduser(os.path.join('~', 'Downloads', 'altum_example', '0000SET'))
    outputPath = 'mosaic.tif'
    #out = alignAndSave(imagePath)
    out = 'C:/Users/Isaac Miller/Downloads/altum_example/stacks'
    stitch(out, outputPath)
