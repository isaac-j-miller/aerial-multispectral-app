import Metashape, os, glob, shutil
from ImageTools import index_generator as ig
from datetime import datetime as dt
LICENSE = 'TXC3V-LUVCT-E1BLK-U83UR-GP25H'  # 30-day temporary license. Will expire Nov. 1, 2019.
CALIB_CSV = 'ImageTools/RP04-1908203-SC.csv'  # Calibration data for Aerial Multispectral Imagery panel
TEST_CALIB_CSV = 'ImageTools/RP04-1808099-SC.csv'  # Calibration data for example altum dataset
PATH = 't_project'


def stitch(main_dir, available_bands, basename, output_dir, license=LICENSE, temp_path=PATH,calib_csv=None):
    """
    stitch: stitches photos and generates an orthomosaic and digital surface model for set of multispectral images.
    Params:
    :param main_dir: string of main source directory location. For micasense sets, this often looks like '0001SET'
    :param available bands: list of bands in order: for altum, ['blue', 'green', 'red', 'nir', 'red_edge', 'lwir']. For rededge: ['blue', 'green', 'red', 'nir', 'red_edge']
    :param basename: base name for outputs. Example: basename = 'test_file', outputs are 'test_file_ortho.tif', 'test_file_dsm.tif'
    :param output_dir: directory to output files into.
    :param license: valid Metashape Pro license (string).
    :param temp_path: temporary path for Metashape project while processing.
    :param calib_csv: location of csv file for specific calibration file.
    returns:
    :return: list of output files as [digital surface model name, stacked orthomosaic name]
    """
    start_time = dt.now()
    t = start_time
    print('stitch started at ', start_time)
    bands = []
    print('finding images...')
    imageCount = 0
    for i in range(len(available_bands)):
        print('finding images for', available_bands[i], 'band...')
        bands.append(glob.glob(os.path.join(main_dir, '***', 'IMG_****_'+str(i+1)+'.tif')))
        imageCount += len(bands[-1])
    print('total images:',imageCount)
    app = Metashape.Application()
    if not app.activated:
        Metashape.License().activate(license_key=license)
    if not app.activated:
        print('invalid or expired license')
        raise PermissionError(
            'Invalid or expired license. Please call the function with license= a valid Metashape license')
    t = ig.elapsed('image search', t)
    doc = Metashape.Document()
    doc.save(temp_path+'.psx')
    doc.read_only = False
    gpus = app.enumGPUDevices()  # list of available GPUS
    s = sum(2**i for i in range(len(gpus)))
    app.gpu_mask = s  # enable all GPUs for max performance
    chunk = doc.addChunk()

    images = list(zip(*bands))
    print('adding photos...')
    chunk.addPhotos(images, Metashape.MultiplaneLayout)
    t = ig.elapsed('add photos', t)
    print('matching photos...')
    chunk.matchPhotos(accuracy=Metashape.HighAccuracy, generic_preselection=True,
                      reference_preselection=False)
    t = ig.elapsed('image matching', t)
    if calib_csv is not None:  # if there is the necessary calibration data
        print('locating reflectance panels...')
        chunk.locateReflectancePanels()
        print('loading calibration data...')
        chunk.loadReflectancePanelCalibration(calib_csv)
        print('calibrating reflectance...')
        chunk.calibrateReflectance()
        t = ig.elapsed('reflectance calibration', t)

    print('aligning cameras...')
    chunk.alignCameras()
    t = ig.elapsed('camera alignment', t)
    doc.save()
    print('optimizing cameras...')
    chunk.optimizeCameras()
    doc.save()
    t = ig.elapsed('camera optimization', t)
    print('building depth maps...')
    chunk.buildDepthMaps(quality=Metashape.MediumQuality, filter=Metashape.AggressiveFiltering)
    t = ig.elapsed('depth map construction', t)
    doc.save()
    print('building dense cloud...')
    chunk.buildDenseCloud()
    doc.save()
    t = ig.elapsed('dense cloud construction', t)
    print('building model...')
    chunk.buildModel(surface=Metashape.Arbitrary, interpolation=Metashape.EnabledInterpolation)
    t = ig.elapsed('model construction', t)
    doc.save()
    print('building UV...')
    chunk.buildUV(mapping=Metashape.GenericMapping)
    t = ig.elapsed('UV construction', t)
    doc.save()
    print('building texture...')
    chunk.buildTexture(blending=Metashape.MosaicBlending, size=4096)
    t = ig.elapsed('texture construction', t)
    print('saving...')
    doc.save()
    print('building dsm...')
    chunk.buildDem(source=Metashape.DenseCloudData, interpolation=Metashape.EnabledInterpolation)
    t = ig.elapsed('dsm constructed', t)
    print('exporting dsm...')
    demname = os.path.join(output_dir, basename+'_dsm.tif')
    chunk.exportDem(demname,
                    format=Metashape.RasterFormatTiles,
                    image_format=Metashape.ImageFormatTIFF,
                    raster_transform=Metashape.RasterTransformNone,
                    projection=chunk.crs,
                    tiff_big=True)
    t = ig.elapsed('dsm export', t)
    print('building orthomosaic...')
    chunk.buildOrthomosaic(surface=Metashape.ModelData, blending=Metashape.MosaicBlending)
    t = ig.elapsed('orthomosaic construction', t)
    doc.save()
    print('exporting orthomosaic...')
    orthoname = os.path.join(output_dir, basename+'_ortho.tif')
    chunk.exportOrthomosaic(orthoname,
                            format=Metashape.RasterFormatTiles,
                            image_format=Metashape.ImageFormatTIFF,
                            projection=chunk.crs,
                            raster_transform=Metashape.RasterTransformNone,
                            tiff_big=True,
                            white_background=False)
    t = ig.elapsed('orthomosaic export', t)
    names = [demname, orthoname]
    doc.clear()
    app.quit()
    end_time = dt.now()
    print('removing extra files...')
    try:
        shutil.rmtree(temp_path+'.files', True)
    except OSError or PermissionError:
        print('permission denied. Please delete', temp_path, '.files')
    t = ig.elapsed('junk removal', t)
    print('stitch ended at ', end_time)
    print('total time to stitch:', end_time-start_time)
    return names



