import Metashape, os, glob
LICENSE = 'TXC3V-LUVCT-E1BLK-U83UR-GP25H'  # 30-day temporary license. Will expire Nov. 1, 2019.
CALIB_CSV = 'RP04-1908203-SC.csv'  # Calibration data for Aerial Multispectral Imagery panel
TEST_CALIB_CSV = 'RP04-1808099-SC.csv'  # Cali
PATH = 't_project.psx'


def stitch(main_dir, available_bands, basename, output_dir, license=LICENSE, temp_path=PATH,calib_csv=None):
    bands = []
    for i in range(len(available_bands)):
        print('finding images for', available_bands[i], 'band...')
        bands.append(glob.glob(os.path.join(main_dir, '***', 'IMG_****_'+str(i+1)+'.tif')))
        print(bands[-1])
    app = Metashape.Application()
    if not app.activated:
        Metashape.License().activate(license_key=license)
    if not app.activated:
        print('invalid or expired license')
        raise PermissionError(
            'Invalid or expired license. Please call the function with license= a valid Metashape license')

    doc = Metashape.Document()
    doc.save(temp_path)
    doc.read_only = False
    gpus = app.enumGPUDevices()  # list of available GPUS
    s = sum(2**i for i in range(len(gpus)))
    app.gpu_mask = s  # enable all GPUs for max performance
    chunk = doc.addChunk()

    images = list(zip(*bands))
    print('adding photos...')
    chunk.addPhotos(images, Metashape.MultiplaneLayout)
    print('matching photos...')
    chunk.matchPhotos(accuracy=Metashape.HighAccuracy, generic_preselection=True,
                      reference_preselection=False)

    if calib_csv is not None:  # if there is the necessary calibration data
        print('locating reflectance panels...')
        chunk.locateReflectancePanels()
        print('loading calibration data...')
        chunk.loadReflectancePanelCalibration(calib_csv)
        print('calibrating reflectance...')
        chunk.calibrateReflectance()

    print('aligning cameras...')
    chunk.alignCameras()
    print('optimizing cameras...')
    chunk.optimizeCameras()
    print('building depth maps...')
    chunk.buildDepthMaps(quality=Metashape.MediumQuality, filter=Metashape.AggressiveFiltering)
    print('building dense cloud...')
    chunk.buildDenseCloud()
    print('building model...')
    chunk.buildModel(surface=Metashape.Arbitrary, interpolation=Metashape.EnabledInterpolation)
    print('building UV...')
    chunk.buildUV(mapping=Metashape.GenericMapping)
    print('building texture...')
    chunk.buildTexture(blending=Metashape.MosaicBlending, size=4096)
    print('saving...')
    doc.save()
    print('building dem...')
    chunk.buildDem(source=Metashape.DenseCloudData, interpolation=Metashape.EnabledInterpolation)
    print('exporting dem...')
    demname = os.path.join(output_dir, basename+'_dem.tif')
    chunk.exportDem(demname,
                    format=Metashape.RasterFormatTiles,
                    image_format=Metashape.ImageFormatTIFF,
                    raster_transform=Metashape.RasterTransformNone,
                    projection=chunk.crs,
                    tiff_big=True)

    print('building orthomosaic...')
    chunk.buildOrthomosaic(surface=Metashape.ModelData,

                           blending=Metashape.MosaicBlending)
    print('exporting orthomosaic...')
    orthoname = os.path.join(output_dir, basename+'_ortho.tif')
    chunk.exportOrthomosaic(orthoname,
                            format=Metashape.RasterFormatTiles,
                            image_format=Metashape.ImageFormatTIFF,
                            projection=chunk.crs,
                            raster_transform=Metashape.RasterTransformNone,
                            tiff_big=True,
                            white_background=False)
    names = [demname, orthoname]
    app.quit()
    return names


if __name__ == '__main__':
    imagePath = os.path.expanduser(os.path.join('altum_example','0000SET'))
    cams = ['blue', 'green', 'red', 'nir', 'red_edge', 'lwir']
    outputs = stitch(imagePath, cams, 'test', os.getcwd(), calib_csv=TEST_CALIB_CSV)
    print(outputs)
