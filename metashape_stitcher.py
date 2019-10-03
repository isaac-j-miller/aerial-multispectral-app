import Metashape, os, glob, math
LICENSE = 'TXC3V-LUVCT-E1BLK-U83UR-GP25H'  # 30-day temporary license. Will expire Nov. 1, 2019.
PATH = 't_project.psx'


# the following 2 functions are courtesy of https://github.com/agisoft-llc/metashape-scripts/blob/master/src/quick_layout.py
def get_photos_delta(chunk):
    mid_idx = int(len(chunk.cameras) / 2)
    if mid_idx == 0:
        return Metashape.Vector([0, 0, 0])
    c1 = chunk.cameras[:mid_idx][-1]
    c2 = chunk.cameras[:mid_idx][-2]
    print(c1.reference.location)
    print(c2.reference.location)
    offset = c1.reference.location - c2.reference.location
    for i in range(len(offset)):
        offset[i] = math.fabs(offset[i])
    return offset


def get_chunk_bounds(chunk):
    min_latitude = min(c.reference.location[1] for c in chunk.cameras if c.reference.location is not None)
    max_latitude = max(c.reference.location[1] for c in chunk.cameras if c.reference.location is not None)
    min_longitude = min(c.reference.location[0] for c in chunk.cameras if c.reference.location is not None)
    max_longitude = max(c.reference.location[0] for c in chunk.cameras if c.reference.location is not None)
    offset = get_photos_delta(chunk)
    offset_factor = 2
    delta_latitude = offset_factor * offset.y
    delta_longitude = offset_factor * offset.x

    min_longitude -= delta_longitude
    max_longitude += delta_longitude
    min_latitude -= delta_latitude
    max_latitude += delta_latitude

    return min_latitude, min_longitude, max_latitude, max_longitude


imagePath = os.path.expanduser(os.path.join('0003SET'))
blue = glob.glob(os.path.join(imagePath,'000','IMG_****_1.tif'))
green = glob.glob(os.path.join(imagePath,'000','IMG_****_2.tif'))
red = glob.glob(os.path.join(imagePath,'000','IMG_****_3.tif'))
nir = glob.glob(os.path.join(imagePath,'000','IMG_****_4.tif'))
rededge = glob.glob(os.path.join(imagePath,'000','IMG_****_5.tif'))
#lwir = glob.glob(os.path.join(imagePath,'000','IMG_****_6.tif'))
cameras = ['blue', 'green', 'red', 'nir', 'red_edge']#, 'lwir']

app = Metashape.Application()
if not app.activated:
    Metashape.License().activate(license_key=LICENSE)
print('active:',app.activated)
doc = Metashape.Document()
doc.save(PATH)

doc.read_only = False
gpus = app.enumGPUDevices()
s = 0
for i in range(len(gpus)):
    s += 2**i
app.gpu_mask = s  # enable all GPUs for max performance
mosaics = dict()
print(doc)
chunk = doc.addChunk()
images = list(zip(blue, green, red, nir, rededge))#, lwir))
print('adding photos...')
chunk.addPhotos(images, Metashape.MultiplaneLayout)
print('matching photos...')
chunk.matchPhotos(accuracy=Metashape.HighAccuracy, generic_preselection=True,
                  reference_preselection=False)
print('aligning cameras...')
chunk.alignCameras()
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
chunk.exportDem('dem.tif',
                format=Metashape.RasterFormatTiles,
                image_format=Metashape.ImageFormatTIFF,
                raster_transform=Metashape.RasterTransformNone,
                projection=chunk.crs,
                tiff_big=True)

print('building orthomosaic...')
chunk.buildOrthomosaic(surface=Metashape.ModelData,

                       blending=Metashape.MosaicBlending)
print('exporting orthomosaic...')
chunk.exportOrthomosaic('ortho.tif',
                        format=Metashape.RasterFormatTiles,
                        image_format=Metashape.ImageFormatTIFF,
                        projection=chunk.crs,
                        raster_transform=Metashape.RasterTransformNone,
                        tiff_big=True,
                        white_background=False)
