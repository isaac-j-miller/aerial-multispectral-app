import Metashape, os, glob

imagePath = os.path.expanduser(os.path.join('~', 'Downloads', 'altum_example', '0000SET'))
blue = glob.glob(os.path.join(imagePath,'000','IMG_****_1.tif'))
green = glob.glob(os.path.join(imagePath,'000','IMG_****_2.tif'))
red = glob.glob(os.path.join(imagePath,'000','IMG_****_3.tif'))
nir = glob.glob(os.path.join(imagePath,'000','IMG_****_4.tif'))
rededge = glob.glob(os.path.join(imagePath,'000','IMG_****_5.tif'))
lwir = glob.glob(os.path.join(imagePath,'000','IMG_****_6.tif'))
cameras = ['blue', 'green', 'red', 'nir', 'red_edge', 'lwir']

doc = Metashape.Document()
mosaics = dict()
print(doc)
chunk = doc.addChunk()
images = list(zip(blue, green, red, nir, rededge, lwir))
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
print('building orthomosaics...')
for camera, i in zip(cameras, range(len(cameras))):
    print('building orthomosaic for', camera, 'band')
    o = chunk.addOthomosaic()
    c = chunk.cameras[i]
    for shape in chunk.shapes:
        p = Metashape.Orthomosaic.Patch()
        p.image_keys = [c.key]
        o.patches[shape] = p
        o.update()
    mosaics[camera] = o.key
    print('key:', o.key, 'bands:', o.bands)
