from ImageTools import stitchSet as ss
from ImageTools import index_generator as ig
from ImageTools import metashape_stitcher as ms
from datetime import datetime as dt
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    t = dt.now()
    t2 = t
    print('stitching images...')
    imagePath = os.path.expanduser(os.path.join('C:/Users/Isaac Miller/Documents/GitHub/aerial-multispectral-app2/Blenheim/raw','0014SET'))
    cams = ['blue', 'green', 'red', 'nir', 'red_edge', 'lwir']
    outputs = ms.stitch(imagePath, cams, 'blenheim_oct_10_19_flight_1', 'Blenheim', calib_csv=ms.CALIB_CSV)
    t = elapsed('image stitching',t)
    
    print('resizing images...')
    ortho_resized = ig.resize_tif('Blenheim/blenheim_oct_10_19_flight_1_ortho.tif', 'Blenheim','blenheim_oct_10_19_1_ortho_resized', 0.25, False)
    t = elapsed('image resizing',t)
    
    print('loading files...')
    stitched = ss.StitchSet('Blenheim/blenheim_ortho_resized.tif','Blenheim/blenheim_oct_10_19_dem.tif',
                            output_directory='Blenheim/tests/10-10-19', output_base='flight_1')
    t = elapsed('loading files',t)
    
    print('setting colormaps...')
    stitched.setColormaps(ig.colormaps)  #unnecessary, just a test
    t = elapsed('colormaps',t)
    
    print('calculating indices...')
    stitched.generateIndices(ig.colormaps.keys())
    t = elapsed('index calculation',t)
    
    print('exporting color images...')
    colors = stitched.exportGeneratedIndicesAsColorImages()
    t = elapsed('color image export',t)
    
    print('exporting mono images...')
    monos = stitched.exportGeneratedIndicesAsMonoImages()
    t = elapsed('mono image export',t)
    
    print('exporting rgba image...')
    rgba = stitched.exportRGBAImage()
    t = elapsed('rgba image export',t)
    
    elapsed('full process',t2)
    
    print('color image files: ')
    print([os.path.split(c[0])[-1] for c in colors])

    print('mono image files: ')
    print([os.path.split(m)[-1] for m in monos])

    print('rgba image file:', os.path.split(rgba)[-1])

