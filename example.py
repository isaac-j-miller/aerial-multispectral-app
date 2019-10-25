from ImageTools import stitchSet as ss
from ImageTools import index_generator as ig
from ImageTools import metashape_stitcher as ms
from datetime import datetime as dt
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    t = dt.now()
    t2 = t
    """
    print('stitching images...')
    imagePath = os.path.expanduser(os.path.join('C:/Users/Isaac Miller/Documents/GitHub/aerial-multispectral-app2/Blenheim/raw','0009SET'))
    cams = ['blue', 'green', 'red', 'nir', 'red_edge', 'lwir']
    outputs = ms.stitch(imagePath, cams, 'blenheim_sep_27_19', 'Blenheim', calib_csv=ms.CALIB_CSV)
    t = ig.elapsed('image stitching',t)
    
    print('resizing images...')
    ortho_resized = ig.resize_tif(outputs[1], 'Blenheim','blenheim_sep_27_19_ortho_resized', 0.25, False)
    t = ig.elapsed('image resizing',t)
    """
    print('loading files...')
    stitched = ss.StitchSet('Blenheim/blenheim_sep_27_19_ortho_resized.tif','Blenheim/blenheim_sep_27_19_dsm.tif',
                            output_directory='Blenheim/tests/09-27-19', output_base='')
    t = ig.elapsed('loading files',t)
    """
    print('setting colormaps...')
    stitched.setColormaps(ig.colormaps)  #unnecessary, just a test and example if you want to use different colormaps
    t = ig.elapsed('colormaps',t)
    """
    print('calculating indices...')
    stitched.generateIndices(ig.colormaps.keys())
    t = ig.elapsed('index calculation',t)
    
    print('exporting color images...')
    colors = stitched.exportGeneratedIndicesAsColorImages()
    t = ig.elapsed('color image export',t)
    
    print('exporting mono images...')
    monos = stitched.exportGeneratedIndicesAsMonoImages()
    t = ig.elapsed('mono image export',t)

    print('exporting rgba image...')
    rgba = stitched.exportRGBAImage()
    t = ig.elapsed('rgba image export',t)
    
    ig.elapsed('full process',t2)
    
    print('color image files: ')
    print([os.path.split(c[0])[-1] for c in colors])

    print('mono image files: ')
    print([os.path.split(m)[-1] for m in monos])
    
    print('rgba image file:', os.path.split(rgba)[-1])

