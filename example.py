from ImageTools import stitchSet as ss
from ImageTools import index_generator as ig
from datetime import datetime as dt
import matplotlib.pyplot as plt
import os

def elapsed(text, t):
    now = dt.now()
    delta = now - t
    print('{} done in {} seconds'.format(text, delta.total_seconds()))
    return now

if __name__ == '__main__':
    #stitch images using metashape_stitcher
    #imagePath = os.path.expanduser(os.path.join('D:/','0006SET'))
    #cams = ['blue', 'green', 'red', 'nir', 'red_edge', 'lwir']
    #outputs = ms.stitch(imagePath, cams, 'blenheim_oct_6_19', 'Blenheim', calib_csv=CALIB_CSV)
    
    #resize orthomosaic output
    #ortho_resized = ig.resize_tif('Blenheim/blenheim_oct_6_19_ortho.tif', 'Blenheim','blenheim_ortho_resized', 0.25, False)
    #generate rgb and grayscale images
    #a = ig.test_all('Blenheim/blenheim_ortho_resized.tif','Blenheim/Blenheim_test_pre_resized')
    t = dt.now()
    t2 = t
    print('loading files...')
    stitched = ss.StitchSet('Blenheim/blenheim_ortho_resized.tif','Blenheim/blenheim_oct_6_19_dem.tif',
                            output_directory='Blenheim/tests', output_base='objtest')
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
    
    elapsed('full process',t2)
    
    print('color image files: ')
    print([os.path.split(c[0])[-1] for c in colors])

    print('mono image files: ')
    print([os.path.split(m)[-1] for m in monos])

