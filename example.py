from ImageTools import metashape_stitcher as ms
from ImageTools import index_generator as ig

if __name__ == '__main__':
    #stitch images using metashape_stitcher
    imagePath = os.path.expanduser(os.path.join('D:/','0006SET'))
    cams = ['blue', 'green', 'red', 'nir', 'red_edge', 'lwir']
    outputs = ms.stitch(imagePath, cams, 'blenheim_oct_6_19', 'Blenheim', calib_csv=CALIB_CSV)
    
    #resize orthomosaic output
    ortho_resized = ig.resize_tif('Blenheim/blenheim_oct_6_19_ortho.tif', 'Blenheim','blenheim_ortho_resized', 0.25, False)
    #generate rgb and grayscale images
    a = ig.test_all('Blenheim/blenheim_ortho_resized.tif','Blenheim/Blenheim_test_pre_resized')

