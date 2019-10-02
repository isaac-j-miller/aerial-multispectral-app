import gdal
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap
import os


def calc_index(numerator, denominator):
    np.seterr(divide='ignore', invalid='ignore')
    a = np.true_divide(numerator, denominator)
    np.seterr(divide='raise', invalid='raise')
    return a


def gen_ndvi(bands):
    numerator = bands['nir'] - bands['red']
    denominator = bands['nir'] + bands['red']
    return calc_index(numerator, denominator)


def gen_ndre(bands):
    numerator = bands['nir'] - bands['red_edge']
    denominator = bands['nir'] + bands['red_edge']
    return calc_index(numerator, denominator)


def adjust(arr):
    return (arr + 1.0)/2.0

equations = {
    'ndvi': gen_ndvi,
    'ndre': gen_ndre
}

colormaps = {
    'ndvi': 'Spectral',
    'ndre': 'Spectral'
}

bandNames = ['blue', 'green', 'red', 'nir', 'red_edge']


def generate_from_stack(files, indexlist, outputpath, outputbase):
    data = []
    names = []
    driver = gdal.GetDriverByName('GTiff')
    projection, geotrans = None, None
    cols, rows = None, None
    for file, i in zip(files, range(len(files))):
        tif = gdal.Open(file)
        geotrans = tif.GetGeoTransform()
        print('transform: ', geotrans)
        projection = tif.GetProjection()
        print('projection: ', projection)
        temp = tif.ReadAsArray()[0]
        temp = np.array(temp, dtype='float32')
        rows, cols = temp.shape
        data.append(temp)
        print(np.min(data[-1]), np.max(data[-1]),np.mean(data[-1]))
        print('filename:', file)
        print('shape:', data[-1].shape)
        del tif
    bands = dict()

    for band, i in zip(bandNames, range(len(bandNames))):
        bands[band] = data[i]

    for index in indexlist:
        indexdata = equations[index](bands)
        #  colormap testing
        print('beginning colormap stuff...')
        v = cm.get_cmap(colormaps[index], 256)
        adj = adjust(indexdata)
        #adj[adj == np.inf] = np.nan
        masked = np.ma.masked_invalid(adj)
        c = v(masked,)
        print(c.shape)
        c = np.transpose(c, (2, 0, 1))
        print(c.shape)
        c *= 255
        c = c.astype(int)
        print(np.min(c[0]), np.max(c[0]), np.mean(c[0]))
        print('ending colormap stuff...')
        #  end test
        outputname = os.path.join(outputpath, outputbase + '_'+index+'.tif')
        names.append(outputname)
        options = ['PHOTOMETRIC=RGB', 'PROFILE=GeoTIFF']
        out = driver.Create(outputname, cols, rows, 4, gdal.GDT_Byte, options=options)
        out.SetGeoTransform(geotrans)
        out.SetProjection(projection)
        for band, i in zip(c, range(1,5)):
            out.GetRasterBand(i).WriteArray(band)
        print('saving...')
        out.FlushCache()
        del out
    return names


if __name__ == '__main__':
    files = ['test1_transparent_mosaic_blue.tif',
             'test1_transparent_mosaic_green.tif',
             'test1_transparent_mosaic_red.tif',
             'test1_transparent_mosaic_nir.tif',
             'test1_transparent_mosaic_red edge.tif']

    fnames = generate_from_stack(files, ['ndvi', 'ndre'],'','test')
    d = gdal.Open(fnames[0])
    print(d.GetProjection())
    print(d.GetGeoTransform())
