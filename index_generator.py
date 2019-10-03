import gdal
import numpy as np
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib.colorbar as cbar
import matplotlib.pyplot as plt
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


def gen_lwir(bands):
    return bands['lwir']


def adjust(arr):
    return (arr + 1.0)/2.0


def normalize(arr):
    mn = np.min(arr)
    mx = np.max(arr)
    return (arr - mn)/(mx-mn),mn,mx


equations = {
    'ndvi': gen_ndvi,
    'ndre': gen_ndre
}

colormaps = {
    'ndvi': 'rainbow_r',
    'ndre': 'Spectral'
}

bandNames = ['blue', 'green', 'red', 'nir', 'red_edge']


def split_stack(filename):
    base = gdal.Open(filename)
    trans = base.GetGeoTransform()
    projection = base.GetProjection()
    data = base.ReadAsArray()
    rows, cols = data[0].shape
    options = ['PROFILE=GeoTIFF']
    driver = gdal.GetDriverByName('GTiff')
    names = []
    for band, num in zip(data, range(data.shape[0])):
        outputname = filename[:filename.index('.tif')]+'_'+str(num)+'.tif'
        names.append(outputname)
        out = driver.Create(outputname, cols, rows, 1, gdal.GDT_Byte, options=options)
        out.SetGeoTransform(trans)
        out.SetProjection(projection)
        out.GetRasterBand(1).WriteArray(band)
        print('saving...')
        out.FlushCache()
        del out
    return names


def generate_from_separate(files, indexlist, outputpath, outputbase):
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
        temp = tif.ReadAsArray()
        temp = np.array(temp, dtype='float32')
        temp = np.ma.masked_where(temp == -10000, temp)
        data.append(temp)
        print(np.min(data[-1]), np.max(data[-1]),np.mean(data[-1]))
        print('filename:', file)

        print('shape:', data[-1].shape)
        rows, cols = temp.shape
        del tif
    bands = dict()

    for band, i in zip(bandNames, range(len(bandNames))):
        bands[band] = data[i]

    for index in indexlist:
        indexdata = equations[index](bands)
        #  colormap testing
        print('beginning colormap stuff...')
        v = cm.get_cmap(colormaps[index], 256)

        masked = np.ma.masked_invalid(indexdata)
        print('minmax:', np.min(masked), np.max(masked))
        adj, mn, mx = normalize(masked)
        c = v(masked)
        print(c.shape)
        c = np.transpose(c, (2, 0, 1))
        print(c.shape)
        c *= 255
        c = c.astype(int)
        print(np.min(c[0]), np.max(c[0]), np.mean(c[0]))
        print('minmax:',mn,mx)
        norm = colors.Normalize(vmin=mn,vmax=mx)
        fig, ax = plt.subplots(figsize=(1, 6),constrained_layout=True)
        cb = cbar.ColorbarBase(ax, v, norm)
        cb.set_label(index.upper(), rotation=90)

        scalename = os.path.join(outputpath, outputbase + '_'+index+'_scale.png')
        fig.savefig(scalename)
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


def generate_from_one(file, indexlist, outputpath, outputbase):
    base = gdal.Open(file)
    trans = base.GetGeoTransform()
    projection = base.GetProjection()
    data = base.ReadAsArray()
    rows,cols = data[0].shape
    names = []
    bands = dict()
    driver = gdal.GetDriverByName('GTiff')
    for band, i in zip(bandNames, range(len(bandNames))):
        temp = data[i]
        temp = np.array(temp, dtype='float32')
        temp = np.ma.masked_where(temp == -10000, temp)
        bands[band] = temp
        del temp

    for index in indexlist:
        indexdata = equations[index](bands)
        #  colormap testing
        print('beginning colormap stuff...')
        v = cm.get_cmap(colormaps[index], 256)

        masked = np.ma.masked_invalid(indexdata)
        print('minmax:', np.min(masked), np.max(masked))
        adj, mn, mx = normalize(masked)
        c = v(masked)
        print(c.shape)
        c = np.transpose(c, (2, 0, 1))
        print(c.shape)
        c *= 255
        c = c.astype(int)
        print(np.min(c[0]), np.max(c[0]), np.mean(c[0]))
        print('minmax:',mn,mx)
        norm = colors.Normalize(vmin=mn,vmax=mx)
        fig, ax = plt.subplots(figsize=(1, 6),constrained_layout=True)
        cb = cbar.ColorbarBase(ax, v, norm)
        cb.set_label(index.upper(), rotation=90)

        scalename = os.path.join(outputpath, outputbase + '_'+index+'_scale.png')
        fig.savefig(scalename)
        print('ending colormap stuff...')
        #  end test
        outputname = os.path.join(outputpath, outputbase + '_'+index+'.tif')
        names.append(outputname)
        options = ['PHOTOMETRIC=RGB', 'PROFILE=GeoTIFF']
        out = driver.Create(outputname, cols, rows, 4, gdal.GDT_Byte, options=options)
        out.SetGeoTransform(trans)
        out.SetProjection(projection)
        for band, i in zip(c, range(1,5)):
            out.GetRasterBand(i).WriteArray(band)
        print('saving...')
        out.FlushCache()
        del out
    return names

if __name__ == '__main__':
    #files = ['test1_transparent_mosaic_blue.tif',
    #         'test1_transparent_mosaic_green.tif',
    #         'test1_transparent_mosaic_red.tif',
    #        'test1_transparent_mosaic_nir.tif',
    #        'test1_transparent_mosaic_red edge.tif']
    files = ['blenheim_test_index_blue.tif',
             'blenheim_test_index_green.tif',
             'blenheim_test_index_red.tif',
             'blenheim_test_index_nir.tif',
             'blenheim_test_index_red_edge.tif']

    #fnames = generate_from_separate(files, ['ndvi', 'ndre'],'','test')
    fnames = generate_from_one('ortho.tif', ['ndvi', 'ndre'],'','test')
    d = gdal.Open(fnames[0])
    print(d.GetProjection())
    print(d.GetGeoTransform())
