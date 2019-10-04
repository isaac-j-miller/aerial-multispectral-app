import gdal
import numpy as np
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib.colorbar as cbar
import matplotlib.pyplot as plt
import os
from datetime import datetime as dt


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


def gen_thermal(bands):
    dat = bands['lwir']
    dat = np.ma.masked_where(dat == 0, dat)
    return dat/100 - 273.15


def adjust(arr):
    return (arr + 1.0)/2.0


def normalize(arr):
    mn = np.min(arr)
    mx = np.max(arr)
    return (arr - mn)/(mx-mn), mn, mx


equations = {
    'ndvi': gen_ndvi,
    'ndre': gen_ndre,
    'thermal': gen_thermal
}

colormaps = {
    'ndvi': 'Spectral',
    'ndre': 'Spectral',
    'thermal': 'CMRmap',
    'dsm': 'terrain'
}

bandNames = ['blue', 'green', 'red', 'nir', 'red_edge', 'lwir']


def split_stack(filename):  # untested, but shouldn't be necessary
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


def generate_from_separate(files, indexlist, outputpath, outputbase):  # functional, but shouldn't be necessary
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
        print('beginning colormap stuff...')
        v = cm.get_cmap(colormaps[index], 256)

        masked = np.ma.masked_invalid(indexdata)
        print('minmax:', np.min(masked), np.max(masked))
        adj, mn, mx = normalize(masked)
        c = v(adj)
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


def generate_from_stack(file, indexlist, outputpath, outputbase, colormap=True, units='F'):  # units can be F or C. units only used for thermal
    start_time = dt.now()
    print('stack analysis started at ', start_time)
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
        if index == 'thermal':
            if units == 'F':
                indexdata = indexdata*9 / 5 + 32
                label = 'Temperature (' + u'\N{DEGREE SIGN}' + 'F)'
            else:
                label = 'Temperature (' + u'\N{DEGREE SIGN}' + 'C)'
        else:
            label = index.upper()

        masked = np.ma.masked_invalid(indexdata)
        print(masked.shape)
        outputname = os.path.join(outputpath, outputbase + '_' + index + '.tif')
        if colormap:
            print('beginning colormap stuff...')
            v = cm.get_cmap(colormaps[index], 256)
            print('minmax:', np.min(masked), np.max(masked))
            adj, mn, mx = normalize(masked)
            c = v(adj)
            print(c.shape)
            c = np.transpose(c, (2, 0, 1))
            print(c.shape)
            c *= 255
            c = c.astype(int)
            print(np.min(c[0]), np.max(c[0]), np.mean(c[0]))
            print('minmax:', mn, mx)
            norm = colors.Normalize(vmin=mn,vmax=mx)

            fig, ax = plt.subplots(figsize=(1, 6), constrained_layout=True)
            plt.close()
            cb = cbar.ColorbarBase(ax, v, norm)

            cb.set_label(label=label, rotation=90)

            scalename = os.path.join(outputpath, outputbase + '_'+index+'_scale.png')
            fig.savefig(scalename)
            print('ending colormap stuff...')

            names.append([outputname, scalename])
            options = ['PHOTOMETRIC=RGB', 'PROFILE=GeoTIFF']
            out = driver.Create(outputname, cols, rows, 4, gdal.GDT_Byte, options=options)
            out.SetGeoTransform(trans)
            out.SetProjection(projection)
            for band, i in zip(c, range(1, 5)):
                out.GetRasterBand(i).WriteArray(band)
            print('saving...')
            out.FlushCache()
        else:
            names.append([outputname, None])
            options = ['PROFILE=GeoTIFF']
            out = driver.Create(outputname, cols, rows, 1, gdal.GDT_Float32, options=options)
            out.GetRasterBand(1).WriteArray(masked.filled(-10000))
            print('saving...')
            out.FlushCache()


        del out
        end_time = dt.now()
        print('stack analysis ended at ', end_time)
        print('total time to analyze:', end_time - start_time)
    return names


def gen_rgb(file, outputpath, outputbase):
    start_time = dt.now()
    print('stack analysis started at ', start_time)
    base = gdal.Open(file)
    trans = base.GetGeoTransform()
    projection = base.GetProjection()
    data = base.ReadAsArray()[:3][::-1]
    rows, cols = data[0].shape
    driver = gdal.GetDriverByName('GTiff')

    outputname = os.path.join(outputpath, outputbase + '_rgb.tif')

    options = ['PHOTOMETRIC=RGB', 'PROFILE=GeoTIFF']
    out = driver.Create(outputname, cols, rows, 3, gdal.GDT_Byte, options=options)
    out.SetGeoTransform(trans)
    out.SetProjection(projection)
    for band, i in zip(data, range(1, 4)):
        band = np.asarray(band, dtype='float32')*255.0/16934.0
        out.GetRasterBand(i).WriteArray(band)
    print('saving...')
    out.FlushCache()

    del out
    end_time = dt.now()
    print('stack analysis ended at ', end_time)
    print('total time to analyze:', end_time - start_time)
    return outputname


def colormap_dsm(file, outputpath, outputbase, colormap=colormaps['dsm'], units='m'):  # units can be 'ft' or 'm'

    base = gdal.Open(file)
    trans = base.GetGeoTransform()
    projection = base.GetProjection()
    data = base.ReadAsArray()
    data = np.array(data, dtype='float32')
    masked = np.ma.masked_where(data < -2000, data)
    if units == 'ft':
        masked *= 3.2808
        label = 'Elevation (ft)'
    else:
        label = 'Elevation (m)'
    rows, cols = data.shape
    driver = gdal.GetDriverByName('GTiff')
    try:
        v = cm.get_cmap(colormap, 256)
    except ValueError:
        print('invalid colormap. using "terrain" instead')
        colormap = colormaps['dsm']
        v = cm.get_cmap(colormap, 256)

    print('minmax:', np.min(masked), np.max(masked))
    adj, mn, mx = normalize(masked)
    c = v(adj)
    print(c.shape)
    c = np.transpose(c, (2, 0, 1))
    print(c.shape)
    c *= 255
    c = c.astype(int)
    print(np.min(c[0]), np.max(c[0]), np.mean(c[0]))

    print('minmax:', mn, mx)
    norm = colors.Normalize(vmin=mn, vmax=mx)

    fig, ax = plt.subplots(figsize=(1, 6), constrained_layout=True)
    plt.close()
    cb = cbar.ColorbarBase(ax, v, norm)
    cb.set_label(label=label, rotation=90)
    outputname = os.path.join(outputpath, outputbase + '_dsm.tif')
    scalename = os.path.join(outputpath, outputbase + '_dsm_scale.png')
    fig.savefig(scalename)
    print('ending colormap stuff...')

    names = [outputname, scalename]
    options = ['PHOTOMETRIC=RGB', 'PROFILE=GeoTIFF']
    out = driver.Create(outputname, cols, rows, 4, gdal.GDT_Byte, options=options)
    out.SetGeoTransform(trans)
    out.SetProjection(projection)
    for band, i in zip(c, range(1, 5)):
        out.GetRasterBand(i).WriteArray(band)
    out.FlushCache()
    return names


if __name__ == '__main__':
    #fnames = generate_from_stack('test_ortho.tif', ['thermal'], os.getcwd(), 'test_colored', units='F')
    #fnames2 = generate_from_stack('test_ortho.tif', ['thermal'], os.getcwd(), 'test_no_color', colormap=False)
    rgb = gen_rgb('test_ortho.tif', os.getcwd(),'test')
    #dsmname = colormap_dsm('test_dem.tif', os.getcwd(), 'test1')
    #print(fnames, dsmname)
