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


def mask_zeros(bands_):
    vals = list(bands_.values())
    return {k: np.ma.masked_where(sum(vals) == 0, bands_[k]) for k in bands_.keys()}


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


def gen_gndvi(bands):
    numerator = bands['nir'] - bands['green']
    denominator = bands['nir'] + bands['green']
    return calc_index(numerator, denominator)


def gen_endvi(bands):
    numerator = bands['nir'] + bands['green'] - 2 * bands['blue']
    denominator = bands['nir'] + bands['green'] + 2 * bands['blue']
    return calc_index(numerator, denominator)


def gen_savi(bands, L=0.5):  # L should be 0.25 for mid-growth, 0.1 for early growth
    numerator = (bands['nir'] - bands['red'])/16934.0
    denominator = (bands['nir'] + bands['red'])/16934.0 + L
    return mask_extremes(calc_index(numerator, denominator)*(1+L))


def gen_gli(bands):
    numerator = 2 * bands['green'] - (bands['red'] + bands['blue'])
    denominator = 2 * bands['green'] + (bands['red'] + bands['blue'])
    return calc_index(numerator, denominator)


def gen_vari(bands):
    numerator = bands['green'] - bands['red']
    denominator = bands['green'] + bands['red'] - bands['blue']
    return calc_index(numerator, denominator)


def gen_gdi(bands):
    return mask_extremes((bands['nir'] - bands['green'])/16934.0)


def gen_dvi(bands):
    return mask_extremes((bands['nir'] - bands['red'])/16934.0)


def mask_extremes(arr, std_devs=5):  # no longer used, but still maybe useful later
    r = arr.std()*std_devs
    low = arr.mean() - r
    high = arr.mean() + r

    return np.ma.masked_outside(arr, low, high)


def adjust(arr):
    return (arr + 1.0)/2.0


def normalize(arr):
    mn = np.min(arr)
    mx = np.max(arr)
    return (arr - mn)/(mx-mn), mn, mx


equations = {
    'ndvi': gen_ndvi,
    'ndre': gen_ndre,
    'thermal': gen_thermal,
    'gndvi': gen_gndvi,
    'endvi': gen_endvi,
    'savi': gen_savi,
    'gli': gen_gli,
    'vari': gen_vari,
    'gdi': gen_gdi,
    'dvi': gen_dvi
}


ranges = {
    'ndvi': [-1.0,1.0],
    'ndre': [-1.0,1.0],
    'thermal': [-np.inf,np.inf],
    'dsm': [-np.inf,np.inf],
    'gndvi': [-1.0,1.0],
    'endvi': [-1.0,1.0],
    'savi': [-1.0,1.0],
    'gli': [-1.0,1.0],
    'vari': [-1.0,1.0],
    'gdi': [-1.0,1.0],
    'dvi': [-1.0,1.0],
}


colormaps = {
    'ndvi': 'RdYlGn',
    'ndre': 'RdYlGn',
    'thermal': 'CMRmap',
    'dsm': 'terrain',
    'gndvi': 'RdYlGn',
    'endvi': 'RdYlGn',
    'savi': 'RdYlGn',
    'gli': 'RdYlGn',
    'vari': 'RdYlGn',
    'gdi': 'RdYlGn',
    'dvi': 'RdYlGn'
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


def diagnose(file, band=None, mask=None):
    d = gdal.Open(file)
    if band is not None:
        a = d.ReadAsArray()[band]
    else:
        a = d.ReadAsArray()
    if mask is not None:
        a = np.ma.masked_where(a <= mask,a)
    plt.imshow(a)
    return a


def generate_from_stack(file, indexdict, outputpath, outputbase, colormap=True, units='F', L=0.25):  # units can be F or C. units only used for thermal, L only used for SAVI
    try:
        indexlist = indexdict.keys()
    except AttributeError:
        indexlist = indexdict
        indexdict = colormaps
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
    bands = mask_zeros(bands)
    for index in indexlist:
        print('beginning analysis on',index,'...')

        if index in equations.keys():
            if index == 'savi':
                indexdata = equations[index](bands, L)
            else:
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
            #print(masked.shape)
            outputname = os.path.join(outputpath, outputbase + '_' + index + '.tif')
            if colormap:
                print('beginning colormap stuff...')
                try:
                    v = cm.get_cmap(indexdict[index], 256)
                except ValueError:
                    v = cm.get_cmap(colormaps[index], 256)
                    print('invalid colormap. using default from colormaps')
                print('minmax:', np.min(masked), np.max(masked))

                masked = np.ma.masked_outside(masked, ranges[index][0], ranges[index][1])
                adj, mn, mx = normalize(masked)
                #experimental:
                if index != 'thermal':
                    mn, mx = ranges[index]
                print('acceptable minmax:', ranges[index][0], ranges[index][1])
                print('new minmax:', mn, mx)
                c = v(adj)
                #print(c.shape)
                c = np.transpose(c, (2, 0, 1))
                #print(c.shape)
                c *= 255
                c = c.astype(int)

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
        else:
            print('invalid key:',index,'; ignoring...')
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


def colormap_tif(file, outputpath, outputbase, dataname, colormap=colormaps['dsm'], units='m'):  # units can be 'ft' or 'm'
    base = gdal.Open(file)
    trans = base.GetGeoTransform()
    projection = base.GetProjection()
    data = base.ReadAsArray()
    data = np.array(data, dtype='float32')
    masked = np.ma.masked_where(data < -2000, data)
    if dataname == 'dsm':
        if units == 'ft':
            masked *= 3.2808
            label = 'Elevation (ft)'
        else:
            label = 'Elevation (m)'
    else:
        label = dataname.upper()

    rows, cols = data.shape
    driver = gdal.GetDriverByName('GTiff')
    try:
        v = cm.get_cmap(colormap, 256)
    except ValueError:
        print('invalid colormap. using "terrain" instead')
        colormap = colormaps['dsm']
        v = cm.get_cmap(colormap, 256)

    print('minmax:', np.min(masked), np.max(masked))
    masked = np.ma.masked_outside(masked, ranges[dataname][0], ranges[dataname][1])
    adj, mn, mx = normalize(masked)
    # experimental:
    if dataname not in ['thermal','dsm']:
        mn, mx = ranges[dataname]
    c = v(adj)
    #print(c.shape)
    c = np.transpose(c, (2, 0, 1))
    #print(c.shape)
    c *= 255
    c = c.astype(int)
    #print(np.min(c[0]), np.max(c[0]), np.mean(c[0]))

    print('minmax:', mn, mx)
    norm = colors.Normalize(vmin=mn, vmax=mx)

    fig, ax = plt.subplots(figsize=(1, 6), constrained_layout=True)
    plt.close()
    cb = cbar.ColorbarBase(ax, v, norm)
    cb.set_label(label=label, rotation=90)
    outputname = os.path.join(outputpath, outputbase + '_'+dataname+'.tif')
    scalename = os.path.join(outputpath, outputbase + '_'+dataname+'_scale.png')
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


#  TESTS
def test_all():
    test_color()
    print('DONE WITH COLOR TEST ####################################################')
    test_no_color()

def test_color():
    fnames = generate_from_stack('test_ortho.tif', colormaps,
                                 os.getcwd(), '0test_colored', units='C')  # does a test of all indices


def test_no_color():
    fnames = generate_from_stack('test_ortho.tif', colormaps,
                                 os.getcwd(), '0test_no_color', units='C',
                                 colormap=False)  # does a test of all indices


def test_specific_colored(indices):
    fnames = generate_from_stack('test_ortho.tif', indices,
                                 os.getcwd(), '0test_colored', units='C')  # does a test of all indices


def test_specific_no_color(indices):
    fnames = generate_from_stack('test_ortho.tif', indices,
                                 os.getcwd(), '0test_no_color', units='C',
                                 colormap=False)  # does a test of all indices


def test_specific_all(indices):
    test_specific_colored(indices)
    print('DONE WITH COLOR TEST ####################################################')
    test_specific_no_color(indices)


if __name__ == '__main__':
    #  example of how to call generate_from_stack:
    #  fnames = generate_from_stack('test_ortho.tif',
    #                               {'thermal':'CMRmap','ndvi':'gnuplot2','ndre':'Spectral'},
    #                               os.getcwd(), 'test_colored', units='C')
    #  example of how to call gen_rgb to make an rgb orthomosaic:
    #  rgb = gen_rgb('test_ortho.tif', os.getcwd(),'test')
    #  example of how to call colormap_tif to make a colormapped dsm:
    #  dsmname = colormap_dsm('test_dem.tif', os.getcwd(), 'test1')
    #indices = ['savi']
    #test_specific_all(indices)
    test_color()
    pass
