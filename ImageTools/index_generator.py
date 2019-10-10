import gdal
import numpy as np
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib.colorbar as cbar
import matplotlib.pyplot as plt
import os
from datetime import datetime as dt
from PIL import Image

def calc_index(numerator, denominator):
    """
    calc_index: dead simple internal function that divides np arrays without throwing errors for illegal operations and doesn't permanently change the np error settings
    Params:
    :param numerator: a np array
    :param denominator: a np array
    
    returns:
    :return: a 2D np array
    """
    np.seterr(divide='ignore', invalid='ignore')
    z = numerator + denominator == 0
    d = denominator == 0
    mask = d | z
    a = numerator/denominator
    np.seterr(divide='raise', invalid='raise')
    return np.ma.masked_array(a, mask=mask)


def mask_zeros(bands_,**kwargs):
    """
    mask_zeros: masks pixels in all bands of a multispectral np array (shape = (n, width, height) where n is number of bands)) where there is no data
    Params:
    :param bands_: dict where each entry is {name_of_band (string): 2D np array}

    returns:
    :return: dict where each entry is {name_of_band (string): 2D np array} with masked nodata values.
    """
    vals = list(bands_.values())
    return {k: np.ma.masked_where(sum(vals) == 0, bands_[k]) for k in bands_.keys()}


def gen_ndvi(bands,**kwargs):
    """
    gen_ndvi: calculates the ndvi for a multispectral np array (shape = (n, width, height) where n is number of bands))
    Params:
    :param bands: dict where each entry is {name_of_band (string): 2D np array}
    returns:
    :return: 2D np array with calculated index. values range from -1 to 1.
    """
    numerator = bands['nir'] - bands['red']
    denominator = bands['nir'] + bands['red']
    return calc_index(numerator, denominator)


def gen_ndre(bands,**kwargs):
    """
    gen_ndre: calculates the ndvi for a multispectral np array (shape = (n, width, height) where n is number of bands))
    Params:
    :param bands: dict where each entry is {name_of_band (string): 2D np array}
    returns:
    :return: 2D np array with calculated index. values range from -1 to 1.
    """
    numerator = bands['nir'] - bands['red_edge']
    denominator = bands['nir'] + bands['red_edge']
    return calc_index(numerator, denominator)


def gen_thermal(bands, temperature_units='C', **kwargs):
    """
    gen_thermal: extracts thermal data for a multispectral np array (shape = (n, width, height) where n is number of bands))
    Params:
    :param bands: dict where each entry is {name_of_band (string): 2D np array}
    returns:
    :return: 2D np array with thermal data in degrees celsius
    """
    try:
        dat = bands['lwir']
        dat = np.ma.masked_where(dat == 0, dat)
        if temperature_units == 'C':
            return dat/100 - 273.15
        else:
            return (dat/100 - 273.15)*9/5 + 32
    except KeyError:
        print('No thermal data available')
        return None


def gen_gndvi(bands,**kwargs):
    """
    gen_gndvi: calculates the gndvi for a multispectral np array (shape = (n, width, height) where n is number of bands))
    Params:
    :param bands: dict where each entry is {name_of_band (string): 2D np array}
    returns:
    :return: 2D np array with calculated index. values range from -1 to 1.
    """
    numerator = bands['nir'] - bands['green']
    denominator = bands['nir'] + bands['green']
    return calc_index(numerator, denominator)


def gen_endvi(bands,**kwargs):
    """
    gen_endvi: calculates the endvi for a multispectral np array (shape = (n, width, height) where n is number of bands))
    Params:
    :param bands: dict where each entry is {name_of_band (string): 2D np array}
    returns:
    :return: 2D np array with calculated index. values range from -1 to 1.
    """
    numerator = bands['nir'] + bands['green'] - 2 * bands['blue']
    denominator = bands['nir'] + bands['green'] + 2 * bands['blue']
    return calc_index(numerator, denominator)


def gen_savi(bands, L=0.5, **kwargs):
    """
    gen_savi: calculates the savi for a multispectral np array (shape = (n, width, height) where n is number of bands))
    Params:
    :param bands: dict where each entry is {name_of_band (string): 2D np array}
    :param L: SAVI coefficient, must be between -1 and 1. L should be 0.25 for mid-growth, 0.1 for early growth, apparently 0.5 is good for vineyards (?)
    returns:
    :return: 2D np array with calculated index. values range from -1 to 1. 
    """
    numerator = (bands['nir'] - bands['red'])/16934.0
    denominator = (bands['nir'] + bands['red'])/16934.0 + L
    return mask_extremes(calc_index(numerator, denominator)*(1+L))


def gen_gli(bands,**kwargs):
    """
    gen_gli: calculates the gli for a multispectral np array (shape = (n, width, height) where n is number of bands))
    Params:
    :param bands: dict where each entry is {name_of_band (string): 2D np array}
    returns:
    :return: 2D np array with calculated index. values range from -1 to 1.
    """
    numerator = 2 * bands['green'] - (bands['red'] + bands['blue'])
    denominator = 2 * bands['green'] + (bands['red'] + bands['blue'])
    return calc_index(numerator, denominator)


def gen_vari(bands,**kwargs):
    """
    gen_vari: calculates the vari for a multispectral np array (shape = (n, width, height) where n is number of bands))
    Params:
    :param bands: dict where each entry is {name_of_band (string): 2D np array}
    returns:
    :return: 2D np array with calculated index. values range from -1 to 1.
    """
    numerator = bands['green'] - bands['red']
    denominator = bands['green'] + bands['red'] - bands['blue']
    return calc_index(numerator, denominator)


def gen_gdi(bands,**kwargs):  # may be incorrect
    """
    gen_gdi: calculates the gdi for a multispectral np array (shape = (n, width, height) where n is number of bands))
    Params:
    :param bands: dict where each entry is {name_of_band (string): 2D np array}
    returns:
    :return: 2D np array with calculated index. values range from -1 to 1.
    """
    return (bands['nir'] - bands['green'])/16934.0


def gen_dvi(bands,**kwargs):  # may be incorrect
    """
    gen_dvi: calculates the dvi for a multispectral np array (shape = (n, width, height) where n is number of bands))
    Params:
    :param bands: dict where each entry is {name_of_band (string): 2D np array}
    returns:
    :return: 2D np array with calculated index. values range from -1 to 1.
    """
    return (bands['nir'] - bands['red'])/16934.0

def gen_dsm(bands, elevation_units='m', **kwargs):
    if elevation_units=='ft':
        return bands['dsm']*3.28
    else:
        return bands['dsm']

def mask_extremes(arr, std_devs=5):
    """
    mask_extremes: masks values above std_devs standard deviations from the mean
    Params:
    :param arr: 2D np array
    :param std_devs: int
    returns:
    :return: 2D np array with outliers masked
    """
    r = arr.std()*std_devs
    low = arr.mean() - r
    high = arr.mean() + r

    return np.ma.masked_outside(arr, low, high)


def normalize(arr):
    """
    normalize: normalizes an input array between 0 and 1
    Params:
    :param arr: 2D np array
    returns:
    :return: tuple (2D np array, old minimum value, old maximum value)
    """
    mn = np.min(arr)
    mx = np.max(arr)
    return (arr - mn)/(mx-mn), mn, mx
    
    
def getLabel(index, **kwargs):
    if index == 'thermal':
        return 'Temperature ({}{})'.format(u'\N{DEGREE SIGN}', kwargs['temperature_units'])
    elif index == 'dsm':
        return 'Elevation ({})'.format(kwargs['elevation_units'])
    else:
        return index


def mask_index(band, index):
    #nodata = band.mask
    valid = np.ma.masked_invalid(band)
    valid = np.ma.masked_outside(valid, ranges[index][0],ranges[index][1])
    #mask = nodata | invalid | outside
    
    return valid


equations = {
    'ndvi': gen_ndvi,
    'ndre': gen_ndre,
    'thermal': gen_thermal,
    'dsm': gen_dsm,
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




def diagnose(file, band=None, mask=None, **kwargs):
    """
    diagnose: opens a georeferenced .tif file, plots it, and returns the np array of the data.
    Params:
    :param file: string of file location
    :param band: number of band to open, if multiband file
    :param mask: lower threshold for masked values. For most data, -10000 should be used.
    :param **kwargs: matplotlib.pyplot kwargs
    returns:
    :return: np array of data in (selected band of) file
    """
    d = gdal.Open(file)
    if band is not None:
        a = d.ReadAsArray()[band]
    else:
        a = d.ReadAsArray()
    if mask is not None:
        a = np.ma.masked_where(a <= mask,a)
    plt.imshow(a, **kwargs)
    return a


def gen_rgb(file, outputpath, outputbase):
    """
    gen_rgb: generates an RGBA image from stacked .tif file
    Params:
    :param filename: string of input file location
    :param outputpath: directory to place output into
    :param outputbase: name to add the index name and .tif into. example: outputbase = 'test_file'. file name output: 'test_file_rgb.tif'
    returns:
    :return: full filepath of output image
    """
    start_time = dt.now()
    print('rgb generation started at ', start_time)
    base = gdal.Open(file)
    trans = base.GetGeoTransform()
    projection = base.GetProjection()
    data = base.ReadAsArray()[:3][::-1]
    alpha = data[0] + data[1] + data[2]
    alpha = np.ma.masked_where(alpha !=0, alpha)
    alpha = alpha.filled(255)
    rows, cols = data[0].shape
    driver = gdal.GetDriverByName('GTiff')

    outputname = os.path.join(outputpath, outputbase + '_rgb.tif')

    options = ['GTIFF_FORCE_RGBA=YES', 'PROFILE=GeoTIFF']
    out = driver.Create(outputname, cols, rows, 4, gdal.GDT_Byte, options=options)
    out.SetGeoTransform(trans)
    out.SetProjection(projection)
    for band, i in zip(data, range(1, 4)):
        band = np.asarray(band, dtype='float32')*255.0/65535
        out.GetRasterBand(i).WriteArray(band)

    out.GetRasterBand(4).WriteArray(alpha)
    print('saving...')
    out.FlushCache()

    del out
    end_time = dt.now()
    print('rgb generation ended at ', end_time)
    print('total time to generate rgb:', end_time - start_time)
    return outputname


def colormap_tif(file, outputpath, outputbase, dataname, colormap=None, units='m'):  # units can be 'ft' or 'm'
    """
    colormap_tif: generates an RGBA image from a single-band .tif file using a colormap
    Params:
    :param file: string of input file location
    :param outputpath: directory to place output into
    :param outputbase: name to add the index name and .tif into. example: outputbase = 'test_file', dataname='dsm'. file name output: 'test_file_dsm.tif'
    :param dataname: name of data in file. This is used to generate the scale, and to apply special behavior if the file is an elevation map.
    :param colormap: string of colormap to use for the output file. Defaults to that defined in colormaps for dataname.
    :param units: only used if dataname = 'dsm'. If units == 'ft', the elevation units are in feet. Else, the units on the scale are in meters.
    returns:
    :return: list: [full filepath of output image, colormap scale]
    """
    start_time = dt.now()
    print('tif colormap started at ', start_time)
    if colormap == None:
        colormap = colormaps[dataname]
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
    #options = ['PHOTOMETRIC=RGB', 'PROFILE=GeoTIFF']
    options = ['PROFILE=GeoTIFF','GTIFF_FORCE_RGBA=YES']
    out = driver.Create(outputname, cols, rows, 4, gdal.GDT_Byte, options=options)
    out.SetGeoTransform(trans)
    out.SetProjection(projection)
    for band, i in zip(c, range(1, 5)):
        out.GetRasterBand(i).WriteArray(band)
        #out.GetRasterBand(i).SetColorInterpretation(RGBA_INTERPS[i-1])
    out.FlushCache()
    end_time = dt.now()
    print('colormap generation ended at ', end_time)
    print('total time to generate:', end_time - start_time)
    return names


def resize_tif(src, destdir, destname, resize_factor, del_orig=False):
    """
    resize_tif: resizes a .tif of any size by a factor
    Params:
    :param src: string of input file location
    :param destdir: directory to place output into
    :param destname: name of output file
    :param resize_factor: float, factor to resize image by. This is the factor by which both the width and height are multiplied by, so the change in file size is resize_factor^2
    :param del_org: bool, default False. If true, deletes the file specified in param src.
    returns:
    :return: full filepath of output image
    """
    start_time = dt.now()
    print('resizing',src)
    print('tif resize started at ', start_time)
    if '.tif' not in destname:
        destname+='.tif'
    outputname =os.path.join(destdir, destname)
    large = gdal.Open(src)
    num_bands = large.RasterCount
    geo_trans = large.GetGeoTransform()
    projection = large.GetProjection()
    datatype = large.GetRasterBand(1).DataType
    new_geo_trans = (geo_trans[0], geo_trans[1]/resize_factor, geo_trans[2],
                     geo_trans[3], geo_trans[4], geo_trans[5]/resize_factor)
    x, y = large.RasterXSize, large.RasterYSize
    newx, newy = int(x*resize_factor), int(y*resize_factor)
    driver = gdal.GetDriverByName('GTiff')
    options = ['PROFILE=GeoTIFF','GTIFF_FORCE_RGBA=YES']
    temp_rst = driver.Create(outputname, newx, newy,
                             num_bands, datatype, options=options)
    temp_rst.SetGeoTransform(new_geo_trans)
    temp_rst.SetProjection(projection)
    
    for band in range(1,num_bands+1):
        temp_array = large.GetRasterBand(band).ReadAsArray()
        
        temp_array = np.ma.masked_where(temp_array==-10000,temp_array)
        new_band = np.array(Image.fromarray(temp_array).resize((newx,newy),Image.NEAREST))
        temp_rst.GetRasterBand(band).WriteArray(new_band)
        temp_rst.GetRasterBand(band).SetNoDataValue(-10000)
    #temp_rst.GetRasterBand(4).SetColorInterpretation(RGBA_INTERPS[3])
    del large
    temp_rst.FlushCache()
    del temp_rst
    if del_orig:
        os.remove(src)
    
    end_time = dt.now()
    print('tif resize ended at ', end_time)
    print('total time to resize tif:', end_time - start_time)
    return outputname


def resize_tif_list(filenames, destpath, suffix, resize_factor, del_orig=False):
    """
    resize_tif_list: resizes all .tifs in the list of any size by a factor
    Params:
    :param filenames: list of string of input file location (output from tifs_from_stack_output)
    :param destpath: directory to place output into
    :param suffix: suffix to add onto output filename
    :param resize_factor: float, factor to resize images by. This is the factor by which both the width and height are multiplied by, so the change in file size is resize_factor^2
    :param del_org: bool, default False. If true, deletes the original src file for each image
    returns:
    :return: list of full filepaths of output images
    """
    return [resize_tif(fname, destpath, os.path.split(fname)[-1][:-4] +'_'+ suffix, resize_factor, del_orig=del_orig) for fname in filenames]

