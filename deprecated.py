# all the stuff in this file is deprecated and will eventually be removed.
import gdal
import numpy as np
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib.colorbar as cbar
import matplotlib.pyplot as plt
import os
from datetime import datetime as dt
from PIL import Image

def split_stack(filename):  # untested, but shouldn't be necessary
    """
    split_stack: splits a stacked .tif file into individual .tif files with 1 band each
    Params:
    :param filename: string of file location
    returns:
    :return: list of output filepaths
    """
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
    
    def generate_from_stack(file, indexdict, outputpath, outputbase, colormap=True, units='F', L=0.25):  # units can be F or C. units only used for thermal, L only used for SAVI
    #TODO: add max size parameter
    """
    generate_from_stack: generates multispectral indices from a stacked .tif file
    Params:
    :param file: string of source file location
    :param indexdict: dict of {index: colormap} where colormap is a string of the matplotlib colormap to use. Alternatively, a list of
    index names as strings, where the colormaps will default to those defined in colormaps. example: {'ndvi':'Spectral'} or ['ndvi']
    :param outputpath: string directory to output the output files into
    :param outputbase: name to add the index name and .tif into. example: outputbase = 'test_file', indexdict = ['ndvi']. ndvi file name output: 'test_file_ndvi.tif'
    :param colormap: bool. If true, outputfiles are 4-band RGBA images with colormaps defined in indexdict. If false, single-band tif which can be colormapped later
    :param units: string 'F' or 'C'. Only used if 'thermal' is in indexdict. If 'F', units of thermal map are Fahrenheit, else Celsius
    :param L: float between -1 and 1. Only used if 'savi' is in indexdict. is passed to gen_savi as L parameter.
    returns:
    :return: list of output filepaths where for each file, list item is [file_path, color_map_key] and if colormap is False, [file_path, None]
    """
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
                options = ['GTIFF_FORCE_RGBA=YES', 'PROFILE=GeoTIFF']
                out = driver.Create(outputname, cols, rows, 4, gdal.GDT_Byte, options=options)
                out.SetGeoTransform(trans)
                out.SetProjection(projection)
                for band, i in zip(c, range(1, 5)):
                    out.GetRasterBand(i).WriteArray(band)
                    #out.GetRasterBand(i).SetColorInterpretation(RGBA_INTERPS[i-1])
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
 