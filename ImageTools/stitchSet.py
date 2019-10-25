import gdal
import numpy as np
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib.colorbar as cbar
import matplotlib.pyplot as plt
import os
from datetime import datetime as dt
from PIL import Image
import ImageTools.index_generator as ig


class StitchSet():
    def __init__(self, ortho_file, dsm_file, band_order=ig.bandNames,
                 output_directory='.', output_base='', 
                 temperature_units='C', elevation_units='m', savi_L=0.5):
        self._ortho_file_name = ortho_file
        self._dsm_file_name = dsm_file
        self._output_directory = output_directory
        self._output_base = output_base
        
        ortho = gdal.Open(self._ortho_file_name)
        dsm = gdal.Open(self._dsm_file_name)
        dsm_data = dsm.ReadAsArray()
        
        self._bands = ig.mask_zeros({band:layer.astype(float) for band, layer in zip(band_order, ortho.ReadAsArray())})
       
        self._bands['dsm'] = np.ma.masked_where(dsm_data < -2000, dsm_data)

        self._projection = ortho.GetProjection()
        self._geo_transform = ortho.GetGeoTransform()
        self._indices = dict()
        self._colormaps = ig.colormaps
        
        self.kwargs = dict(temperature_units = temperature_units,
                           elevation_units = elevation_units,
                           savi_L = savi_L)
                           
        ortho, dsm = None, None
    
    def setColormaps(self, colormaps_dict):
        self._colormaps.update(colormaps_dict)
    
    def updateKwargs(self, **kwargs):
        self.kwargs.update(**kwargs)
        return self.kwargs
        
    def generateIndex(self, index, **kwargs):
        kwargs = self.updateKwargs(**kwargs)
        self._indices[index] = ig.equations[index](self._bands, **kwargs)
        
    def generateIndices(self, indices, **kwargs):
        kwargs = self.updateKwargs(**kwargs)
        self._indices.update({index:ig.equations[index](self._bands, **kwargs) for index in indices})
        
    def guaranteeIndex(self, index):
        if index not in self._indices.keys():
            self.generateIndices([index], **self.kwargs)
    
    def exportRGBAImage(self, mean_value=40):
        mask = self._bands['red'].mask
        alpha = np.ma.masked_array(np.full(mask.shape, 255), mask=mask).filled(0)
        bands = np.stack((self._bands['red'], self._bands['green'], self._bands['blue']))
        bands = bands.astype(float)*255.0/65535.0
        if mean_value is not None:
            masked = np.stack([np.ma.masked_array(band, mask) for band in bands])
            mean = masked.mean()
            print('mean:', mean)
            bands*=mean_value/mean
        bands = np.stack((bands[0], bands[1], bands[2], alpha))
        tif_path = os.path.join(self._output_directory, '{}_rgba.tif'.format(self._output_base))
        return self.saveColorGeoTiff(bands, tif_path)
    
    def saveGeoTiff(self, bands, filepath, dtype, options, **kwargs):  # bands is list of np arrays or a 3d np array
        num_bands = len(bands)
        driver = gdal.GetDriverByName('GTiff')
        shape = bands[0].shape[::-1]
        file = driver.Create(filepath, shape[0], shape[1], num_bands, dtype, options=options)
        if num_bands > 1:
            for band, i in zip(bands, range(1, num_bands+1)):
                file.GetRasterBand(i).WriteArray(band)
        elif 'index' in kwargs.keys() and 'fill' in kwargs.keys() and kwargs['fill']:
            index = kwargs['index']
            data = np.ma.masked_outside(bands[0], ig.ranges[index][0], ig.ranges[index][1])
            file.GetRasterBand(1).WriteArray(data.filled(-10000))
        else:
            file.GetRasterBand(1).WriteArray(bands[0])  
            
        file.SetProjection(self._projection)
        file.SetGeoTransform(self._geo_transform)
        file.FlushCache()
        file = None
        return filepath
    
    def saveColorGeoTiff(self, bands, filepath):
        dtype = gdal.GDT_Byte
        return self.saveGeoTiff(bands, filepath, dtype, ['PROFILE=GeoTIFF','GTIFF_FORCE_RGBA=YES'])
    
    def saveMonoGeoTiff(self, band, index, filepath):
        dtype = gdal.GDT_Float32
        return self.saveGeoTiff([band], filepath, dtype, ['PROFILE=GeoTIFF'], fill=True, index=index)
    
    def saveColorKey(self, colormap, label, minimum, maximum, filepath):
        plt.ioff()
        fig, ax = plt.subplots(figsize=(1, 6), constrained_layout=True)
        norm = colors.Normalize(vmin=minimum, vmax=maximum)
        color_bar = cbar.ColorbarBase(ax, colormap, norm)
        color_bar.set_label(label=label, rotation=90)
        fig.savefig(filepath)
        fig, ax = None, None
        plt.ion()
        return filepath
        
    def exportColorImage(self, index, colormap_name, label_name):
        print('exporting colored',index)
        color_map = cm.get_cmap(colormap_name, 256)
        self.guaranteeIndex(index)
        data = self._indices[index]
        data = np.ma.masked_outside(data, ig.ranges[index][0], ig.ranges[index][1])
        data, min_value, max_value = ig.normalize(data)
        print('minmax', min_value, max_value)
        if index not in ['thermal', 'dsm']:
            min_value, max_value = ig.ranges[index]
        
        colormapped_data = color_map(data)
        colormapped_data = (np.transpose(colormapped_data,(2, 0, 1))*255).astype(int)
        
        tif_path = os.path.join(self._output_directory, '{}_{}_color.tif'.format(self._output_base,index))
        scale_path = os.path.join(self._output_directory, '{}_{}_scale.png'.format(self._output_base,index))
        
        self.saveColorGeoTiff(colormapped_data, tif_path)
        self.saveColorKey(color_map, label_name, min_value, max_value, scale_path)
        return [tif_path, scale_path]
    
    def exportMonoImage(self, index):
        print('exporting mono',index)
        self.guaranteeIndex(index)
        tif_path = os.path.join(self._output_directory, '{}_{}_mono.tif'.format(self._output_base,index))
        return self.saveMonoGeoTiff(self._indices[index], index, tif_path)
    
    def exportGeneratedIndicesAsColorImages(self):
        return [self.exportColorImage(index, self._colormaps[index], ig.getLabel(index, **self.kwargs)) for index in self._indices.keys()]
    
    def exportGeneratedIndicesAsMonoImages(self):
        return [self.exportMonoImage(index) for index in self._indices.keys()]
            