#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
import numpy as np
import pandas as pd
import cPickle as pickle
from glob import glob
import xml.etree.cElementTree as ET

from scipy.interpolate import RectBivariateSpline as interp2d
from scipy import ndimage, array, ogrid, mgrid

import matplotlib.pyplot as plt


# In[2]:


def readDataBlock(xmlnode):
    """ Turn any 'DataBlock' XML node into a numpy array of floats
    """
    
    string = xmlnode.text
    string = re.sub("[\t\s\n]", "", string)
    data = np.asarray(
        bytearray.fromhex(string),
        dtype = float
        )
    
#     vmin = float(xmlnode.get('min'))
#     vmax = float(xmlnode.get('max'))
#     return data * (vmax - vmin) / 255. + vmin
    return data


# In[3]:


def downsample(a, n, align=0):
    '''a: input array of 1-3 dimentions
       n: downsample to n bins
       optional:
       align : if non-zero, downsample grid (coords) 
               will have a bin at same location as 'align'
               ( typically max(sum profile) )
               useful for plots vs. phase
         
    '''
    if type(a) in [list]:
        result = []
        for b in a:
            result.append(downsample(b))
        return result
    else:
        shape = a.shape
        D = len(shape)
        if D == 1:
            coords = mgrid[0:1-1./n:1j*n]
        elif D == 2:
            d1,d2 = shape
            if align: 
                #original phase bins
                x2 = mgrid[0:1.-1./d2:1j*d2]
                #downsampled phase bins
                crd = mgrid[0:1-1./n[1]:1j*n[1]]
                crd += x2[align]
                crd = (crd % 1)
                crd.sort()
                offset = crd[0]*d2
                coords = mgrid[0:d1-1:1j*n[0], offset:d2-float(d2)/n[1]+offset:1j*n[1]]
            else:
                coords = mgrid[0:d1-1:1j*n[0], 0:d2-1:1j*n[1]]
        elif D == 3:
            d1,d2,d3 = shape
            coords = mgrid[0:d1-1:1j*n[0], 0:d2-1:1j*n[1], 0:d3-1:1j*n[2]]
        else:
            raise "too many dimentions %s " % D
        def map_to_index(x,bounds,N):
            xmin, xmax= bounds
            return (x - xmin)/(xmax-xmin)*N
        if D == 1:
            m = len(a)
            x = mgrid[0:1-1./m:1j*m]
            if align:
                #ensure new grid lands on max(a)
                coords += x[align]
                coords = coords % 1
                coords.sort()
            #newf = interp(x, a, bounds_error=True)
            #return newf(coords)
            return np.interp(coords, x, a)
        elif D == 2:
            #k,l = a.shape
            #x = mgrid[0:1:1j*k]
            #y = mgrid[0:1:1j*l]
            #f = interp2d(x, y, a)
            #coords = mgrid[0:1:1j*n]
            #return f(coords, coords)
            newf = ndimage.map_coordinates(a, coords, cval=np.median(a))
            return newf
        else:
            #coeffs = ndimage.spline_filter(a)
            newf = ndimage.map_coordinates(coeffs, coords, prefilter=False)
            #newf = ndimage.map_coordinates(coeffs, coords )
            return newf


# In[4]:


class Candidate(object):
    def __init__(self, fname):
        """ Build a new Candidate object from a PHCX file path.
        """
        xmlroot = ET.parse(fname).getroot()
        
        # Read Coordinates
        coordNode = xmlroot.find('head').find('Coordinate')
        # Separate PDMP & FFT sections
        for section in xmlroot.findall('Section'):
            if 'pdmp' in section.get('name').lower():
                self.opt_section = section
            else:
                self.fft_section = section
                
        profileNode = self.opt_section.find('Profile')
        self.profile = readDataBlock(profileNode)
        self.align = self.profile.argmax()
        

                
                
    def getdata(self, profile_bins=64, subbands_bins=64, subints=64, DM_bins=200):
        ### Sub-Integrations
        
        def greyscale(array2d):
            """
            greyscale(array2d, **kwargs):
                Plot a 2D array as a greyscale image using the same scalings
                    as in prepfold.
            """
            # Use the same scaling as in prepfold_plot.c
            global_max = array2d.max()
            min_parts = np.minimum.reduce(array2d, 1)
            array2d = (array2d - min_parts[:,np.newaxis]) / (np.fabs(global_max) - np.fabs(min_parts.max()))
            return array2d
        
        
        def subints_fig(M):
            subintsNode = self.opt_section.find('SubIntegrations')
            nsubs = int(subintsNode.get('nSub'))
            nbins = int(subintsNode.get('nBins'))
            subints = readDataBlock(subintsNode).reshape(nsubs, nbins)
            subints = subints[-16:, :64]
            subints /= 255.

#             sed = np.mean(subints, axis=1)
#             valid_index = np.where(sed > 0)
#             self.missed_ratio_time_vs_phase = 1. - len(valid_index) * 1.0 / len(sed)
#             if self.missed_ratio_time_vs_phase < 0.8:
#                 subints = subints[valid_index]
#             else:
#                 self.missed_ratio_time_vs_phase = 0.
            if (M is not None) and (subints.shape[0] != M):
                subints = downsample(subints, [M, 64], self.align)

#             subints = greyscale(subints)

            return subints


        ### Sub-Bands
        def subbands_fig(M):
            subbandsNode = self.opt_section.find('SubBands')
            nsubs = int(subbandsNode.get('nSub'))
            nbins = int(subbandsNode.get('nBins'))
            subbands = readDataBlock(subbandsNode).reshape(nsubs, nbins)
            subbands = subbands[-16:, :64]
            subbands /= 255.

#             sed = subbands.mean(axis=1)
#             valid_index = np.where(sed > 0)[0]
#             self.missed_ratio_subbands = 1. - len(valid_index) * 1.0 / len(sed)
#             if self.missed_ratio_subbands < 0.8:
#                 subbands = subbands[valid_index]
#             else:
#                 self.missed_ratio_subbands = 0.
            if (M is not None) and (subbands.shape[0] != M):
                subbands = downsample(subbands, [M, 64], self.align)

#             subbands = greyscale(subbands)

            return subbands

        ### Profile
        def profile_curve(M):

            if (M is not None) and (len(self.profile) != M):
                profile = downsample(self.profile, M, self.align)
            else:
                profile = self.profile

            normprof = profile - min(profile)

            if np.max(normprof) != 0:
                profile = normprof / max(normprof)

            return profile

        ### DmCurve: FFT S/N vs. PEASOUP Trial DM, at best candidate acceleration
        def DM_curve(M):
            dmcurve_node = self.fft_section.find('DmCurve')
            text = dmcurve_node.find('SnrValues').text
            DM = np.asarray(map(float, text.split()))
            if (M is not None) and (len(DM) != M):
                DM = downsample(DM, M)

            norm_DM = DM - min(DM)

            if np.max(norm_DM) != 0:
                DM = norm_DM / max(norm_DM)
            return DM


        data = {'sumprof': profile_curve(profile_bins),
                'subbands': subbands_fig(subbands_bins),
                'time_vs_phase': subints_fig(subbands_bins),
                'DM': DM_curve(DM_bins)
               }
        return data


# In[5]:


pulsar_flist = glob('../pulsar & RFI/HTRU 1/pulsars/*.phcx')
rfi_flist = glob('../pulsar & RFI/HTRU 1/negatives/RFI/*phcx')


# In[6]:


print('number of pulsars: ', len(pulsar_flist))
print('number of rfis: ', len(rfi_flist))


# In[ ]:


pulsar_list = []
rfi_list = []
for f in pulsar_flist:
    p = Candidate(f)
    pulsar_list.append(p.getdata())
    
print('pulsar finished')
    
# random_index = np.random.permutation(len(rfi_flist))
# valid_index = random_index[:20000]
# for f in np.array(rfi_flist)[valid_index]:
for f in rfi_flist:
    p = Candidate(f)
    rfi_list.append(p.getdata())

print('rfi finished')
# In[ ]:


print('number of pulsars: ', len(pulsar_list))
print('number of rfis: ', len(rfi_list))


# In[ ]:


pfd_data = pd.concat([pd.DataFrame(pulsar_list), pd.DataFrame(rfi_list)])
y = np.concatenate([np.ones(len(pulsar_list)), np.zeros(len(rfi_list))])


# In[28]:


flist = pulsar_flist
flist.extend(list(np.array(rfi_flist)))
print('The number of samples: ', len(flist))
pfd_data.index = flist


# In[ ]:


data = {'HTRU_data': pfd_data, 'y': y}
pickle.dump(data, open('HTRU_for_FAST.pkl', 'wb'))


# In[ ]:




