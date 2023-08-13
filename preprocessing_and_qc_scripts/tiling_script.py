from __future__ import print_function
import json
import openslide
from openslide import open_slide, ImageSlide
from openslide.deepzoom import DeepZoomGenerator
from optparse import OptionParser
import re
import shutil
from unicodedata import normalize
import numpy as np
import scipy.misc
import subprocess
from glob import glob
from multiprocessing import Process, JoinableQueue
import time
import os
import sys
import pydicom as dicom
from imageio import imwrite as imsave
from imageio import imread
from xml.dom import minidom
from PIL import Image, ImageDraw, ImageCms
from skimage import color, io
from skimage.transform import resize
Image.MAX_IMAGE_PIXELS = None


VIEWER_SLIDE_NAME = 'slide'


class TileWorker(Process):
    """A child process that generates and writes tiles."""

    def __init__(self, queue, slidepath, tile_size, limit_bounds,quality):
        Process.__init__(self, name='TileWorker')
        self.daemon = True
        self._queue = queue
        self._slidepath = slidepath
        self._tile_size = tile_size
        self._limit_bounds = limit_bounds
        self._quality = quality
        self._slide = None

    def RGB_to_lab(self, tile):
        print("RGB to Lab")
        Lab = color.rgb2lab(tile)
        return Lab

    def Lab_to_RGB(self,Lab):
        print("Lab to RGB")
        newtile = (color.lab2rgb(Lab) * 255).astype(np.uint8)
        return newtile


    def normalize_tile(self, tile, NormVec):
        Lab = self.RGB_to_lab(tile)
        TileMean = [0,0,0]
        TileStd = [1,1,1]
        newMean = NormVec[0:3] 
        newStd = NormVec[3:6]
        for i in range(3):
            TileMean[i] = np.mean(Lab[:,:,i])
            TileStd[i] = np.std(Lab[:,:,i])
            # print("mean/std chanel " + str(i) + ": " + str(TileMean[i]) + " / " + str(TileStd[i]))
            tmp = ((Lab[:,:,i] - TileMean[i]) * (newStd[i] / TileStd[i])) + newMean[i]
            if i == 0:
                tmp[tmp<0] = 0 
                tmp[tmp>100] = 100 
                Lab[:,:,i] = tmp
            else:
                tmp[tmp<-128] = 128 
                tmp[tmp>127] = 127 
                Lab[:,:,i] = tmp
        tile = self.Lab_to_RGB(Lab)
        return tile

    def run(self):
        self._slide = open_slide(self._slidepath)
        last_associated = None
        dz = self._get_dz()
        while True:
            data = self._queue.get()
            if data is None:
                self._queue.task_done()
                break
            #associated, level, address, outfile = data
            associated, level, address, outfile, format, outfile_bw, PercentMasked, TileMask, Normalize = data
            if last_associated != associated:
                dz = self._get_dz(associated)
                last_associated = associated
            #try:
            if True:
                try:
                    tile = dz.get_tile(level, address)
                    if Normalize != '':
                        print("normalize " + str(outfile))
                        tile = Image.fromarray(self.normalize_tile(tile, Normalize).astype('uint8'),'RGB')

                    if tile.size[0] == tile.size[1]:
                        tile.save(outfile, quality=self._quality)
                    self._queue.task_done()
                except Exception as e:
                    # print(level, address)
                    print("image %s failed at dz.get_tile for level %f" % (self._slidepath, level))
                    # e = sys.exc_info()[0]
                    print(e)
                    self._queue.task_done()

    def _get_dz(self, associated=None):
        if associated is not None:
            image = ImageSlide(self._slide.associated_images[associated])
        else:
            image = self._slide
        return DeepZoomGenerator(image, self._tile_size, overlap=0, limit_bounds=self._limit_bounds)


class DeepZoomImageTiler(object):
    """Handles generation of tiles and metadata for a single image."""

    def __init__(self, dz, basename, format, associated, queue, slide, basenameJPG, threshold, penthreshold, ImgExtension, Mag, normalize, mask=None,pen_mask = None,tile_size=None, workers = 4):
        self._dz = dz
        self._basename = basename
        self._basenameJPG = basenameJPG
        self._format = format
        self._associated = associated
        self._queue = queue
        #self._queue = None
        self._processed = 0
        self._slide = slide
        self._threshold = threshold
        self._penthreshold = penthreshold
        self._ImgExtension = ImgExtension
        self._Mag = Mag
        self._normalize = normalize
        self._mask = mask
        self._pen_mask = pen_mask
        self._tile_size = tile_size
        self._workers = workers

    def run(self):
        self._write_tiles()
        self._write_dzi()

    def _write_tiles(self):
            ########################################3
            # nc_added
        #level = self._dz.level_count-1
        Magnification = 20
        tol = 2
        #get slide dimensions, zoom levels, and objective information
        Factors = self._slide.level_downsamples
        
        try:
            Objective = float(self._slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
            # print(self._basename + " - Obj information found")
        except:
            print(self._basename + " - No Obj information found")
            print(self._ImgExtension)
            if ("jpg" in self._ImgExtension) | ("dcm" in self._ImgExtension) | ("tif" in self._ImgExtension):
                #Objective = self._ROIpc
                Objective = 1.
                Magnification = Objective
                print("input is jpg - will be tiled as such with %f" % Objective)
            else:
                return
        #calculate magnifications
        Available = tuple(Objective / x for x in Factors)
        #find highest magnification greater than or equal to 'Desired'
        Mismatch = tuple(x-Magnification for x in Available)
        AbsMismatch = tuple(abs(x) for x in Mismatch)
        if len(AbsMismatch) < 1:
          print(self._basename + " - Objective field empty!")
          return

        if True:

            for level in range(self._dz.level_count-1,-1,-1):
                ThisMag = Available[0]/pow(2,self._dz.level_count-(level+1))
                if self._Mag > 0:
                    if ThisMag != self._Mag:
                        continue
                ########################################
                #tiledir = os.path.join("%s_files" % self._basename, str(level))
                tiledir = os.path.join("%s_files" % self._basename, str(ThisMag))
                if not os.path.exists(tiledir):
                    os.makedirs(tiledir)
                cols, rows = self._dz.level_tiles[level]

                for row in range(rows):
                    for col in range(cols):
                        InsertBaseName = False
                        if InsertBaseName:
                          tilename = os.path.join(tiledir, '%s_%d_%d.%s' % (
                                          self._basenameJPG, col, row, self._format))
                          tilename_bw = os.path.join(tiledir, '%s_%d_%d_mask.%s' % (
                                          self._basenameJPG, col, row, self._format))
                        else:
                          tilename = os.path.join(tiledir, '%d_%d.%s' % (
                                          col, row, self._format))
                          tilename_bw = os.path.join(tiledir, '%d_%d_mask.%s' % (
                                          col, row, self._format))

                        if type(self._mask) == np.ndarray and self._tile_size:
                            mask_tile_size = int((self._tile_size/self._Mag)*1.25)
                            
                            TileMask = self._mask[row*mask_tile_size:(row+1)*mask_tile_size,
                                 col*mask_tile_size:(col+1)*mask_tile_size]/255
                            
                            
                            
                            PercentMasked = TileMask.mean()
                            
                            if type(self._pen_mask) == np.ndarray:
                                PenMask = self._pen_mask[row*mask_tile_size:(row+1)*mask_tile_size,
                                         col*mask_tile_size:(col+1)*mask_tile_size]/255
                                PercentPen = PenMask.mean()
                            else:
                                PercentPen = 0.0
                            
                        else:
                            PercentMasked = 1.0
                            PercentPen = 0.0
                            TileMask = []

                        if not os.path.exists(tilename) and PercentMasked >= (self._threshold / 100.0) and PercentPen < (self._penthreshold / 100.0):
                            if self._workers == 1:
                                tile = self._dz.get_tile(level, (col, row))
                                if self._normalize != '':
                                    print("normalize " + str(tilename))
                                    tile = Image.fromarray(TileWorker.normalize_tile(tile, Normalize).astype('uint8'),'RGB')
                                
                                if tile.size[0] == tile.size[1]:
                                    tile.save(tilename)
                            else:
                                self._queue.put((self._associated, level, (col, row),
                                            tilename, self._format, tilename_bw, PercentMasked, TileMask, self._normalize))
                        self._tile_done()

    def _tile_done(self):
        self._processed += 1
        count, total = self._processed, self._dz.tile_count
        if count % 100 == 0 or count == total:
            print("Tiling %s: wrote %d/%d tiles" % (
                    self._associated or 'slide', count, total),
                    end='\r', file=sys.stderr)
            if count == total:
                print(file=sys.stderr)

    def _write_dzi(self):
        with open('%s.dzi' % self._basename, 'w') as fh:
            fh.write(self.get_dzi())

    def get_dzi(self):
        return self._dz.get_dzi(self._format)


class DeepZoomStaticTiler(object):
    """Handles generation of tiles and metadata for all images in a slide."""

    def __init__(self, slidepath, basename, format, tile_size,limit_bounds, quality, workers, basenameJPG, threshold, penthreshold, oLabel, ImgExtension, Mag, normalize, mask_filepath=None):

        self._slide = open_slide(slidepath)
        self._limit_bounds = limit_bounds
        self._basename = basename
        self._basenameJPG = basenameJPG
        self._format = format
        self._tile_size = tile_size
        self._queue = JoinableQueue(2 * workers)
        #self._queue = None
        self._workers = workers
        self._threshold = threshold
        self._penthreshold = penthreshold
        self._ImgExtension = ImgExtension
        self._Mag = Mag
        self._normalize = normalize
        if mask_filepath:
            self._mask = np.asarray(Image.open(mask_filepath +'_mask_use.png'))
            try:
                self._pen_mask = np.asarray(Image.open(mask_filepath +'_pen_markings.png'))
            except:
                self._pen_mask = None
        else:
            self._mask = None
            self._pen_mask = None

        if workers > 1:
            for _i in range(workers):
                TileWorker(self._queue, slidepath, tile_size, quality, self._threshold).start()

    def run(self):
        self._run_image()
        self._shutdown()

    def _run_image(self, associated=None):
        """Run a single image from self._slide."""
        if associated is None:
            image = self._slide
            basename = self._basename
        else:
            image = ImageSlide(self._slide.associated_images[associated])
            basename = os.path.join(self._basename, self._slugify(associated))
        # print("enter DeepZoomGenerator")
        dz = DeepZoomGenerator(image, self._tile_size, overlap = 0,limit_bounds=self._limit_bounds)
        
        # print("enter DeepZoomImageTiler")
        tiler = DeepZoomImageTiler(dz, basename, self._format, associated,self._queue, self._slide, self._basenameJPG, self._threshold, self._penthreshold, self._ImgExtension, self._Mag, self._normalize, self._mask,self._pen_mask,self._tile_size, self._workers)
        tiler.run()



    def _url_for(self, associated):
        if associated is None:
            base = VIEWER_SLIDE_NAME
        else:
            base = self._slugify(associated)
        return '%s.dzi' % base


    def _write_static(self):
        basesrc = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                'static')
        basedst = os.path.join(self._basename, 'static')
        self._copydir(basesrc, basedst)
        self._copydir(os.path.join(basesrc, 'images'),
                os.path.join(basedst, 'images'))

    def _copydir(self, src, dest):
        if not os.path.exists(dest):
            os.makedirs(dest)
        for name in os.listdir(src):
            srcpath = os.path.join(src, name)
            if os.path.isfile(srcpath):
                shutil.copy(srcpath, os.path.join(dest, name))

    @classmethod
    def _slugify(cls, text):
        text = normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode()
        return re.sub('[^a-z0-9]+', '_', text)

    def _shutdown(self):
        for _i in range(self._workers):
            self._queue.put(None)
        self._queue.join()



def ImgWorker(queue):
    # print("ImgWorker started")
    while True:
        cmd = queue.get()
        if cmd is None:
            queue.task_done()
            break
        # print("Execute: %s" % (cmd))
        subprocess.Popen(cmd, shell=True).wait()
        queue.task_done()



if __name__ == '__main__':
    parser = OptionParser(usage='Usage: %prog [options] <slide>')
    parser.add_option('-L', '--ignore-bounds', dest='limit_bounds',
        default=True, action='store_false',
        help='display entire scan area')
    parser.add_option('-f', '--format', metavar='{jpeg|png}', dest='format',
        default='jpeg',
        help='image format for tiles [jpeg]')
    parser.add_option('-j', '--jobs', metavar='COUNT', dest='workers',
        type='int', default=4,
        help='number of worker processes to start [4]')
    parser.add_option('-o', '--output', metavar='NAME', dest='basename',
        help='base name of output file')
    parser.add_option('-Q', '--quality', metavar='QUALITY', dest='quality',
        type='int', default=90,
        help='JPEG compression quality [90]')
    parser.add_option('-s', '--size', metavar='PIXELS', dest='tile_size',
        type='int', default=512,
        help='tile size [512]')
    parser.add_option('-t', '--threshold', metavar='PIXELS', dest='threshold',
        type='float', default=75,
        help='Minimimum of tile mask flagged as useable')
    parser.add_option('-p', '--penthreshold', metavar='PIXELS', dest='penthreshold',
        type='float', default=2.5,
        help='Minimimum of tile mask flagged as useable')
    parser.add_option('-M', '--Mag', metavar='PIXELS', dest='Mag',
        type='float', default=-1,
        help='Magnification at which tiling should be done (-1 of all)')
    parser.add_option('-N', '--normalize', metavar='NAME', dest='normalize',
        help='if normalization is needed, N list the mean and std for each channel. For example \'57,22,-8,20,10,5\' with the first 3 numbers being the targeted means, and then the targeted stds')
    parser.add_option('-m','--mask', metavar='HistoQC mask', dest='mask_filepath',
        help='prefix used in HistoQC output')




    (opts, args) = parser.parse_args()


    try:
        slidepath = args[0]
    except IndexError:
        parser.error('Missing slide argument')
    if opts.basename is None:
        opts.basename = os.path.splitext(os.path.basename(slidepath))[0]

    try:
        if opts.normalize is not None:
            opts.normalize = [float(x) for x in opts.normalize.split(',')]
            if len(opts.normalize) != 6:
                opts.normalize = ''
                parser.error("ERROR: NO NORMALIZATION APPLIED: input vector does not have the right length - 6 values expected")
        else:
            opts.normalize  = ''

    except:
        opts.normalize = ''
        parser.error("ERROR: NO NORMALIZATION APPLIED: input vector does not have the right format")


    # get  images from the data/ file.
    files = glob(slidepath)  
    #ImgExtension = os.path.splitext(slidepath)[1]
    ImgExtension = slidepath.split('*')[-1]


    files = sorted(files)
    for imgNb in range(len(files)):
        filename = files[imgNb]
        #print(filename)
        opts.basenameJPG = os.path.splitext(os.path.basename(filename))[0]
        print("processing: " + opts.basenameJPG + " with extension: " + ImgExtension)
        output = os.path.join(opts.basename, opts.basenameJPG)
        #if os.path.exists(output + "_files"):
        #    print("Image %s already tiled" % opts.basenameJPG)
        #    continue
        #try:
        print(opts.mask_filepath)
        DeepZoomStaticTiler(filename, output, opts.format, opts.tile_size, opts.limit_bounds, opts.quality, opts.workers, opts.basenameJPG, opts.threshold, opts.penthreshold, '', ImgExtension, opts.Mag, opts.normalize, opts.mask_filepath).run()
    #except Exception as e:
        #print("Failed to process file %s, error: %s" % (filename, sys.exc_info()[0]))
        #print(e)

    print("End")









