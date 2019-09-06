import os
import matplotlib as mpl
if os.getenv('DISPLAY') is None: mpl.use('Agg')
import matploblib.pyplot as py

import numpy as np

from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.widgets      import Slider, TextBox

class ThreeAxisViewer:
  """ simplistic three axis viewer for multiple aligned 3d or 4d arrays

  Parameters
  ----------
  vols : list
    a list of 3d or 4d numpy arrays containing the image volumes
  vxosize: list, optional
    a 3 element with the voxel size
  width: float, optional
    width of the figure
  sl_x, sl_y, sl_z, sl_t: int, optional
    slices to show at beginning
  ls : string, optional
     str specifying the line style of the cross hair (use '' for no cross hair)
  imshow_kwargs : list of dictionaries
     list of dictionaries with keyword arguments passed to pylab.imshow()
  rowlabels : list of strings
     containing the labels for every row (volume)

  Note
  ----
  Scrolling with the mouse or the up and down keys can be used to scroll thorugh the slices.
  The left and right keys scroll through time frames.
  The viewer expects the input volumes to be in LPS orientation.
  If the input is 4D, the time axis should be the left most axis.

  Example
  -------
  ims_kwargs = [{'vmin':-1,'vmax':1},{'vmin':-2,'vmax':2,'cmap':py.cm.jet}]
  vi = ThreeAxisViewer([np.random.randn(90,90,80),np.random.randn(90,90,80)], imshow_kwargs = ims_kwargs)
  """
  def __init__(self, vols, 
                     voxsize       = [1.,1.,1.], 
                     width         = None, 
                     sl_x          = None, 
                     sl_y          = None, 
                     sl_z          = None, 
                     sl_t          = 0, 
                     ls            = ':',
                     rowlabels     = None,
                     imshow_kwargs = {}):

    if not isinstance(vols,list): 
      self.vols = [vols]
    else:
      self.vols = vols

    self.n_vols = len(self.vols)
    self.ndim   = self.vols[0].ndim

    if self.ndim: self.nframes = self.vols[0].shape[0]
    else:         self.nframes = 1 

    # set up the slice objects for correct slicing of 3d and 4d arrays
    if self.ndim == 3:
      self.ix = 0
      self.iy = 1
      self.iz = 2
    
    if self.ndim == 4:
      self.ix = 1
      self.iy = 2
      self.iz = 3
   
    self.shape = self.vols[0].shape

    if sl_x is None: 
      self.sl_x = self.shape[self.ix] // 2
    else:
      self.sl_x = sl_x

    if sl_y is None: 
      self.sl_y = self.shape[self.iy] // 2
    else:
      self.sl_y = sl_y

    if sl_z is None: 
      self.sl_z = self.shape[self.iz] // 2
    else:
      self.sl_z = sl_z

    self.sl_t = sl_t

    if self.ndim == 4:
      self.fstr = ', ' + str(self.sl_t)
    else:
      self.fstr = ''

    self.imshow_kwargs = imshow_kwargs
    
    if not isinstance(self.imshow_kwargs,list): 
      tmp = self.imshow_kwargs.copy()
      self.imshow_kwargs = []
      for i in range(self.n_vols): self.imshow_kwargs.append(tmp.copy())

    for i in range(self.n_vols):
      if not 'cmap' in self.imshow_kwargs[i]:          
        self.imshow_kwargs[i]['cmap'] = py.cm.Greys
      if not 'interpolation' in self.imshow_kwargs[i]: 
        self.imshow_kwargs[i]['interpolation'] = 'nearest'
      if not 'vmin'          in self.imshow_kwargs[i]: 
        self.imshow_kwargs[i]['vmin'] = self.vols[i].min()
      if not 'vmax'          in self.imshow_kwargs[i]: 
        self.imshow_kwargs[i]['vmax'] = self.vols[i].max()
      
    # generat the slice objects sl0, sl2, sl2
    self.recalculate_slices()
   
    # set up the figure with the images
    if width == None: width = min(12,24/len(self.vols))
    fig_asp = self.n_vols*max(self.shape[self.iy], 
               self.shape[self.iz]) / (2*self.shape[self.ix] + self.shape[self.iy])

    self.fig, self.ax = py.subplots(self.n_vols, 3, figsize = (width,width*fig_asp), squeeze = False)
    self.axes         = self.fig.get_axes()

    self.imgs = []

    for i in range(self.n_vols):
      im0 = np.squeeze(self.vols[i][tuple(self.sl0)].T)
      im1 = np.squeeze(np.flip(self.vols[i][tuple(self.sl1)].T,0))
      im2 = np.squeeze(np.flip(self.vols[i][tuple(self.sl2)].T,0))
 
      tmp  = []
      tmp.append(self.ax[i,0].imshow(im0, aspect=voxsize[1]/voxsize[0], **self.imshow_kwargs[i]))
      tmp.append(self.ax[i,1].imshow(im1, aspect=voxsize[2]/voxsize[0], **self.imshow_kwargs[i]))
      tmp.append(self.ax[i,2].imshow(im2, aspect=voxsize[2]/voxsize[1], **self.imshow_kwargs[i]))

      self.imgs.append(tmp)

      self.ax[i,0].set_axis_off()
      self.ax[i,1].set_axis_off()    
      self.ax[i,2].set_axis_off()
    
    self.ax[0,0].set_title(str(self.sl_z) + self.fstr, fontsize='small') 
    self.ax[0,1].set_title(str(self.sl_y) + self.fstr, fontsize='small') 
    self.ax[0,2].set_title(str(self.sl_x) + self.fstr, fontsize='small') 

    # connect the image figure with actions
    self.fig.canvas.mpl_connect('scroll_event',self.onscroll)
    self.fig.canvas.mpl_connect('button_press_event',self.onbuttonpress)
    self.fig.canvas.mpl_connect('key_press_event', self.onkeypress)

    if rowlabels is None:
      self.fig.subplots_adjust(left=0,right=1,bottom=0,top=0.97,wspace=0.01,hspace=0.01)

    else:
      for ivol, label in enumerate(rowlabels):
        self.fig.text(0.01,1 - (ivol + 0.5)/self.n_vols, label, rotation='vertical', 
                      size = 'large', verticalalignment = 'center')
      self.fig.subplots_adjust(left=0.03,right=1,bottom=0,top=0.97,wspace=0.01,hspace=0.01)
    self.fig.show()

    # set up figure for colorbar and ...
    self.fig_cb, self.ax_cb = py.subplots(self.n_vols, 1, figsize = (1,width*fig_asp), squeeze = False)
    self.fig_cb.subplots_adjust(left=0,right=0.3,bottom=0.02,top=0.96,wspace=0.05,hspace=0.05)
    self.fig_cb.show()

    cbmax     = 255
    cbar_size = 20
    cbarr     = np.outer(np.arange((int(cbar_size*cbmax))),np.ones(cbmax))

    for i in range(self.n_vols): 
      cb = self.ax_cb[i,0].imshow(cbarr, origin = 'lower', cmap = self.imshow_kwargs[i]['cmap'])
      self.ax_cb[i,0].get_xaxis().set_ticks([])
      self.ax_cb[i,0].get_xaxis().set_ticklabels([])
      self.ax_cb[i,0].get_yaxis().set_ticks([0,cbmax*cbar_size])
      self.ax_cb[i,0].get_yaxis().set_ticklabels(['%.1e' % self.imshow_kwargs[i]['vmin'],
                                                '%.1e' % self.imshow_kwargs[i]['vmax']])
      self.ax_cb[i,0].yaxis.tick_right()


    # add cross hair
    self.l0x = [] 
    self.l0y = []
    self.l1x = []
    self.l1y = []
    self.l2x = []
    self.l2y = []
   
    self.showCross = True
    if ls == '': self.showCross = False

    if self.showCross:
      for i in range(self.n_vols):
        self.l0x.append(self.axes[3*i + 0].axvline(self.sl_x, color = 'r',ls = ls)) 
        self.l0y.append(self.axes[3*i + 0].axhline(self.sl_y, color = 'r',ls = ls))

        self.l1x.append(self.axes[3*i + 1].axvline(self.sl_x, color = 'r',ls = ls))
        self.l1y.append(self.axes[3*i + 1].axhline(self.shape[self.iz] - self.sl_z, color = 'r',ls = ls))

        self.l2x.append(self.axes[3*i + 2].axvline(self.sl_y, color = 'r',ls = ls))
        self.l2y.append(self.axes[3*i + 2].axhline(self.shape[self.iz] - self.sl_z, color = 'r',ls = ls))

    # set up figure for sliders
    self.fig_sl, self.ax_sl = py.subplots(2*self.n_vols, 1, figsize = (4,3), squeeze = False)

    self.vmin_boxes = []
    self.vmax_boxes = []

    for i in range(self.n_vols):
      self.vmin_boxes.append(TextBox(self.ax_sl[2*i,0], 'vmin', 
                                     initial = str(self.imshow_kwargs[i]['vmin'])))
      self.vmax_boxes.append(TextBox(self.ax_sl[2*i+1,0], 'vmax',
                                     initial = str(self.imshow_kwargs[i]['vmax'])))

      self.vmin_boxes[-1].on_submit(self.update_vmin)
      self.vmax_boxes[-1].on_submit(self.update_vmax)

    self.fig_sl.show()

  #------------------------------------------------------------------------
  def update_vmin(self,text):
    for i in range(self.n_vols):
      self.imshow_kwargs[i]['vmin'] = float(self.vmin_boxes[i].text)
      self.imgs[i][0].set_clim([self.imshow_kwargs[i]['vmin'], self.imshow_kwargs[i]['vmax']])
      self.imgs[i][1].set_clim([self.imshow_kwargs[i]['vmin'], self.imshow_kwargs[i]['vmax']])
      self.imgs[i][2].set_clim([self.imshow_kwargs[i]['vmin'], self.imshow_kwargs[i]['vmax']])
      self.ax_cb[i,0].get_yaxis().set_ticklabels(['%.1e' % self.imshow_kwargs[i]['vmin'],
                                                '%.1e' % self.imshow_kwargs[i]['vmax']])
      self.fig.canvas.draw()
      self.fig_cb.canvas.draw()

  #------------------------------------------------------------------------
  def update_vmax(self,text):
    for i in range(self.n_vols):
      self.imshow_kwargs[i]['vmax'] = float(self.vmax_boxes[i].text)
      self.imgs[i][0].set_clim([self.imshow_kwargs[i]['vmin'], self.imshow_kwargs[i]['vmax']])
      self.imgs[i][1].set_clim([self.imshow_kwargs[i]['vmin'], self.imshow_kwargs[i]['vmax']])
      self.imgs[i][2].set_clim([self.imshow_kwargs[i]['vmin'], self.imshow_kwargs[i]['vmax']])
      self.ax_cb[i,0].get_yaxis().set_ticklabels(['%.1e' % self.imshow_kwargs[i]['vmin'],
                                                '%.1e' % self.imshow_kwargs[i]['vmax']])
      self.fig.canvas.draw()
      self.fig_cb.canvas.draw()

  #------------------------------------------------------------------------
  def redraw_transversal(self):
    for i in range(self.n_vols):
      self.imgs[i][0].set_data(np.squeeze(self.vols[i][tuple(self.sl0)].T))
    self.ax[0,0].set_title(str(self.sl_z) + self.fstr,fontsize='small') 
    if self.showCross:
        for l in self.l0x: l.set_xdata(self.sl_x) 
        for l in self.l0y: l.set_ydata(self.sl_y) 
    py.draw()

  #------------------------------------------------------------------------
  def redraw_coronal(self):
    for i in range(self.n_vols):
      self.imgs[i][1].set_data(np.squeeze(np.flip(self.vols[i][tuple(self.sl1)].T,0)))
    self.ax[0,1].set_title(str(self.sl_y) + self.fstr,fontsize='small') 
    if self.showCross:
        for l in self.l1x: l.set_xdata(self.sl_x) 
        for l in self.l1y: l.set_ydata(self.shape[self.iz] - self.sl_z - 1) 
    py.draw()

  #------------------------------------------------------------------------
  def redraw_sagittal(self):
    for i in range(self.n_vols):
      self.imgs[i][2].set_data(np.squeeze(np.flip(self.vols[i][tuple(self.sl2)].T,0)))
    self.ax[0,2].set_title(str(self.sl_x) + self.fstr,fontsize='small') 
    if self.showCross:
        for l in self.l2x: l.set_xdata(self.sl_y) 
        for l in self.l2y: l.set_ydata(self.shape[self.iz] - self.sl_z - 1) 
    py.draw()

  #------------------------------------------------------------------------
  def redraw(self):
    self.redraw_transversal()
    self.redraw_coronal()
    self.redraw_sagittal()

  #------------------------------------------------------------------------
  def recalculate_slices(self):
    if self.ndim == 3:
      self.sl0          = [slice(None)]*self.ndim
      self.sl0[self.iz] = slice(self.sl_z, self.sl_z+1)
      self.sl1          = [slice(None)]*self.ndim
      self.sl1[self.iy] = slice(self.sl_y, self.sl_y+1)
      self.sl2          = [slice(None)]*self.ndim
      self.sl2[self.ix] = slice(self.sl_x, self.sl_x+1)
    elif self.ndim == 4:
      self.sl0          = [slice(None)]*self.ndim
      self.sl0[self.iz] = slice(self.sl_z, self.sl_z+1)
      self.sl0[0]       = slice(self.sl_t, self.sl_t+1)
      self.sl1          = [slice(None)]*self.ndim
      self.sl1[self.iy] = slice(self.sl_y, self.sl_y+1)
      self.sl1[0]       = slice(self.sl_t, self.sl_t+1)
      self.sl2          = [slice(None)]*self.ndim
      self.sl2[self.ix] = slice(self.sl_x, self.sl_x+1)
      self.sl2[0]       = slice(self.sl_t, self.sl_t+1)
      self.fstr         = ', ' + str(self.sl_t)

  #------------------------------------------------------------------------
  def add_contour(self, N = 3, source = 0, target = 1, cmap = py.cm.autumn):
    for i in range(3):
      self.ax[target,i].contour(self.imgs[source][i].get_array(), N, cmap = cmap)
    self.fig.canvas.draw()

  #------------------------------------------------------------------------
  def remove_contour(self, target):
    for i in range(3):
      for col in self.ax[target,i].collections:
        col.remove()
      # a 2nd loop is needed to remove the last element (not clear why)
      for col in self.ax[target,i].collections:
        col.remove()

    self.fig.canvas.draw()

  #------------------------------------------------------------------------
  def onkeypress(self,event):
    if ((event.key == 'left' or event.key == 'right' or event.key == 'up' or event.key == 'down') and 
        (self.ndim >= 3)):
      if event.key == 'left' and self.ndim == 4:    
        self.sl_t = (self.sl_t - 1) % self.nframes 
        self.recalculate_slices()
        self.redraw()
      elif event.key == 'right' and self.ndim == 4: 
        self.sl_t = (self.sl_t + 1) % self.nframes 
        self.recalculate_slices()
        self.redraw()
      else:
        if event.inaxes in self.axes:
          iax = self.axes.index(event.inaxes)
          
          if (iax %3 == 0):
            if event.key == 'up':
                self.sl_z = (self.sl_z + 1) % self.shape[self.iz]
            elif event.key == 'down':
                self.sl_z = (self.sl_z - 1) % self.shape[self.iz]

            self.recalculate_slices()
            self.redraw()

          elif (iax %3 == 1):
            if event.key == 'up':
                self.sl_y = (self.sl_y + 1) % self.shape[self.iy]
            elif event.key == 'down':
                self.sl_y = (self.sl_y - 1) % self.shape[self.iy]

            self.recalculate_slices()
            self.redraw()

          elif (iax %3 == 2):
            if event.key == 'up':
                self.sl_x = (self.sl_x + 1) % self.shape[self.ix]
            elif event.key == 'down':
                self.sl_x = (self.sl_x - 1) % self.shape[self.ix]

            self.recalculate_slices()
            self.redraw()

  #------------------------------------------------------------------------
  def onbuttonpress(self,event):
    if py.get_current_fig_manager().toolbar.mode == '':
      if event.inaxes in self.axes:
        iax =  self.axes.index(event.inaxes)
        if iax < 3*self.n_vols:
          if iax % 3 == 0:
            self.sl_x = int(event.xdata) % self.shape[self.ix]
            self.sl_y = int(event.ydata) % self.shape[self.iy]
            self.recalculate_slices()
            self.redraw()
          elif iax % 3 == 1:
            self.sl_x = int(event.xdata) % self.shape[self.ix]
            self.sl_z = (self.shape[self.iz] - int(event.ydata)) % self.shape[self.iz]
            self.recalculate_slices()
            self.redraw()
          elif iax % 3 == 2:
            self.sl_y = int(event.xdata) % self.shape[self.iy]
            self.sl_z = (self.shape[self.iz] - int(event.ydata)) % self.shape[self.iz]
            self.recalculate_slices()
            self.redraw()

  #------------------------------------------------------------------------
  def onscroll(self,event):
    if event.inaxes in self.axes:
      iax = self.axes.index(event.inaxes)

      if (iax %3 == 0):
        if event.button == 'up':
            self.sl_z = (self.sl_z + 1) % self.shape[self.iz]
        elif event.button == 'down':
            self.sl_z = (self.sl_z - 1) % self.shape[self.iz]

        self.recalculate_slices()
        self.redraw()

      elif (iax %3 == 1):
        if event.button == 'up':
            self.sl_y = (self.sl_y + 1) % self.shape[self.iy]
        elif event.button == 'down':
            self.sl_y = (self.sl_y - 1) % self.shape[self.iy]

        self.recalculate_slices()
        self.redraw()

      elif (iax %3 == 2):
        if event.button == 'up':
            self.sl_x = (self.sl_x + 1) % self.shape[self.ix]
        elif event.button == 'down':
            self.sl_x = (self.sl_x - 1) % self.shape[self.ix]

        self.recalculate_slices()
        self.redraw()
