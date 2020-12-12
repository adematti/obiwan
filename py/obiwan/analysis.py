import os
import sys
import logging
import glob
import numpy as np
from scipy import special
from matplotlib import pyplot as plt
import fitsio
from .kenobi import find_file,get_randoms_id
from .catalog import BaseCatalog,SimCatalog
from . import utils

logger = logging.getLogger('obiwan.analysis')

class BaseImage(object):

    def read_image(self, fn, format='jpeg', xmin=None, ymin=None, **kwargs):
        """Extracted from https://github.com/legacysurvey/obiwan/blob/master/py/obiwan/qa/visual.py"""
        self.img_fn = fn
        if format in ['jpg','jpeg']:
            #import skimage.io
            #img = skimage.io.imread(fn)
            img = np.array(plt.imread(fn,**kwargs))
            for i in range(3):
                img[:,:,i] = np.rot90(img[:,:,i].T,1)
        elif format.startswith('fits'):
            img = fitsio.read(fn,**kwargs)
        else:
            raise ValueError('Unkown image format %s' % format)
        self.img = img
        self.xmin = xmin
        self.ymin = ymin

    def set_subimage(self,slicex=slice(0,None),slicey=slice(0,None)):
        """Zero-indexed."""
        self.xmin = self.xmin + slicex.start
        self.ymin = self.ymin + slicey.start
        self.img = self.img[slicey,slicex]

    @property
    def shape(self):
        if not hasattr(self,'img'): return self.baseshape
        return self.img.shape

    def isRGB(self):
        return len(self.img.shape) == 3

    @utils.saveplot()
    def plot(self, ax, cmap='gray', vmin=None, vmax=None, kwargs_range={}):
        if kwargs_range:
            vmin,vmax = Binning(samples=self.img.flatten(),nbins=1,**kwargs_range).range
        ax.imshow(self.img,interpolation='none',origin='lower',cmap=cmap,vmin=vmin,vmax=vmax)

class ImageAnalysis(BaseImage):

    baseshape = (3600,3600)

    def __init__(self, base_dir='.', brickname=None, kwargs_file={}):
        self.base_dir = base_dir
        self.brickname = brickname
        self.kwargs_file = kwargs_file

    def read_image(self,filetype='image-jpeg', band=['g','r','z'], xmin=0, ymin=0):
        format = 'jpeg' if 'jpeg' in filetype else 'fits'
        if format == 'fits' and not np.isscalar(band):
            assert len(band) == 3
            img = []
            for b in band:
                fn = find_file(self.base_dir,filetype,brickname=self.brickname,source='obiwan',band=b,**self.kwargs_file)
                super(ImageAnalysis,self).read_image(fn=fn,format=format,xmin=xmin,ymin=ymin)
                img.append(self.img)
            self.img = np.moveaxis(img,0,-1)
        else:
            fn = find_file(self.base_dir,filetype,brickname=self.brickname,source='obiwan',band=band,**self.kwargs_file)
            super(ImageAnalysis,self).read_image(fn=fn,format=format,xmin=xmin,ymin=ymin)

    def read_sources(self, filetype='randoms'):
        self.sources_fn = find_file(self.base_dir,filetype,brickname=self.brickname,source='obiwan',**self.kwargs_file)
        sources = SimCatalog(self.sources_fn)
        #from obiwan import BrickCatalog
        #bricks = BrickCatalog()
        #sources.bx,sources.by = bricks.get_xy_from_radec(sources.ra,sources.dec,brickname=sources.brickname)
        self.sources = sources[~sources.collided]

    def suggest_zooms(self, boxsize_in_pixels=None, match_in_degree=0.1/3600, range_observed_injected_in_degree=[5./3600,30./3600]):
        fn = find_file(self.base_dir,'tractor',brickname=self.brickname,source='obiwan',**self.kwargs_file)
        #print(fn)
        tractor = SimCatalog(fn)
        index_sources,index_tractor,distance = self.sources.match_radec(tractor,radius_in_degree=range_observed_injected_in_degree[-1],nearest=False,return_distance=True)
        matched_sources = np.unique(index_sources[distance<match_in_degree])
        mask_matched = np.in1d(index_sources,matched_sources)
        if not mask_matched.any():
            raise ValueError('No match found between random and Tractor catalogs.')
        mask_inrange = mask_matched & (distance > range_observed_injected_in_degree[0]) #& (distance < range_observed_injected_in_degree[-1])
        if not mask_inrange.any():
            raise ValueError('Not random/tractor pair found within range = %s, you should try larger range' % range_observed_injected_in_degree)
        index_sources = index_sources[mask_inrange]
        if boxsize_in_pixels is None: boxsize_in_pixels = np.hypot(*self.shape)/20.
        halfsize = round(boxsize_in_pixels//2)
        bx,by = np.rint(self.sources.bx[index_sources]).astype(int)-self.xmin,np.rint(self.sources.by[index_sources]).astype(int)-self.ymin
        rangex = bx-halfsize,bx+halfsize+1
        rangey = by-halfsize,by+halfsize+1
        mask_boxsize = (rangex[0]>=0) & (rangex[-1]<=self.shape[1]) & (rangey[0]>=0) & (rangey[-1]<=self.shape[0])
        if not mask_boxsize.any():
            raise ValueError('boxsize too large')
        rangex = tuple(r[mask_boxsize] for r in rangex)
        rangey = tuple(r[mask_boxsize] for r in rangey)
        toret = [(slice(rangex[0][i],rangex[1][i]),slice(rangey[0][i],rangey[1][i])) for i in range(mask_boxsize.sum())]
        return toret

    def plot_sources(self, ax, radius_in_pixel=3./0.262, dr=None, color='r'):
        from matplotlib.patches import Wedge
        from matplotlib.collections import PatchCollection
        if dr is None: dr = radius_in_pixel/20.
        patches = [Wedge((x-self.xmin,y-self.ymin),r=radius_in_pixel,theta1=0,theta2=360,width=dr) for x,y in zip(self.sources.bx,self.sources.by)]
        coll = PatchCollection(patches,color=color) #,alpha=1)
        ax.add_collection(coll)

class RunCatalog(BaseCatalog):

    @staticmethod
    def read_bricklist(bricklist):
        toret = []
        for brickname in bricklist:
            if os.path.isfile(brickname):
                logger.info('Reading brick list %s' % brickname)
                with open(brickname,'r') as file:
                    for line in file:
                        brickname = line.replace('\n','')
                        if brickname not in toret:
                            toret.append(brickname)
            else:
                toret.append(brickname)
        return toret

    @staticmethod
    def get_input_parser(parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()
        for key,default in zip(get_randoms_id.keys(),get_randoms_id.defs()):
            parser.add_argument('--%s' % key, nargs='*', type=int, default=default, help='Use these %ss.' % key)
        parser.add_argument('--brick', nargs='*', type=str, default=None, help='Use these bricknames. Can be a brick list file.')
        parser.add_argument('--list', nargs='*', type=str, default=None, help='Use these run lists. Overrides all other run arguments')

    @staticmethod
    def get_output_parser(parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--outdir', dest='output_dir', help='Output base directory, default "."')
        #parser.add_argument('--survey-dir', type=str, default=None, help='Override the $LEGACY_SURVEY_DIR environment variable')
        for key in get_randoms_id.keys():
            parser.add_argument('--%s' % key, nargs='*', type=int, default=None, help='If provided, restrict to these %ss.' % key)
        parser.add_argument('--brick', nargs='*', type=str, default=None, help='If provided, restrict to these bricknames. Can be a brick list file.')
        parser.add_argument('--list', nargs='*', type=str, default=None, help='Restrict to these run lists. Overrides all other run arguments')

    @classmethod
    def from_input_cmdline(cls, opt):
        if not isinstance(opt,dict):
            opt = vars(opt)
        for key in ['list','brick']:
            if key not in opt: opt[key] = None
        if opt['list'] is not None:
            return cls.from_list(opt['list'])
        bricknames = cls.read_bricklist(opt['brick'])
        kwargs_files = []
        args = np.atleast_2d(np.array([opt[key] for key in get_randoms_id.keys()]).T)
        for arg in args:
            kwargs_files.append({key:val for key,val in zip(get_randoms_id.keys(),arg)})
        return cls.from_brick_ranid(bricknames=bricknames, kwargs_files=kwargs_files)

    @classmethod
    def from_output_cmdline(cls, opt, force_from_disk=False, filetype='tractor'):

        if not isinstance(opt,dict):
            opt = vars(opt)
        for key in ['list','brick'] + get_randoms_id.keys():
            if key not in opt: opt[key] = None
        if (not force_from_disk)\
            and ((opt['list'] is not None)\
                or (opt['brick'] is not None) and all([opt[key] is not None for key in get_randoms_id.keys()])):
            return cls.from_input_cmdline(opt)

        bricknames = None
        if opt['brick'] is not None:
            bricknames = cls.read_bricklist(opt['brick'])

        def decode_output_fn(dirname):
            kwargs_file = get_randoms_id.split(os.path.basename(dirname))
            dirname = os.path.dirname(dirname)
            brickname = os.path.basename(dirname)
            return {**{self.fields[0]:brickname},**kwargs_file}

        if bricknames is None:
            fns = glob.iglob(os.path.join(opt['output_dir'],filetype,'*','*','*'))
        else:
            fns = []
            for brickname in bricknames:
                fns.append(glob.iglob(os.path.join(opt['output_dir'],filetype,brickname[:3],brickname,'*')))
            import itertools
            fns = itertools.chain(*fns) # chain iterators

        self = cls()
        for field in self.fields:
            self.set(field,[])

        for fn in fns:
            decode = decode_output_fn(fn)
            for field in self.fields:
                self.get(field).append(decode[field])
        self.to_np_arrays()

        # restrictions
        mask = self.trues()
        if opt['list'] is not None:
            other = cls.from_dict(opt['list'])
            mask &= self.in1d(other)
        if bricknames is not None:
            mask &= np.in1d(self.brickname,bricknames)
        for key in get_randoms_id.keys():
            if opt[key] is not None:
                mask &= self.get(key) == opt[key]

        return self[mask]

    @classmethod
    def from_brick_ranid(cls, bricknames=[], kwargs_files={}):
        if np.isscalar(bricknames): bricknames = [bricknames]
        if isinstance(kwargs_files,dict): kwargs_files = [kwargs_files]
        self = cls()
        for field in self.fields: self.set(field,[])
        for brickname in bricknames:
            for kwargs_file in kwargs_files:
                tmp = {**{self.fields[0]:brickname},**kwargs_file}
                for field in self.fields:
                    self.get(field).append(tmp[field])
        self.to_np_arrays()
        return self

    @classmethod
    def from_catalog(cls, cat):
        self = cls()
        for field in self.fields: self.set(field,np.array(cat.get(field)))
        return self.prune()

    @property
    def fields(self):
        return ['brickname'] + get_randoms_id.keys()

    def unique(self, field):
        return np.unique(self.get(field))

    def uniqid(self):
        uniqid = []
        for run in self:
            uniqid.append('-'.join([str(run.get(field)) for field in self.fields]))
        return np.array(uniqid)

    def prune(self):
        uniqid = self.uniqid()
        indices = np.unique(uniqid,return_index=True)[1]
        return self[indices]

    def in1d(self,other):
        selfid,otherid = self.uniqid(),other.uniqid()
        return np.in1d(selfid,otherid)

    def __iter__(self):
        for run in super(RunCatalog,self).__iter__():
            run.kwargs_file = {key:run.get(key) for key in get_randoms_id.keys()}
            yield run

    def iter_mask(self, cat):
         for run in self:
             yield np.all([cat.get(field) == run.get(field) for field in self.fields],axis=0)

    def iter_index(self, cat):
        for mask in self.iter_mask(cat):
            yield np.flatnonzero(mask)

    def count_runs(self, cat):
        return sum(mask.any() for mask in self.iter_mask(cat))

    def write_list(self, fn):
        """
        Write run list to ``fn``.

        Parameters
        ----------
        fn : string
            Path to run list.
        """
        utils.mkdir(os.path.dirname(fn))
        template = '%s.<8'*len(self.fields) + '\n'
        with open(fn,'w') as file:
            file.write(template % self.fields)
            for run in self:
                file.write(template % (run.get(field) for field in self.fields))

    @classmethod
    def from_list(cls, fns):
        self = cls()
        for field in self.fields: self.set(field,[])
        if np.isscalar(fns):
            fns = [fns]
        for fn in fns:
            with open(fn,'w') as file:
                for iline,line in enumerate(file):
                    if iline == 0:
                        header = line.split()
                        assert header == self.fields, 'Wrong header: %s, expected: %s' % (header,self.fields)
                    for field,val in zip(self.fields,line.split()):
                        self.get(field).append(val)
        self.to_np_arrays()

class BaseAnalysis(object):

    """Class to load, merge and save Obiwan products."""

    def __init__(self, base_dir='.', runcat=None, bricknames=[], kwargs_files={}, cats_dir=None, save_fn=None):
        self.base_dir = base_dir
        if runcat is None:
            self.runcat = RunCatalog.from_brick_ranid(bricknames=bricknames,kwargs_files=kwargs_files)
        else:
            self.runcat = runcat
        self.cats = {}
        self.cats_fn = {}
        self.cats_dir = cats_dir
        self.save_fn = save_fn

    def get_key(self, filetype='tractor', source='obiwan'):
        return '%s_%s' % (source,filetype.replace('-','_'))

    def merge_catalogs(self, filetype='tractor', base_dir=None, source='obiwan', keep_columns=None, set=False, write=False, **kwargs_write):
        if base_dir is None: base_dir = self.base_dir
        if (not write) and (not set):
            logger.warning('Nothing to be done with loaded files. Escaping.')
            return
        cat = 0
        def read_catalog(fn,**kwargs):
            if os.path.isfile(fn):
                return SimCatalog(fn,**kwargs)
            logger.warning('File %s not found.' % fn)
            return None
        for run in self.runcat:
            if filetype in ['ps','ps-events']:
                fn = find_file(base_dir,'ps',brickname=run.brickname,source='obiwan',**run.kwargs_file)
                tmp = read_catalog(fn,ext=1)
                if tmp is None: continue
                tf = tmp.unixtime.max()
                events = read_catalog(fn,ext=2)
                ti = events.unixtime.min()
                if filetype == 'ps-events':
                    tmp = events
                else:
                    tmp.mid = tmp.full(tmp._header['PPID'])
                tmp.unixti = tmp.full(ti)
                tmp.unixtf = tmp.full(tf)
                tmp.brickname = tmp.full(run.brickname)
            else:
                fn = find_file(base_dir,filetype,brickname=run.brickname,source=source,**run.kwargs_file)
                tmp = read_catalog(fn)
                if tmp is None: continue
            for key,val in run.kwargs_file.items():
                tmp.set(key,tmp.full(val))
            if filetype == 'randoms':
                tmp.cut(~tmp.collided)
            if keep_columns is not None:
                tmp.keep_columns(*keep_columns)
            cat += tmp
        key = self.get_key(filetype=filetype,source=source)
        if write:
            self.write_catalog(key=key,cat=cat,**kwargs_write)
        if set:
            self.cats[key] = cat
        return cat

        #setattr(filetype.replace('-','_'),cat)

    def set_cat_fn(self, key=None, cats_dir=None, cat_base=None, cat_fn=None, **kwargs_key):
        if key is None:
            key = self.get_key(**kwargs_key)
        if cat_fn is None:
            if cats_dir is not None:
                self.cats_dir = cats_dir
            if cat_base is not None:
                cat_fn = os.path.join(self.cats_dir,cat_base)
        if cat_fn is not None:
            self.cats_fn[key] = cat_fn
        return key

    def write_catalog(self, cat=None, **kwargs):
        key = self.set_cat_fn(**kwargs)
        if cat is None: cat = self.cats[key]
        cat.writeto(self.cats_fn[key])

    def read_catalog(self, set=True, **kwargs):
        key = self.set_cat_fn(**kwargs)
        cat = SimCatalog(self.cats_fn[key])
        if set: self.cats[key] = cat
        return cat

    def get(self, *args, **kwargs):
        return getattr(self,*args,**kwargs)

    def set(self, *args, **kwargs):
        return setattr(self,*args,**kwargs)

    def has(self, *args, **kwargs):
        return hasattr(self,*args,**kwargs)

    def getstate(self):
        state = {}
        for key in ['base_dir','cats_fn','cats_dir']:
            state[key] = self.get(key)
        state['runcat'] = self.runcat.to_dict()
        return state

    def setstate(self, state):
        self.cats = {}
        for key in state:
            self.set(key,state[key])
        self.runcat = RunCatalog.from_dict(self.runcat)

    def save(self, save_fn=None):
        if save_fn is not None: self.save_fn = save_fn
        logger.info('Saving %s to %s.' % (self.__class__.__name__,self.save_fn))
        utils.mkdir(os.path.dirname(self.save_fn))
        np.save(self.save_fn,self.getstate())

    @classmethod
    def load(cls, save_fn):
        state = np.load(save_fn,allow_pickle=True)[()]
        self = object.__new__(cls)
        self.setstate(state)
        return self

    def set_catalog(self, name, filetype=None, source=None, **kwargs_merge):
        key = self.get_key(filetype=filetype,source=source)
        if key in self.cats:
            pass
        elif key in self.cats_fn:
            self.read_catalog(key=key)
        else:
            self.merge_catalogs(filetype=filetype,source=source,set=True,**kwargs_merge)
        self.set(name,self.cats[key])

class RessourceEventAnalysis(BaseAnalysis):

    sorted_events = ['start', 'stage_tims: starting', 'stage_tims: starting calibs', 'stage_tims: starting read_tims', 'stage_tims: done read_tims',
        'stage_refs: starting', 'stage_outliers: starting', 'stage_halos: starting', 'stage_srcs: starting', 'stage_srcs: detection maps', 'stage_srcs: sources', 'stage_srcs: SED-matched',
        'stage_fitblobs: starting', 'stage_coadds: starting', 'stage_coadds: model images', 'stage_coadds: coadds', 'stage_coadds: extras', 'stage_writecat: starting']

    def process_events(self, events=None, time='reltime', statistic='mean'):
        """Assumes events are recorded at the beginning of the step."""
        self.set_catalog(name='events',filetype='ps-events')

        def get_sorted_events():
            sorted_events = np.array(self.sorted_events)
            return sorted_events[np.in1d(sorted_events,self.events.event)]

        if events is None:
            events = get_sorted_events()
        elif isinstance(events,str) and events == 'start':
            uniques = get_sorted_events()
            #print(uniques.tolist())
            base = uniques[0].split(':')[0]
            events_sel = [uniques[0]]
            for event in uniques:
                if event.startswith(base): continue
                base = event.split(':')[0]
                events_sel.append(event)
            events = np.asarray(events_sel)
        if time == 'reltime':
            indices = self.events.event
            values = self.events.unixtime - self.events.unixti
        elif time == 'steptime':
            indices = self.events.event
            values = self.events.nans()
            for mask in self.runcat.iter_mask(self.events):
                events_run = self.events[mask]
                dt = np.zeros(events_run.size,dtype='f4')
                tf = events_run.unixtf[0]
                for event in events[::-1]:
                    mask_event = events_run.event == event
                    if mask_event.any():
                        tmp = events_run.unixtime[mask_event]
                        dt[mask_event] = tf - tmp
                        tf = tmp[0]
                    #if event == 'start' and (tf - tmp[0])>400.: print(events_run.event)
                values[mask] = dt
        else:
            raise ValueError('Unknown requested time %s' % time)

        uniques,indices,inverses = np.unique(indices,return_index=True,return_inverse=True)
        uniqid = np.arange(len(uniques))
        edges = np.concatenate([uniqid,[uniqid[-1]+1]])
        import scipy
        stats = scipy.stats.binned_statistic(uniqid[inverses],values,statistic=statistic,bins=edges)[0]
        stats = stats[np.searchsorted(uniques,events)]

        return events,stats

    def plot_bar(self, ax, events='start', label_entries=True, kwargs_bar={}):
        kwargs_bar = {**{'align':'center','alpha':0.5,'ecolor':'black','capsize':2},**kwargs_bar}
        names,mean = self.process_events(events=events,time='steptime',statistic='mean')
        std = self.process_events(events=names,time='steptime',statistic='std')[1]
        labels = []
        for name in names:
            label = name
            if events in ['start']: label = name.split(':')[0]
            labels.append(label.replace('stage_',''))
        labels = np.array(labels)
        ax.yaxis.grid(True)
        ax.bar(labels,mean,yerr=std,**kwargs_bar)
        ax.set_xticklabels(labels=labels,rotation=40,ha='right')
        ax.set_ylabel('Average wall time [s]')
        if label_entries:
            label = '%d entries' % self.runcat.count_runs(self.events)
            ax.text(0.05,0.95,label,horizontalalignment='left',verticalalignment='top',transform=ax.transAxes,color='k')

class RessourceAnalysis(RessourceEventAnalysis):

    @staticmethod
    def process_one_series(series,quantities=['proc_icpu','vsz']):
        steps,index_steps = np.unique(series.step, return_index=True)
        map_steps = np.zeros(steps.max()+1,dtype='i4')
        map_steps[steps] = np.arange(len(steps))
        reltime = series.unixtime[index_steps] - series.unixti[index_steps]
        mid = series.mid[0]

        toret = {'time':reltime}
        for q in quantities:
            toret[q] = {}
            toret[q]['ps'] = np.zeros(len(steps),dtype='f4')
            toret[q]['others'] = np.zeros(len(steps),dtype='f4')
            toret[q]['main'] = np.zeros(len(steps),dtype='f4')
            toret[q]['workers'] = []

        mask_main = series.pid == mid
        mask_mine = mask_main | (series.ppid == mid)
        pids = np.unique(series.pid[mask_mine])

        for pid in pids:
            index_pid = np.flatnonzero(series.pid == pid)
            step_pid = map_steps[series.step[index_pid]]
            cmds = np.unique(series.command[index_pid])
            if len(cmds) == 1 and cmds[0].startswith('ps ax'):
                for q in quantities:
                    toret[q]['ps'][step_pid] += series.get(q)[index_pid]
            else:
                for q in quantities:
                    tmp = np.zeros(len(steps),dtype='f4')
                    tmp[step_pid] = series.get(q)[index_pid]
                    if pid == mid: toret[q]['main'] = tmp
                    else: toret[q]['workers'].append(tmp)
        for q in quantities: toret[q]['workers'] = np.array(toret[q]['workers'])

        pids = np.unique(series.pid[~mask_mine])
        for pid in pids:
            index_pid = np.flatnonzero(series.pid == pid)
            step_pid = map_steps[series.step[index_pid]]
            for q in quantities:
                toret[q]['others'][step_pid] += series.get(q)[index_pid]

        return toret

    def process_all_series(self,quantities=['proc_icpu','vsz']):
        self.set_catalog(name='series',filetype='ps')
        toret = {q:{} for q in quantities}
        toret['time'] = []
        for mask in self.runcat.iter_mask(self.series):
            series = self.series[mask]
            qseries = self.process_one_series(series,quantities=quantities)
            for q in quantities:
                for key,val in qseries[q].items():
                    if key not in toret[q]: toret[q][key] = []
                    if key == 'workers':
                        val = val.max(axis=0)
                    toret[q][key].append(val)
            toret['time'].append(qseries['time'])
        time_range = (0.,max(time.max() for time in toret['time']))
        num = max(len(time) for time in toret['time'])
        time = np.linspace(*time_range,num=num)
        for q in quantities:
            for key,val in toret[q].items():
                toret[q][key] = np.mean([np.interp(time,t,v) for t,v in zip(toret['time'],val)],axis=0)
        toret['time'] = time
        return toret

    def add_event_vlines(self, ax, events=None):
        names,times = self.process_events(events=events)
        for ievent,(name,time) in enumerate(zip(names,times)):
            ax.axvline(time,color='k',alpha=0.1)
            label = name
            if events in ['start']: label = name.split(':')[0]
            label = label.replace('stage_','')
            ycoord = [0.05,0.35,0.65][ievent % 3]
            ax.text(time,ycoord,label,rotation='vertical',horizontalalignment='left',
                    verticalalignment='bottom',transform=ax.get_xaxis_transform(),color='k')

    @utils.saveplot()
    def plot_one_series(self, ax, series=None, events='start', ids=['main','workers'], kwargs_fig={'figsize':(10,5)}, kwargs_plot={}):
        if series is None:
            self.set_catalog(name='series',filetype='ps')
            series = self.series
        series = self.process_one_series(series=series,quantities=['proc_icpu','vsz'])
        if ids is None: ids = list(series['proc_icpu'].keys())
        colors = kwargs_plot.pop('colors',plt.rcParams['axes.prop_cycle'].by_key()['color'])
        alpha = kwargs_plot.pop('alpha',1.)
        ax2 = ax.twinx()
        for key,color in zip(ids,colors):
            if key in ['workers']:
                alpha_ = 0.25*alpha
                cpu,vsz = series['proc_icpu'][key],series['vsz'][key]
            else:
                alpha_ = alpha
                cpu,vsz = [series['proc_icpu'][key]],[series['vsz'][key]]
            for iwork,(cpu_,vsz_) in enumerate(zip(cpu,vsz)):
                ax.plot(series['time'],cpu_,label=key if iwork==0 else None,color=color,alpha=alpha_,**kwargs_plot)
                ax2.plot(series['time'],vsz_/1e6,color=color,alpha=alpha_,linestyle='--',**kwargs_plot)
        ax.set_xlabel('Wall time [s]')
        ax.set_ylabel('Proc [%]')
        ax2.set_ylabel('VSS [GB]')
        self.add_event_vlines(ax,events=events)
        ax.legend()

    @utils.saveplot()
    def plot_all_series(self, ax, events='start', ids=['main','workers'], label_entries=True, kwargs_plot={}):
        series = self.process_all_series(quantities=['proc_icpu','vsz'])
        if ids is None: ids = list(series['proc_icpu'].keys())
        colors = kwargs_plot.pop('colors',plt.rcParams['axes.prop_cycle'].by_key()['color'])
        ax2 = ax.twinx()
        for key,color in zip(ids,colors):
            ax.plot(series['time'],series['proc_icpu'][key],label=key,color=color,**kwargs_plot)
            ax2.plot(series['time'],series['vsz'][key]/1e6,color=color,linestyle='--',**kwargs_plot)
        ax.set_xlabel('Wall time [s]')
        ax.set_ylabel('Proc [%]')
        ax2.set_ylabel('VSS [GB]')
        self.add_event_vlines(ax,events=events)
        ax.legend()
        if label_entries:
            label = '%d entries' % self.runcat.count_runs(self.events)
            ax.text(0.05,0.95,label,horizontalalignment='left',verticalalignment='top',transform=ax.transAxes,color='k')

class MatchAnalysis(BaseAnalysis):

    def getstate(self):
        state = super(MatchAnalysis,self).getstate()
        for key in ['add_input_tractor','injected','observed','distance']:
            if self.has(key):
                state[key] = self.get(key)
        for put in ['input','output']:
            for template in ['inter_%s','extra_%s','inter_%s_injected']:
                key = template % put
                if self.has(key):
                    state[key] = self.get(key)
        return state

    def setstate(self, state):
        super(MatchAnalysis,self).setstate(state)
        if self.has('add_input_tractor'):
            self.setup(add_input_tractor=state['add_input_tractor'])

    def setup(self, add_input_tractor=False):
        self.add_input_tractor = add_input_tractor
        self.set_catalog(name='input',filetype='randoms',source='obiwan')
        self.set_catalog(name='output',filetype='tractor',source='obiwan')
        if add_input_tractor:
            kwargs = {}
            if isinstance(add_input_tractor,str): kwargs = {'base_dir':add_input_tractor}
            self.set_catalog(name='input_tractor',filetype='tractor',source='legacypipe',**kwargs)
        self.injected = self.input.index()
        self.observed = np.array([],dtype=int)
        if add_input_tractor:
            self.observed = np.arange(self.input.size,self.input.size+self.input_tractor.size)
            self.input.fill(self.input_tractor,index_self='after')

    def match(self, radius_in_degree=1.5/3600., add_input_tractor=False):
        self.setup(add_input_tractor=add_input_tractor)
        self.inter_input,self.inter_output,self.distance = [],[],[]
        index_input,index_output = self.input.index(),self.output.index()
        #for field in self.runcat.fields: print(field,self.runcat.get(field))
        for mask_input,mask_output in zip(self.runcat.iter_mask(self.input),self.runcat.iter_mask(self.output)):
            mask_input[self.observed] = self.input.brickname[self.observed] == self.input.brickname[self.injected][0]
            #print(self.input.size,mask_input.sum(),self.output.size,mask_output.sum())
            inter_input,inter_output,distance = self.input[mask_input].match_radec(self.output[mask_output],nearest=True,radius_in_degree=radius_in_degree,return_distance=True)
            self.inter_input.append(index_input[mask_input][inter_input])
            self.inter_output.append(index_output[mask_output][inter_output])
            self.distance.append(distance)
        for key in ['inter_input','inter_output','distance']:
            self.set(key,np.concatenate(self.get(key)))

        logger.info('Matching %d objects / %d in input, %d in output' % (self.inter_input.size,self.input.size,self.output.size))
        mask_injected = np.in1d(self.inter_input,self.injected)
        for key in ['input','output']:
            self.set('extra_%s' % key,np.setdiff1d(self.get(key).index(),self.get('inter_%s' % key)))
            self.set('inter_%s_injected' % key,self.get('inter_%s' % key)[mask_injected])
        self.distance_injected = self.distance[mask_injected]
        logger.info('Matching %d injected objects / %d in input, %d in output' % (self.inter_input_injected.size,self.injected.size,self.output.size))

    def export(self, base='input', key_input='input', key_output=None, key_distance='distance', key_match='matched', key_injected='injected', injected=True, write=True, **kwargs_write):
        input = self.input.copy()
        if key_input:
            for field in input.fields: input.rename(field,'%s_%s' % (key_input,field))
        output = self.output.copy()
        if key_output:
            for field in output.fields: output.rename(field,'%s_%s' % (key_output,field))
        if key_distance is not None:
            for key,cat in zip(['input','output'],[input,output]):
                distance = cat.nans()
                distance[self.get('inter_%s' % key)] = self.distance
                cat.set(key_distance,distance)
        if key_match is not None:
            for key,cat in zip(['input','output'],[input,output]):
                match = cat.falses()
                match[self.get('inter_%s' % key)] = True
                cat.set(key_match,match)

        inter_input,inter_output = self.inter_input,self.inter_output
        extra_input,extra_output = self.extra_input,self.extra_output
        if injected:
            inter_input,inter_output = self.inter_input_injected,self.inter_output_injected
        if key_injected:
            mask_injected = input.falses()
            mask_injected[self.injected] = True
            input.set(key_injected,mask_injected)
        if base == 'input':
            cat = input
            cat.fill(output,index_self=inter_input,index_other=inter_output)
        elif base == 'output':
            cat = output
            cat.fill(input,index_self=inter_output,index_other=inter_input)
        elif base == 'inter':
            cat = input[inter_input]
            cat.fill(output,index_self=None,index_other=inter_output)
        elif base == 'extra':
            cat = input[extra_input]
            cat.fill(output,index_self='after',index_other=extra_output)
        elif base == 'all':
            cat = input[self.injected] if injected else input
            cat.fill(output,index_self='after',index_other=None)
        key = self.get_key(filetype='match_%s' % base,source='obiwan')
        if write:
            self.write_catalog(cat=cat,key=key,**kwargs_write)
        return cat

    @utils.saveplot()
    def plot_scatter(self, ax, field, injected=True, xlabel=None, ylabel=None,
                    square=False, regression=False, diagonal=False, label_entries=True,
                    kwargs_xlim={}, kwargs_ylim={}, kwargs_scatter={}, kwargs_regression={}, kwargs_diagonal={}):
        """
        Scatter plot of output v.s. input.

        Parameters
        ----------
        ax : plt.axes
            Where to plot.

        field : str
            Name of catalog column to plot.

        injected : bool, default=True
            If ``True``, restrict to injected sources. Else all (actual and injected) sources are considered.

        xlabel : string, default=None
            x label, if ``None``, defaults to ``'input_%s' % field``.

        ylabel : string, default=None
            y label, if ``None``, defaults to ``'output_%s' % field``.

        square : bool, default=False
            Whether to enforce square plot.

        regression : bool, default=False
            Whether to plot regression line.

        diagonal : bool, default=False
            Whether to plot diagonal line.

        label_entries : bool, default=True
            Whether to add the number of entries to the plot.

        kwargs_xlim : dict, default={}
            Arguments to ``Binning``, to define the x-range.

        kwargs_ylim : dict, default={}
            Arguments to ``Binning``, to define the y-range.

        kwargs_scatter : dict, default={}
            Arguments for ``plt.scatter()``.

        kwargs_regression : dict, default={}
            Arguments for ``plt.plot()`` regression line.

        kwargs_diagonal : dict, default={}
            Arguments for ``plt.plot()`` diagonal line.
        """
        kwargs_scatter = {**{'s':10,'marker':'.','alpha':1,'edgecolors':'none'},**kwargs_scatter}
        kwargs_diagonal = {**{'linestyle':'--','linewidth':2,'color':'k'},**kwargs_diagonal}
        kwargs_regression = {**{'linestyle':'--','linewidth':2,'color':'r','label':''},**kwargs_regression}
        label_regression = kwargs_regression.pop('label',None)

        if injected:
            inter_input,inter_output = self.inter_input_injected,self.inter_output_injected
        else:
            inter_input,inter_output = self.inter_input,self.inter_output

        values1 = self.input.get(field)[inter_input]
        values2 = self.output.get(field)[inter_output]

        if xlabel is None: xlabel = 'input_%s' % field
        if ylabel is None: ylabel = 'output_%s' % field

        if kwargs_xlim:
            xlim = Binning(samples=values1,nbins=1,**kwargs_xlim).range
            ax.set_xlim(xlim)

        if kwargs_ylim:
            ylim = Binning(samples=values2,nbins=1,**kwargs_ylim).range
            ax.set_ylim(ylim)

        ax.scatter(values1,values2,**kwargs_scatter)

        if square:
            xlim,ylim = ax.get_xlim(),ax.get_ylim()
            xylim = min(xlim[0],ylim[0]),max(xlim[1],ylim[1])
            ax.axis('square')
            ax.set_xlim(xylim)
            ax.set_ylim(xylim)

        xlim,ylim = [np.array(tmp) for tmp in [ax.get_xlim(),ax.get_ylim()]]
        try:
            a,b = np.polyfit(values1,values2,1)
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError('Regression failed, %s x-range = %.3f - %.3f, y-range = %.3f - %.3f'
                                        % (field,values1.min(),values1.max(),values2.min(),values2.max()))
        y = a*xlim + b
        r = np.corrcoef(values1,values2)[0,1]

        label_regression_ = label_regression
        if label_regression_ is not None:
            label_regression = '$\\rho = %.3f$' % r
            if label_regression_:
                label_regression = '%s %s' % (label_regression_,label_regression)
        else: label_regression = None

        if regression:
            ax.plot(xlim,y,label=label,**kwargs_regression)
            ax.set_ylim(ylim)
        elif label_regression:
            ax.text(0.95,0.05,label_regression,horizontalalignment='right',verticalalignment='bottom',transform=ax.transAxes,color='k')
        if diagonal:
            ax.plot(xlim,xlim,**kwargs_diagonal)
            ax.set_ylim(ylim)
        if label_entries:
            label = '%d entries' % len(values1)
            ax.text(0.05,0.95,label,horizontalalignment='left',verticalalignment='top',transform=ax.transAxes,color='k')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    @utils.saveplot()
    def plot_hist(self, ax, field, injected=True, xlabel=None, ylabel=None, divide_uncer=True,
                label_entries=True, label_mean_std=True, kwargs_xedges={}, kwargs_hist={}):
        if injected:
            inter_input,inter_output = self.inter_input_injected,self.inter_output_injected
        else:
            inter_input,inter_output = self.inter_input,self.inter_output
        values = self.output.get(field)[inter_output]-self.input.get(field)[inter_input]
        if divide_uncer:
            if 'flux' in field: field_ivar = field.replace('flux','flux_ivar')
            else: field_ivar = '%s_ivar' % field
            ivar = self.output.get(field_ivar)[inter_output]
            values = values*np.sqrt(ivar)
        if xlabel is None:
            xlabel = '\\Delta \\mathrm{%s}' % field.replace('_','\_')
            if divide_uncer: xlabel = '$%s \\times \\sqrt{\\mathrm{%s}}$' % (xlabel,field_ivar.replace('_','\_'))
            else: xlabel = '$%s$' % xlabel
        kwargs_hist = {**{'histtype':'step','color':'k'},**kwargs_hist}
        edges = Binning(samples=values,**kwargs_xedges).edges
        ax.hist(values,bins=edges,**kwargs_hist)
        ax.set_xlabel(xlabel)
        if ylabel is not None: ax.set_ylabel(ylabel)
        if label_entries:
            label = '%d entries' % len(values)
            ax.text(0.05,0.95,label,horizontalalignment='left',verticalalignment='top',transform=ax.transAxes,color='k')
        if label_mean_std:
            label = '$\mathrm{median} = %.2g$\n' % np.median(values)
            label += '$\mathrm{mean} = %.2g$\n' % np.mean(values)
            label +='$\mathrm{std} = %.2g$\n' % np.std(values,ddof=1)
            label +='$\mathrm{std(med)} = %.2g$' % estimate_std_outliers(values)
            ax.text(0.95,0.95,label,horizontalalignment='right',verticalalignment='top',transform=ax.transAxes,color='k')

"""Utilities."""

from scipy import stats

class Binning(object):

    def __init__(self,samples=None,weights=None,edges=None,nbins=10,range=None,quantiles=None,scale='linear'):
        self.edges = edges
        if edges is None:
            if range is None:
                if quantiles is None:
                    range = [samples.min(axis=-1),samples.max(axis=-1)]
                    range[-1] = range[-1]+(range[-1]-range[0])*1e-5
                else:
                    range = np.percentile(samples,q=np.array(quantiles)*100.,axis=-1).T
            if range[0] is None: range[0] = samples.min(-1)
            if range[-1] is None:
                range[-1] = samples.max(-1)
                range[-1] = range[-1]+(range[-1]-range[0])*1e-5
            if isinstance(nbins,np.integer):
                if scale == 'linear':
                    self.edges = np.linspace(range[0],range[-1],nbins+1)
                elif scale == 'log':
                    self.edges = np.logspace(np.log10(range[0]),np.log10(range[-1]),nbins+1,base=10)
                else:
                    raise ValueError('Scale {} is unkown.'.format(scale))
            else:
                self.edges = np.histogram_bin_edges(samples,bins=nbins,range=range,weights=weights)

    @property
    def range(self):
        return (self.edges[0],self.edges[-1])

    @property
    def nbins(self):
        return len(self.edges)-1

    @property
    def centers(self):
        return (self.edges[:-1]+self.edges[1:])/2.


def estimate_std_outliers(tab):
        # trick to estimate standard deviation in presence of outliers
        return np.median(np.abs(tab-np.median(tab)))/(2.**0.5*special.erfinv(1./2.))
