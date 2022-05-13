"""
=========
beamforming_helper
=========

beamforming_helper model template The System Development Kit
Used as a template for all TheSyDeKick Entities.

Current docstring documentation style is Numpy
https://numpydoc.readthedocs.io/en/latest/format.html

This text here is to remind you that documentation is important.
However, youu may find it out the even the documentation of this 
entity may be outdated and incomplete. Regardless of that, every day 
and in every way we are getting better and better :).

Initially written by Marko Kosunen, marko.kosunen@aalto.fi, 2017.

"""

import os
import sys
if not (os.path.abspath('../../thesdk') in sys.path):
    sys.path.append(os.path.abspath('../../thesdk'))

from thesdk import *

import numpy as np

class beamforming_helper(thesdk):
    @property
    def _classfile(self):
        return os.path.dirname(os.path.realpath(__file__)) + "/"+__name__

    def __init__(self,*arg): 
        self.print_log(type='I', msg='Inititalizing %s' %(__name__)) 
        self.proplist = [ 'Rs' ];    # Properties that can be propagated from parent
        self.Rs =  100e6;            # Sampling frequency
        self.IOS=Bundle()            # Pointer for input data
        self.IOS.Members['A']=IO()   # Pointer for input data
        self.IOS.Members['Z']= IO()
        self.model='py';             # Can be set externally, but is not propagated
        self.par= False              # By default, no parallel processing
        self.queue= []               # By default, no parallel processing

        if len(arg)>=1:
            parent=arg[0]
            self.copy_propval(parent,self.proplist)
            self.parent =parent;

        self.init()

    def init(self):
        pass #Currently nohing to add

    def main(self):
        '''Guideline. Isolate python processing to main method.
        
        To isolate the interna processing from IO connection assigments, 
        The procedure to follow is
        1) Assign input data from input to local variable
        2) Do the processing
        3) Assign local variable to output

        '''
        inval=self.IOS.Members['A'].Data
        out=inval
        if self.par:
            self.queue.put(out)
        self.IOS.Members['Z'].Data=out

    def run(self,*arg):
        '''Guideline: Define model depencies of executions in `run` method.

        '''
        if len(arg)>0:
            self.par=True      #flag for parallel processing
            self.queue=arg[0]  #multiprocessing.queue as the first argument
        if self.model=='py':
            self.main()

if __name__=="__main__":
    import matplotlib.pyplot as plt
    from  beamforming_helper import *
    from  beamforming_helper.controller import controller as beamforming_helper_controller
    import pdb
    import math
    length=1024
    rs=100e6
    indata=np.cos(2*math.pi/length*np.arange(length)).reshape(-1,1)

    models=[ 'py']
    duts=[]
    for model in models:
        d=beamforming_helper()
        duts.append(d) 
        d.model=model
        d.Rs=rs
        d.IOS.Members['A'].Data=indata
        d.init()
        d.run()

    for k in range(len(duts)):
        hfont = {'fontname':'Sans'}
        figure,axes=plt.subplots(2,1,sharex=True)
        x = np.arange(length).reshape(-1,1)
        axes[0].plot(x,indata)
        axes[0].set_ylim(-1.1, 1.1);
        axes[0].set_xlim((np.amin(x), np.amax(x)));
        axes[0].set_ylabel('Input', **hfont,fontsize=18);
        axes[0].grid(True)
        axes[1].plot(x, duts[k].IOS.Members['Z'].Data)
        axes[1].set_ylim(-1.1, 1.1);
        axes[1].set_xlim((np.amin(x), np.amax(x)));
        axes[1].set_ylabel('Output', **hfont,fontsize=18);
        axes[1].set_xlabel('Sample (n)', **hfont,fontsize=18);
        axes[1].grid(True)
        titlestr = "Myentity model %s" %(duts[k].model) 
        plt.suptitle(titlestr,fontsize=20);
        plt.grid(True);
        printstr="./inv_%s.eps" %(duts[k].model)
        plt.show(block=False);
        figure.savefig(printstr, format='eps', dpi=300);
    input()

