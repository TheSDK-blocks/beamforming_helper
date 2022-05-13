"""
========
Inverter
========

Inverter model template The System Development Kit
Used as a template for all TheSyDeKick Entities.

Current docstring documentation style is Numpy
https://numpydoc.readthedocs.io/en/latest/format.html

This text here is to remind you that documentation is iportant.
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
from rtl import *
from rtl.testbench import *
from rtl.testbench import testbench as vtb
from eldo import *
from eldo.testbench import *
from eldo.testbench import testbench as etb 

import numpy as np

class beamforming_helper(rtl,eldo,thesdk):
    @property
    def _classfile(self):
        return os.path.dirname(os.path.realpath(__file__)) + "/"+__name__

    def __init__(self,*arg): 
        self.print_log(type='I', msg='Inititalizing %s' %(__name__)) 
        self.proplist = [ 'Rs' ];    # Properties that can be propagated from parent
        self.Rs =  100e6;            # Sampling frequency
        self.vdd = 1.0
        self.IOS=Bundle()
        self.IOS.Members['A']=IO() # Pointer for input data
        self.IOS.Members['Z']= IO()
        self.model='py';             # Can be set externally, but is not propagated
        self.par= False              # By default, no parallel processing
        self.queue= []               # By default, no parallel processing
        self.IOS.Members['control_write']= IO() 
        # File for control is created in controller

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
        out=np.array(1-inval)
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
        else: 
          if self.model=='sv':
              # Verilog simulation options here
              _=rtl_iofile(self, name='A', dir='in', iotype='sample', ionames=['A'], datatype='sint') # IO file for input A
              _=rtl_iofile(self, name='Z', dir='out', iotype='sample', ionames=['Z'], datatype='sint')
              self.rtlparameters=dict([ ('g_Rs',self.Rs),]) #Defines the sample rate
              self.run_rtl()
          if self.model=='vhdl':
              # VHDL simulation options here
              _=rtl_iofile(self, name='A', dir='in', iotype='sample', ionames=['A']) # IO file for input A
              _=rtl_iofile(self, name='Z', dir='out', iotype='sample', ionames=['Z'], datatype='int')
              self.rtlparameters=dict([ ('g_Rs',self.Rs),]) #Defines the sample rate
              self.run_rtl()
          
          elif self.model=='eldo':
              _=eldo_iofile(self, name='A', dir='in', iotype='sample', ionames=['IN'], rs=self.Rs, \
                vhi=self.vdd, trise=1/(self.Rs*4), tfall=1/(self.Rs*4))
              _=eldo_iofile(self, name='Z', dir='out', iotype='event', sourcetype='V', ionames=['OUT'])

              # Saving the analog waveform of the input as well
              self.IOS.Members['A_OUT']= IO()
              _=eldo_iofile(self, name='A_OUT', dir='out', iotype='event', sourcetype='V', ionames=['IN'])
              #self.preserve_iofiles = True
              #self.preserve_eldofiles = True
              #self.interactive_eldo = True
              self.nproc = 1
              self.eldooptions = {
                          'eps': '1e-6'
                      }
              self.eldoparameters = {
                          'exampleparam': '0'
                      }
              self.eldoplotextras = ['v(IN)','v(OUT)']

              # Example of defining supplies (not used here because the example beamforming_helper has no supplies)
              #_=eldo_dcsource(self,name='dd',value=self.vdd,pos='VDD',neg='VSS',extract=True,ext_start=2e-9)
              #_=eldo_dcsource(self,name='ss',value=0,pos='VSS',neg='0')

              # Simulation command
              _=eldo_simcmd(self,sim='tran')
              self.run_eldo()

          if self.par:
              self.queue.put(self.IOS.Members[Z].Data)

          #Delete large iofiles
          del self.iofile_bundle 

    def define_io_conditions(self):
        '''This overloads the method is called by run_rtl method. It defines the read/write conditions for the files

        '''
        # Input A is read to verilog simulation after 'initdone' is set to 1 by controller
        self.iofile_bundle.Members['A'].verilog_io_condition='initdone'
        # Output is read to verilog simulation when all of the outputs are valid, 
        # and after 'initdone' is set to 1 by controller
        self.iofile_bundle.Members['Z'].verilog_io_condition_append(cond='&& initdone')


if __name__=="__main__":
    import matplotlib.pyplot as plt
    from  beamforming_helper import *
    from  beamforming_helper.controller import controller as beamforming_helper_controller
    import pdb
    length=1024
    rs=100e6
    indata=np.random.randint(2,size=length).reshape(-1,1);
    #indata=np.random.randint(2,size=length)
    controller=beamforming_helper_controller()
    controller.Rs=rs
    #controller.reset()
    #controller.step_time()
    controller.start_datafeed()

    models=[ 'py', 'sv', 'vhdl', 'eldo' ]
    duts=[]
    for model in models:
        d=beamforming_helper()
        duts.append(d) 
        d.model=model
        d.Rs=rs
        #d.interactive_rtl=True
        #d.interactive_eldo=True
        d.IOS.Members['A'].Data=indata
        d.IOS.Members['control_write']=controller.IOS.Members['control_write']
        d.init()
        d.run()

    # Obs the latencies may be different
    latency=[ 0 , 1, 1, 0 ]
    for k in range(len(duts)):
        hfont = {'fontname':'Sans'}
        if duts[k].model == 'eldo':
            figure,axes = plt.subplots(2,1,sharex=True)
            axes[0].plot(duts[k].IOS.Members['A_OUT'].Data[:,0],duts[k].IOS.Members['A_OUT'].Data[:,1],label='Input')
            axes[1].plot(duts[k].IOS.Members['Z'].Data[:,0],duts[k].IOS.Members['Z'].Data[:,1],label='Output')
            axes[0].set_ylabel('Input', **hfont,fontsize=18);
            axes[1].set_ylabel('Output', **hfont,fontsize=18);
            axes[1].set_xlabel('Time (s)', **hfont,fontsize=18);
            axes[0].set_xlim(0,11/rs)
            axes[1].set_xlim(0,11/rs)
            axes[0].grid(True)
            axes[1].grid(True)
        else:
            figure,axes=plt.subplots(2,1,sharex=True)
            x = np.linspace(0,10,11).reshape(-1,1)
            axes[0].stem(x,indata[0:11,0])
            axes[0].set_ylim(0, 1.1);
            axes[0].set_xlim((np.amin(x), np.amax(x)));
            axes[0].set_ylabel('Input', **hfont,fontsize=18);
            axes[0].grid(True)
            axes[1].stem(x, duts[k].IOS.Members['Z'].Data[0+latency[k]:11+latency[k],0])
            axes[1].set_ylim(0, 1.1);
            axes[1].set_xlim((np.amin(x), np.amax(x)));
            axes[1].set_ylabel('Output', **hfont,fontsize=18);
            axes[1].set_xlabel('Sample (n)', **hfont,fontsize=18);
            axes[1].grid(True)
        titlestr = "Inverter model %s" %(duts[k].model) 
        plt.suptitle(titlestr,fontsize=20);
        plt.grid(True);
        printstr="./inv_%s.eps" %(duts[k].model)
        plt.show(block=False);
        figure.savefig(printstr, format='eps', dpi=300);
    input()
