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
import matplotlib.pyplot as plt

class beamforming_helper(thesdk):
    @property
    def _classfile(self):
        return os.path.dirname(os.path.realpath(__file__)) + "/"+__name__

    def __init__(self,*arg): 
        self.print_log(type='I', msg='Inititalizing %s' %(__name__)) 
        #self.proplist = [ 'Rs' ];    # Properties that can be propagated from parent
        self.angular_resolution =  10;            # steps per degree
        self.speedoflight = 299792458 #m/s
        self.fc = 1e9 
        self.bw = 0.1e9 
        self.antenna_spacing_type = 'fmax' # 'fmax' or 'fc' 
        self.Nant = 16
        self.sub_array_size = 2
        self.target_angle_layer0 = [-50] # degrees
        self.target_angle_final = [40] # degrees
        self.inter_element_spacing = 0
        self.sign_vector = np.ones((self.Nant)) #select between beamforming and nulling



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
        self.angles = np.arange(-90,90,1/self.angular_resolution)
        self.calculate_inter_element_spacing()
        pass #Currently nohing to add

    def main(self): #simple usage example

        sub_array_size = self.Nant

        input_signals = self.calculate_input_signals()
        compensating_weights = self.calculate_compensating_weights() * self.sign_vector

        output_signals = self.beamforming_layer(input_signals = input_signals, weights = compensating_weights, sub_array_size = sub_array_size)

        return output_signals

    def calculate_compensating_weights(self, **kwargs):
        """
        ----------
        Parameters
        ----------
         **kwargs:  
                 target_angle: float
                    angle-of-arrival of signal or blocker or intended beam in degrees
                 inter_element_spacing: float
                    distance between beamformer inputs
        """
        target_angle=kwargs.get('target_angle', self.target_angle_final) 
        inter_element_spacing=kwargs.get('inter_element_spacing', self.inter_element_spacing) 

        compensating_delays = self.calculate_propagation_delays(target_angle=target_angle, inter_element_spacing = inter_element_spacing)
        compensating_phases_TTD = 2*np.pi*(self.fc + self.bw/2) * compensating_delays
        self.compensating_weights_TTD = np.exp(1j*-compensating_phases_TTD)
        self.compensating_phases_phaseshift = 2*np.pi*(self.fc) * compensating_delays
        return self.compensating_weights_TTD

    def calculate_input_signals(self):
        input_delays = self.calculate_propagation_delays(target_angle = self.angles)
        input_phases = 2*np.pi*(self.fc + self.bw/2) * input_delays #upper band edge
        self.input_signals = np.exp(1j*input_phases)
        return self.input_signals

    def calculate_inter_element_spacing(self):
        if self.antenna_spacing_type == 'fmax':
            self.inter_element_spacing = self.speedoflight /( (self.fc+self.bw/2) *2)
        elif self.antenna_spacing_type == 'fc':
            self.inter_element_spacing = self.speedoflight /(self.fc*2)

    def calculate_propagation_delays(self, **kwargs):
        """
        ----------
        Parameters
        ----------
         **kwargs:  
                 target_angle: float
                    angle-of-arrival of signal or blocker or intended beam in degrees
                 inter_element_spacing: float
                    distance between beamformer inputs
        """
        target_angle=kwargs.get('target_angle', self.target_angle_final) 
        inter_element_spacing=kwargs.get('inter_element_spacing', self.inter_element_spacing) 

        self.propagation_delays=np.zeros((len(target_angle),self.Nant))
        i=0
        for t in target_angle:
            self.propagation_delays[i] = np.arange(self.Nant)*inter_element_spacing*np.sin(np.deg2rad(t))/self.speedoflight;
            i += 1
        if np.min(self.propagation_delays) < 0:
            self.propagation_delays += abs(np.min(self.propagation_delays)) #remove negative delays

        return self.propagation_delays

    def beamforming_layer(self,**kwargs):
        """
        ----------
        plt.ylim(-40,5)
        Parameters
        ----------
         **kwargs:  
                input_signals: array
                    signals before beamforming weights applied
                weights : array
                    beamforming weights as array of numpy exponentials (array length is self.Nant)
                sub_array_size: int
                    how many antennas in sub_array
        """
        input_signals=kwargs.get('input_signals') 
        number_of_inputs = np.shape(input_signals)[1]
        weights=kwargs.get('weights') 
        sub_array_size=kwargs.get('sub_array_size', self.Nant) #No sub arrays by default

        outputs = np.zeros((len(input_signals),int(number_of_inputs/sub_array_size)),dtype=complex)
        compensated_signals = weights*input_signals

        for ind in np.arange(int(self.Nant/sub_array_size)):
            #outputs[:,ind] = np.sum(compensated_signals[:,[ind*sub_array_size,ind*sub_array_size+sub_array_size-1]], axis=1)
            outputs[:,ind] = np.sum(compensated_signals[:,np.arange(ind*sub_array_size,ind*sub_array_size+sub_array_size)], axis=1)
        return outputs



    def test_hybrid_beamforming(self):
        self.target_angle0 = [-50] #nulling
        sub_array_size = self.sub_array_size
        #self.sign_vector=[1,1]
        self.sign_vector=[1,-1]
        max_sub_array_gain = 20*np.log10(self.sub_array_size)
        max_bf_gain = 20*np.log10(self.Nant)

        input_signals = self.calculate_input_signals()
        compensating_weights = self.calculate_compensating_weights(target_angle = self.target_angle_layer0)

        for ind in np.arange(sub_array_size):
            compensating_weights[0][ind::sub_array_size] = compensating_weights[0][ind]*self.sign_vector[ind]
        analog_bf_outputs = self.beamforming_layer(input_signals = input_signals, weights = compensating_weights, sub_array_size = sub_array_size)

        # digital bf    
        self.target_angle1 = [50]
        self.Nant = np.shape(analog_bf_outputs)[1]
        digital_weights = self.calculate_compensating_weights(target_angle = self.target_angle_final, inter_element_spacing = self.inter_element_spacing*2)
        digital_bf_outputs = self.beamforming_layer(input_signals = analog_bf_outputs, weights = digital_weights)

        analog_bf_outputs_log = 20*np.log10(abs(analog_bf_outputs))
        analog_bf_outputs_log_norm = analog_bf_outputs_log - max_sub_array_gain#np.max(analog_bf_outputs_log)
        digital_bf_outputs_log = 20*np.log10(abs(digital_bf_outputs))
        digital_bf_outputs_log_norm = digital_bf_outputs_log - max_bf_gain#np.max(digital_bf_outputs_log)

        figure = plt.figure()
        plt.plot(self.angles,analog_bf_outputs_log_norm)
        self.default_plot_options(plt.gca())

        figure2 = plt.figure()
        plt.plot(self.angles,digital_bf_outputs_log_norm)
        self.default_plot_options(plt.gca())

    def default_plot_options(self,ax):
        plt.axvspan(-90,-60,alpha=0.2)
        plt.axvspan(60,90,alpha=0.2)
        plt.xticks(np.linspace(-90,90,7))
        plt.minorticks_on()
        #plt.xticks(ticks=np.linspace(-90,90,19),minor=True)
        ax.set_xticks(ticks=np.linspace(-90,90,19),minor=True)
        #plt.axes().tick_params(axis='y', which='minor', bottom=False)
        ax.yaxis.set_tick_params(which='minor', bottom=False)

        plt.grid(visible=True, which='both', axis ='x')
        plt.ylim(-40,5)

    def test_overlapping_sub_array_hybrid_beamforming(self):
        sub_array_size = self.sub_array_size
        #self.sign_vector=[1,1]
        self.sign_vector=[1,-1]
        max_sub_array_gain = 20*np.log10(self.sub_array_size)
        max_bf_gain = 20*np.log10(self.Nant)

        input_signals = self.calculate_input_signals()
        compensating_weights = self.calculate_compensating_weights(target_angle = self.target_angle_layer0)

        #inputs of overlapped sub-arrays
        overlap_inputs = np.zeros((len(input_signals),int((self.Nant-1)*sub_array_size)),dtype=complex)
        overlap_weights = np.zeros((int((self.Nant-1)*sub_array_size)),dtype=complex)

        for ind in np.arange(sub_array_size):
            overlap_weights[ind::sub_array_size] = compensating_weights[0][ind]*self.sign_vector[ind]
            overlap_inputs[:,np.arange(ind,overlap_inputs.shape[1],sub_array_size)] = input_signals[:,np.arange(0+ind, int((overlap_weights.shape[0]/2)+ind) )]

        analog_bf_outputs = self.beamforming_layer(input_signals = overlap_inputs, weights = overlap_weights, sub_array_size = sub_array_size)

        # digital bf    
        self.Nant = np.shape(analog_bf_outputs)[1]
        digital_weights = self.calculate_compensating_weights(target_angle = self.target_angle_final)
        digital_bf_outputs = self.beamforming_layer(input_signals = analog_bf_outputs, weights = digital_weights)

        analog_bf_outputs_log = 20*np.log10(abs(analog_bf_outputs))

        #analog_bf_outputs_log_norm = analog_bf_outputs_log - np.max(analog_bf_outputs_log)
        #digital_bf_outputs_log = 20*np.log10(abs(digital_bf_outputs))
        #digital_bf_outputs_log_norm = digital_bf_outputs_log - np.max(digital_bf_outputs_log)

        analog_bf_outputs_log_norm = analog_bf_outputs_log - max_sub_array_gain#np.max(analog_bf_outputs_log)
        digital_bf_outputs_log = 20*np.log10(abs(digital_bf_outputs))
        digital_bf_outputs_log_norm = digital_bf_outputs_log - max_bf_gain#np.max(digital_bf_outputs_log)

        figure = plt.figure()
        plt.plot(self.angles,analog_bf_outputs_log_norm)
        self.default_plot_options(plt.gca())

        figure2 = plt.figure()
        plt.plot(self.angles,digital_bf_outputs_log_norm)
        self.default_plot_options(plt.gca())

    def test_sub_array_nulling(self):
        sub_array_size = self.sub_array_size
        self.sign_vector=[1,-1] #sign vector needs lenghts self.Nant/sub_array_size

        input_signals = self.calculate_input_signals()
        compensating_weights = self.calculate_compensating_weights(target_angle = self.target_angle_layer0)

        for ind in np.arange(sub_array_size):
            compensating_weights[0][ind::sub_array_size] = compensating_weights[0][ind]*self.sign_vector[ind]
        output_signals = self.beamforming_layer(input_signals = input_signals, weights = compensating_weights, sub_array_size = sub_array_size)
            

        figure = plt.figure()
        #for ind in np.arange(int(self.Nant/sub_array_size)):
        #    plt.plot(self.angles,20*np.log10(abs(x[:,ind])))
        plt.plot(self.angles,20*np.log10(abs(output_signals)))
        self.default_plot_options(plt.gca())

    def test_main(self):
        bf_outputs = self.main()

        bf_outputs_log = 20*np.log10(abs(bf_outputs))
        bf_outputs_log_norm = bf_outputs_log - np.max(bf_outputs_log)

        plt.plot(self.angles,bf_outputs_log_norm)
        self.default_plot_options(plt.gca())


    def test_truncated_hadamar_matrix(self, **kwargs):
        self.Nant=4
        self.sign_vector = [1,-1,1,-1]
        x1 = self.main()
        self.sign_vector = [1,1,-1,-1]
        x2 = self.main()
        self.sign_vector = [1,-1,-1,1]
        x3 = self.main()

        figure = plt.figure()
        plt.plot(self.angles,20*np.log10(abs(x1)))
        plt.plot(self.angles,20*np.log10(abs(x2)))
        plt.plot(self.angles,20*np.log10(abs(x3)))
        plt.ylim(-30,18)
        plt.show(block=False);

    #def run(self,*arg):
    #    '''Guideline: Define model depencies of executions in `run` method.

    #    '''
    #    if len(arg)>0:
    #        self.par=True      #flag for parallel processing
    #        self.queue=arg[0]  #multiprocessing.queue as the first argument
    #    if self.model=='py':
    #        self.main()

if __name__=="__main__":
    import matplotlib.pyplot as plt
    from  beamforming_helper import *
    from  beamforming_helper.controller import controller as beamforming_helper_controller
    import pdb
    import math
    import plot_format
    plot_format.set_style('ieeetran')
    bh=beamforming_helper()
    x=bh.Nant
    bh.test_main()
    #bh.test_truncated_hadamar_matrix()
    #bh.test_sub_array_nulling()
    bh.test_hybrid_beamforming()
    bh.Nant=x
    bh.test_overlapping_sub_array_hybrid_beamforming()


    plt.show(block=False);
    pdb.set_trace()

    input()

