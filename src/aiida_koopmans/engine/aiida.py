from koopmans.engines.engine import Engine
from koopmans.step import Step
from koopmans.calculators import Calc
from koopmans.pseudopotentials import read_pseudo_file

from aiida.engine import run_get_node, submit

from aiida_koopmans.utils import *

from aiida_pseudo.data.pseudo import UpfData

import time

import dill as pickle

from aiida import orm, load_profile
load_profile()

class AiiDAEngine(Engine):
    
    """
    Step data is a dictionary containing the following information:
    step_data = {calc.directory: {'workchain': workchain, 'remote_folder': remote_folder}}
    and any other info we need for AiiDA.  
    """
    
    def __init__(self, *args, **kwargs):
        self.blocking = kwargs.pop('blocking', True)
        self.step_data = { # TODO: change to a better name
            'configuration': kwargs.pop('configuration', None),
            'steps': {}
            }
        
        # here we add the logic to populate configuration by default
        # 1. we look for codes stored in AiiDA at localhost, e.g. pw-version@localhost,
        # 2. we look for codes in the PATH, 
        # 3. if we don't find the code in AiiDA db but in the PATH, we store it in AiiDA db.
        # 4. if we don't find the code in AiiDA db and in the PATH and not configuration is provided, we raise an error.
        if self.step_data['configuration'] is None:
            raise NotImplementedError("Configuration not provided")
        
        # 5. if no resource info in configuration, we try to look at PARA_PREFIX env var.

        
        super().__init__(*args, **kwargs)
    
    
    def _run_steps(self, steps: tuple[Step, ...]) -> None:
        try:
            with open('step_data.pkl', 'rb') as f:
                self.step_data = pickle.load(f) # this will overwrite the step_data[configuration], ie. if we change codes or res we will not see it if the file already exists.
        except:
            pass
        
        self.from_scratch = False
        for step in steps:
            # self._step_running_message(step)
            if step.directory in self.step_data['steps']:
                continue
            elif step.prefix in ['wannier90_preproc', 'pw2wannier90']:
                print(f'skipping {step.prefix} step')
                continue
            else:
                self.from_scratch = True
            
            #step.run()
            self.step_data['steps'][step.directory] = {}
            builder = get_builder_from_ase(calculator=step, step_data=self.step_data) # ASE to AiiDA conversion. put some error message if the conversion fails
            running = submit(builder)
            # running = aiidawrapperwchain.submit(builder) # in the non-blocking case.
            self.step_data['steps'][step.directory] = {'workchain': running.pk, } #'remote_folder': running.outputs.remote_folder}
        
        #if self.from_scratch: 
        with open('step_data.pkl', 'wb') as f:
            pickle.dump(self.step_data, f) 
        
        if not self.blocking and self.from_scratch:
            raise CalculationSubmitted("Calculation submitted to AiiDA, non blocking")
        elif self.blocking:
            for step in self.step_data['steps'].values():
                while not orm.load_node(step['workchain']).is_finished:
                    time.sleep(5)
            
        for step in steps:
            # convert from AiiDA to ASE results and populate ASE calculator            
            # TOBE put in a separate function
            if step.prefix in ['wannier90_preproc', 'pw2wannier90']:
                continue
            workchain = orm.load_node(self.step_data['steps'][step.directory]['workchain'])
            if "remote_folder" in workchain.outputs:
                self.step_data['steps'][step.directory]['remote_folder'] = workchain.outputs.remote_folder.pk
            output = None
            if step.ext_out == ".wout":
                output = read_output_file(step, workchain.outputs.wannier90.retrieved)
            elif step.ext_out in [".pwo",".kho"]:
                output = read_output_file(step, workchain.outputs.retrieved)
                if hasattr(output.calc, 'kpts'):
                    step.kpts = output.calc.kpts
            else:
                output = read_output_file(step, workchain.outputs.retrieved)
            if step.ext_out in [".pwo",".wout",".kso",".kho"]:
                step.calc = output.calc
                step.results = output.calc.results
                step.generate_band_structure(nelec=int(workchain.outputs.output_parameters.get_dict()['number_of_electrons']))
                
            self._step_completed_message(step)

        # If we reached here, all future steps should be performed from scratch
        self.from_scratch = True

        # dump again to have update the information
        with open('step_data.pkl', 'wb') as f:
            pickle.dump(self.step_data, f) 
        
        return

    def load_old_calculator(self, calc: Calc):
        raise NotImplementedError # load_old_calculator(calc)
    
    def get_pseudo_data(self, workflow):
        pseudo_data = {}
        symbols_list = []
        for symbol in workflow.pseudopotentials.keys():
            symbols_list.append(symbol)

        qb = orm.QueryBuilder()
        qb.append(orm.Group, filters={'label': {'==': 'pseudo_group'}}, tag='pseudo_group')
        qb.append(UpfData, filters={'attributes.element': {'in': symbols_list}}, with_group='pseudo_group')
        
        for pseudo in qb.all():
            with tempfile.TemporaryDirectory() as dirpath:
                temp_file = pathlib.Path(dirpath) / pseudo[0].attributes.element + '.upf'
                with pseudo[0].open(pseudo[0].attributes.element + '.upf', 'wb') as handle:
                    temp_file.write_bytes(handle.read())
                pseudo_data[pseudo[0].attributes.element] =  read_pseudo_file(temp_file)
        
        return pseudo_data


def load_old_calculator(calc):
    # This is a separate function so that it can be imported by other engines
    loaded_calc = calc.__class__.fromfile(calc.directory / calc.prefix)

    if loaded_calc.is_complete():
        # If it is complete, load the results
        calc.results = loaded_calc.results

        # Check the convergence of the calculation
        calc.check_convergence()

        # Load k-points if relevant
        if hasattr(loaded_calc, 'kpts'):
            calc.kpts = loaded_calc.kpts

        if isinstance(calc, ReturnsBandStructure):
            calc.generate_band_structure()

        if isinstance(calc, ProjwfcCalculator):
            calc.generate_dos()

        if isinstance(calc, PhCalculator):
            calc.read_dynG()

    return loaded_calc