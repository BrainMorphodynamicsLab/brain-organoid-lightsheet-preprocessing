import sys
import os
import pathlib

# insert path to directory above current file to import correct pymcs library
root_directory = pathlib.Path(os.path.abspath(__file__)).parent.parent.absolute()
sys.path.insert(1, str(root_directory))
# insert path to scripts folder for imports inside the script files located in scripts directory
sys.path.insert(2, os.path.join(str(root_directory), "scripts"))

# Determine the requested script before importing Tkinter.  On systems with an
# incompatible Tcl/Tk runtime, importing the generic GUI stack can abort the
# process before Python can raise a catchable exception.  The full pipeline is
# therefore routed to a standard-library local browser GUI first.
if len(sys.argv) >= 2:
    script_name = sys.argv[1]
else:
    script_name = "example_script"

if script_name == "full_pipeline":
    from web_pipeline_gui import main as web_pipeline_gui_main
    web_pipeline_gui_main(sys.argv[2:])
    sys.exit(0)

from script_build import build_user_interface
from script_build import show_help

import threading
import json
import ctypes
import tkinter
import importlib
import pymcs

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ignore all warnings (will not appear in terminal) -> to be diseabled for development
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


# custom exception raised on script thread to abort it
class ScriptAborted(Exception):
    pass


def start_button_click():
    global script_running; global script_thread
    global figure_count; global figures; global figure_canvasses; global graph_plotted

    if script_running:
        return    
    
    # clear script output on the user interface
    user_interface_controls.output_text.configure(state="normal")
    user_interface_controls.output_text.delete('1.0', 'end')
    user_interface_controls.output_text.configure(state="disable")

    # convert values to correct type
    parameter_values = {}    
    for parameter in script.parameters: 
        parameter_values[parameter] = user_interface_controls.parameter_controls[parameter].var.get()
        
    converted_values, error_message = convert_parameters(script.parameters, parameter_values, script.parameter_type, script.parameter_label)
    if error_message != None:
        return
    
    # creates the correct number of canvasses
    figure_count = script.get_figure_count(parameter_values)
    figures, figure_canvasses = create_figure_canvasses(user_interface_controls.matplotlib_frame, figure_count)
    graph_plotted = False

    script_started.clear()
    script_thread = threading.Thread(name="Script", target=script_run, args=(converted_values,), daemon=True)
    script_thread.start()
    script_started.wait()
    
    user_interface_controls.script_status.config(text='Running')
    user_interface_controls.start_button.configure(state='disable')
    user_interface_controls.stop_button.configure(state='normal')
    user_interface_controls.abort_button.configure(state='normal')
    user_interface_controls.load_button.configure(state='disable')
    user_interface_controls.save_button.configure(state='disable')

    for parameter in user_interface_controls.parameter_controls:
        user_interface_controls.parameter_controls[parameter].configure(state='disable')
        
    script_running = True


def create_figure_canvasses(matplotlib_frame, number_plot):
    figures = []
    figure_canvasses = []

    # destroy the canvasses created for the previous run
    for child in matplotlib_frame.winfo_children():
        child.destroy()

    for index in range(number_plot):
        # configure the number of columns nedded in the matplotlib_frame
        matplotlib_frame.columnconfigure(index, weight=1)

        # list of figures needed
        figures.append(Figure())
        # each canvas must be manually attached to its figure
        figure_canvasses.append(FigureCanvasTkAgg(figures[index], master=matplotlib_frame))
        figure_canvasses[index].get_tk_widget().id = index
        figure_canvasses[index].get_tk_widget().bind("<MouseWheel>", mouse_wheel_event)
        figure_canvasses[index].get_tk_widget().bind("<Button-1>", mouse_left_click)

        # canvas placed in the matplotlib_frame
        figure_canvasses[index].get_tk_widget().grid(row=0, column=(index%number_plot), sticky="NWES")

    return figures, figure_canvasses


def script_run(parameter_values):
    script_started.set()

    try:
        script.run(parameter_values, update_figures_callback)

        # in case script did not disconnect from Microscope do it here
        # class keeps last instance of microscope in this static variable
        if pymcs.Microscope._microscope_instance != None:
            pymcs.Microscope._microscope_instance.disconnect()
    except ScriptAborted:
        # if aborted script may say connected to Microscope, disconnect here
        # class keeps last instance of microscope in this static variable
        if pymcs.Microscope._microscope_instance != None:
            pymcs.Microscope._microscope_instance.disconnect()
        raise


def stop_button_click():
    if script_thread != None:
        script.stop()


def abort_button_click():
    if script_thread != None:
        thread_id = script_thread.ident
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, ctypes.py_object(ScriptAborted))


def save_button_click(): 
    parameter_values = {}    
    for parameter in script.parameters: 
        parameter_values[parameter] = user_interface_controls.parameter_controls[parameter].var.get()
        
    converted_values, error_message = convert_parameters(script.parameters, parameter_values, script.parameter_type, script.parameter_label)
    
    if error_message == None:
        file_name = tkinter.filedialog.asksaveasfilename(filetype=[("JSON files", "*.json")], defaultextension = json)
        if file_name == '':
            return        
        with open(file_name, 'w') as file_open:
            json.dump(converted_values, file_open, indent=5)
    else:
        tkinter.messagebox.showinfo(title="Save parameters", message="Cannot save parameters. " + error_message)


def load_button_click():    
    file_name = tkinter.filedialog.askopenfilename(filetype=[("JSON files", "*.json")])
    if file_name == '':
        return
    with open(file_name) as file_opened:
         parameter_values = json.load(file_opened)
        
    load_parameters_to_ui(parameter_values)


def load_parameters_to_ui(values_to_load): 
    for parameter in script.parameters: 
        if not parameter in script.parameter_control:
            parameter_control = 'entry'
        elif isinstance(script.parameter_control[parameter], (list, tuple)):
            parameter_control = script.parameter_control[parameter][0]
        else:
            parameter_control = script.parameter_control[parameter]
        
        # if parameter value exists in values to load, load into ui
        if parameter in values_to_load:
            if parameter_control == 'entry' or parameter_control == 'folder':
                user_interface_controls.parameter_controls[parameter].var.set(str(values_to_load[parameter]))
                
            elif parameter_control == 'checkbox':
                user_interface_controls.parameter_controls[parameter].var.set(bool(values_to_load[parameter]))
                
            elif parameter_control == 'combobox':
                if values_to_load[parameter] in script.parameter_control[parameter][1]:
                    user_interface_controls.parameter_controls[parameter].var.set(values_to_load[parameter])
                else:
                    user_interface_controls.parameter_controls[parameter].var.set("")                       
        # otherwise initialize to default value
        else:
            if parameter_control == 'checkbox':
                user_interface_controls.parameter_controls[parameter].var.set(False)
            else:
                user_interface_controls.parameter_controls[parameter].var.set("")
    
    user_interface_validate(None)


def user_interface_validate(event):
    parameter_values = {}    
    for parameter in script.parameters: 
        parameter_values[parameter] = user_interface_controls.parameter_controls[parameter].var.get()
        
    _, error_message = convert_parameters(script.parameters, parameter_values, script.parameter_type, script.parameter_label)
    
    if error_message == None:
        user_interface_controls.validation_label.config(foreground='green')
        user_interface_controls.validation_label.config(text='Parameters are correct.')
        user_interface_controls.start_button.configure(state='normal')
    else:
        user_interface_controls.validation_label.config(foreground='red')
        user_interface_controls.validation_label.config(text=error_message)
        user_interface_controls.start_button.configure(state='disable')
        
    return True


def convert_parameters(parameters, parameter_values, parameter_types, parameter_labels):
    converted_values = {}
    error_messages = []
    
    for parameter in parameters:
        if parameter in parameter_types:
            if isinstance(parameter_types[parameter], (list, tuple)):
                parameter_type = parameter_types[parameter][0]
            else:
                parameter_type = parameter_types[parameter]
            
            if parameter_type == float:
                try:
                    converted_values[parameter] = float(parameter_values[parameter])
                except ValueError:
                    error_messages.append(parameter_labels[parameter] + ' must be numeric')
                    
            elif parameter_type == int:
                try:
                    converted_values[parameter] = int(parameter_values[parameter])
                except ValueError:
                    error_messages.append(parameter_labels[parameter] + ' must be integer')
                    
            elif parameter_type == bool:
                try:
                    converted_values[parameter] = bool(parameter_values[parameter])
                except ValueError:
                    error_messages.append(parameter_labels[parameter] + ' not type bool')
                    
            elif parameter_type == str: 
                converted_values[parameter] = parameter_values[parameter]
                
            else:
                raise Exception("Type '" + parameter_type + "' not recognized.")
            
            # verify that numerical values are whithint correct range
            if isinstance(parameter_types[parameter], (list, tuple)) and len(parameter_types[parameter])==3 and parameter in converted_values:
                if (converted_values[parameter] < parameter_types[parameter][1]) or (converted_values[parameter] > parameter_types[parameter][2]):
                    error_messages.append(parameter_labels[parameter] + " must be between " + str(parameter_types[parameter][1]) + " and " + str(parameter_types[parameter][2]))
            
            # if a field is empty generate an error message
            if parameter_values[parameter] == '' and parameter_type == str:
                error_messages.append(parameter_labels[parameter] + ' field cannot be empty')
        
        else: 
            raise Exception("Key '" + parameter + "' not supported by script parameter_type")
        
    if len(error_messages) == 0:
        return converted_values, None
    else:
        return None, ", ".join(error_messages) + "."


def mouse_wheel_event(event):
    if hasattr(script, 'figure_event') and callable(getattr(script, 'figure_event')):
        if event.delta > 0:
            script.figure_event('MouseWheelUp', None, event.widget.id, update_figures)
        if event.delta < 0:
            script.figure_event('MouseWheelDown', None, event.widget.id, update_figures)

def mouse_left_click(event):
    if hasattr(script, 'figure_event') and callable(getattr(script, 'figure_event')):
        script.figure_event('MouseLeftClick', None, event.widget.id, update_figures)


def update_figures_callback():
    figures_updated.clear()
    user_interface_controls.root.after(0, update_figures)
    figures_updated.wait()


def update_figures():
    global graph_plotted
    script.update_figures(figures)
    for idx in range(figure_count):
        figure_canvasses[idx].draw()
    graph_plotted = True
    figures_updated.set()


def update_script_status():
    global script_running     
    global script_thread

    # update status of script in user interface
    if script_running == True and script_thread.is_alive() == False:
        user_interface_controls.script_status.config(text='Stopped')
        user_interface_controls.start_button.configure(state='normal')
        user_interface_controls.stop_button.configure(state='disable')
        user_interface_controls.abort_button.configure(state='disable')
        user_interface_controls.load_button.configure(state='normal')
        user_interface_controls.save_button.configure(state='normal')
        script_running = False
        script_thread.join()
        script_thread = None
        
        for parameter in user_interface_controls.parameter_controls:
            state = user_interface_controls.parameter_controls[parameter].access
            user_interface_controls.parameter_controls[parameter].configure(state=state)
     
    user_interface_controls.root.after(200, update_script_status)


def close_program():
    user_interface_controls.root.destroy()
    sys.exit()


# import script
script_spec = importlib.util.spec_from_file_location("script", os.path.join(sys.path[0], "..", "scripts", script_name + ".py"))
script = importlib.util.module_from_spec(script_spec)
script_spec.loader.exec_module(script)

# decorator to output string to user interface
def decorator(func):
    def inner(input_string):
        printed_on_ui.clear()
        print_on_ui(input_string)
        printed_on_ui.wait()
        return func(input_string)
    return inner

def print_on_ui_callback(input_string):
    printed_on_ui.clear()
    global print_string
    print_string = input_string
    user_interface_controls.root.after(0, print_on_ui)
    printed_on_ui.wait()

def print_on_ui():
    user_interface_controls.output_text.configure(state="normal")
    user_interface_controls.output_text.insert(tkinter.END, print_string)  
    number_of_lines = int(user_interface_controls.output_text.index('end-1c').split('.')[0])
    if number_of_lines > 25:
        user_interface_controls.output_text.delete('1.0', str(number_of_lines - 25) + ".0")
    user_interface_controls.output_text.see("end")
    user_interface_controls.output_text.configure(state="disabled")
    printed_on_ui.set()

# output stderr and stdout to the user interface 
sys.stdout.write=print_on_ui_callback
sys.stderr.write=print_on_ui_callback

# global variables
script_thread = None
script_running = False
graph_plotted = False
figure_count = 0
figures = []
figure_canvasses = []
print_string = ""

# event raised by decorator applied on print function
printed_on_ui = threading.Event()
# event raised by update figure callback
figures_updated = threading.Event()
# event raised on script thread to signal it started
script_started = threading.Event()

script = script.Script()

# building interface using parameters from the Script class
user_interface_controls = build_user_interface(script, user_interface_validate, script_name)

load_parameters_to_ui(script.parameter_default_value)
user_interface_validate(None)

# bind the plotting area to display a legend if it exist
user_interface_controls.matplotlib_frame.bind("<Enter>", lambda event: show_help(event, script, user_interface_controls.help_canvas, script_name, "enter_figure", graph_plotted))
user_interface_controls.matplotlib_frame.bind("<Leave>", lambda event: show_help(event, script, user_interface_controls.help_canvas, script_name, "leave_figure", graph_plotted))

# link buttons with commands and change state
user_interface_controls.start_button.configure(command=start_button_click, state='normal')
user_interface_controls.stop_button.configure(command=stop_button_click, state='disable')
user_interface_controls.abort_button.configure(command=abort_button_click, state='disable')
user_interface_controls.save_button.configure(command=save_button_click, state='normal')
user_interface_controls.load_button.configure(command=load_button_click, state='normal')

user_interface_controls.root.after(200, update_script_status)

user_interface_controls.root.protocol("WM_DELETE_WINDOW", close_program)
user_interface_controls.root.mainloop()