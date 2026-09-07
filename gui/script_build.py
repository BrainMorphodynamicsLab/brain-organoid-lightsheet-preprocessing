from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk
import tkinter as tk
import os
import sys
import functools


# class holding user interface elements
class UserInterfaceControls:
    def __init__(self):
        self.root = None
        self.parameter_controls = None 
        self.script_status = None 
        self.start_button = None 
        self.stop_button = None 
        self.abort_button = None 
        self.save_button = None 
        self.load_button = None 
        self.matplotlib_frame = None
        self.help_canvas = None
        self.validation_label = None 
        self.output_text = None

parameter_in_focus = None

def select_directory(entry, user_interface_validate):
    dir_name = filedialog.askdirectory()
    if dir_name != '':
        entry.set(dir_name)            
    user_interface_validate(None)


def show_help(event, script, help_canvas, script_name, info = None, graph_plotted = False):
    global parameter_in_focus
    image_path = None

    if info == "enter_figure":
        if graph_plotted:
            help_canvas.delete('all')
            image_path = os.path.join(sys.path[0], "..", "scripts", script_name, "legend.png")
            if hasattr(script, "figures_help"):
                help_canvas.create_text(3, 0, text='Figure legend:', anchor='nw', width=250, fill='black')
                help_canvas.create_text(3, 20, text=script.figures_help, anchor='nw', width=250, fill='black')
    else:
        if info == "leave_figure":
            parameter = parameter_in_focus
        else:
            parameter = event.widget.parameter
            parameter_in_focus = parameter

        help_canvas.delete('all')
        help_canvas.create_text(3, 0, text='Parameter help:', anchor='nw', width=250, fill='black')
        if parameter == None:
            help_canvas.create_text(3, 20, text="Select a field to display help.", anchor='nw', width=250, fill='black')
        elif parameter in script.parameters and parameter in script.parameter_help:
            help_canvas.create_text(3, 20, text=script.parameter_help[parameter], anchor='nw', width=600, fill='black')
            image_path = os.path.join(sys.path[0], "..", "scripts", script_name, parameter + ".png")
        else:
            help_canvas.create_text(3, 20, text="No documentation for current parameter.", anchor='nw', width=250, fill='black')
    
    if image_path != None and os.path.exists(image_path):
        help_image = tk.PhotoImage(file=image_path)
        help_canvas.create_image(300, 0, image=help_image, anchor='nw')
        help_canvas.image = help_image


def build_user_interface(script, user_interface_validate, script_name):
    user_interface_controls = UserInterfaceControls()
    
    root = tk.Tk()
    if hasattr(script, "window_title"):
        root.title(script.window_title)
    else:
        root.title(script_name + ".py")
    root.geometry("1100x900")
    root.minsize(800, 400)
    try:
        root.iconbitmap(os.path.join(sys.path[0], 'icon_window.ico'))
    except Exception:
        pass
    root.configure(background='#293955')
    root.rowconfigure((0, 1, 3), weight=0)
    root.rowconfigure(2, weight=1)
    root.columnconfigure(0, weight=1)
    user_interface_controls.root = root   

    # frame to hold parameter controls
    top_frame = tk.Frame(root, background='white')
    top_frame.grid(column=0, row=0, sticky=("NWSE"), padx=(9,9), pady=(9,0))
    for col in range(6):
        if col % 2 == 0:
            top_frame.columnconfigure(col, weight=0, minsize=160)
        else:
            top_frame.columnconfigure(col, weight=1)
    
    parameter_controls = {}
    
    pad_left = 15; pad_right = 18; column_number = 3
    label = tk.Label(top_frame, text='Parameters:', background='white', foreground='black', font=('TkDefaultFont', 12, 'bold'))
    label.grid(column=0, columnspan=6, row=0, sticky=("W"), padx=(pad_left,0), pady=(15,3))

    for parameter_idx, parameter in enumerate(script.parameters):
        if not parameter in script.parameter_control:
            parameter_control = 'entry'
        elif isinstance(script.parameter_control[parameter], (list, tuple)):
            parameter_control = script.parameter_control[parameter][0]
        else:
            parameter_control = script.parameter_control[parameter]
            
        col_label = (parameter_idx % column_number) * 2
        col_entry = col_label + 1
        row = parameter_idx // column_number + 1

        if parameter_control == 'entry':
            label = tk.Label(top_frame, text=script.parameter_label[parameter]+':', background='white', foreground='black', anchor='w')
            label.grid(column=col_label, row=row, sticky=("W"), padx=(pad_left,5), pady=3)        
            var = tk.StringVar()
            entry = tk.Entry(top_frame, textvariable=var, background='white', foreground='black', insertbackground='black')
            entry.parameter = parameter
            entry.access = 'normal'
            entry.bind("<FocusIn>", lambda event: show_help(event, script, help_canvas, script_name))
            entry.bind("<Any-KeyRelease>", user_interface_validate)
            entry.var = var
            entry.grid(column=col_entry, row=row, sticky=("WE"), padx=(0,pad_right), pady=3) 
            parameter_controls[parameter] = entry
        
        elif parameter_control == 'folder':
            label = tk.Label(top_frame, text=script.parameter_label[parameter]+':', background='white', foreground='black', anchor='w')
            label.grid(column=col_label, row=row, sticky=("W"), padx=(pad_left,5), pady=3)           
            frame = tk.Frame(top_frame, background='white')
            frame.grid(column=col_entry, row=row, sticky="NWE")
            frame.columnconfigure(0, weight=1)
            frame.columnconfigure(1, weight=0)   
            
            var = tk.StringVar()
            entry = tk.Entry(frame, textvariable=var, background='white', foreground='black', insertbackground='black')
            entry.parameter = parameter
            entry.access = 'normal'
            entry.bind("<FocusIn>", lambda event: show_help(event, script, help_canvas, script_name))
            entry.bind("<Any-KeyRelease>", user_interface_validate)
            entry.var = var
            entry.grid(column=0, row=0, sticky=("WE"), pady=3) 
            parameter_controls[parameter] = entry
            
            icon = ImageTk.PhotoImage(file=os.path.join(sys.path[0], 'icon_open.png'))
            button = tk.Button(frame, image=icon, command=functools.partial(select_directory, var, user_interface_validate), background='white')
            button.image = icon
            button['relief'] = 'flat'
            button.grid(column=1, row=0, sticky="NW", padx=(0,pad_right))      
        
        elif parameter_control == 'checkbox':
            label = tk.Label(top_frame, text=script.parameter_label[parameter]+':', background='white', foreground='black', anchor='w')
            label.grid(column=col_label, row=row, sticky=("W"), padx=(pad_left,5), pady=3)            
            var = tk.BooleanVar(value=False)
            check = tk.Checkbutton(top_frame, text="Enable", variable=var, background='white', foreground='black', selectcolor='white')
            check.parameter = parameter
            check.access = 'normal'
            check.bind("<Button>", lambda event: show_help(event, script, help_canvas, script_name))
            check.var = var
            check['relief'] = 'flat'
            check.grid(column=col_entry, row=row, sticky="NW", padx=(0,pad_right), pady=3)
            parameter_controls[parameter] = check
        
        elif parameter_control == 'combobox':
            label = tk.Label(top_frame, text=script.parameter_label[parameter]+':', background='white', foreground='black', anchor='w')
            label.grid(column=col_label, row=row, sticky=("W"), padx=(pad_left,5), pady=3)
            var = tk.StringVar()
            if script.parameter_control[parameter][2] == "readonly":
                state = 'readonly'
            elif script.parameter_control[parameter][2] == 'normal':
                state = 'normal'
            else:
                raise Exception("Unsupported combobox state " + script.parameter_control[parameter][2])
            combobox = ttk.Combobox(top_frame, textvariable=var, value=script.parameter_control[parameter][1], state=state)
            combobox.parameter = parameter
            combobox.access = state
            combobox.bind("<FocusIn>", lambda event: show_help(event, script, help_canvas, script_name))
            combobox.bind("<FocusOut>", user_interface_validate)
            combobox.var = var
            combobox.grid(column=col_entry, row=row, sticky="WE", padx=(0,pad_right), pady=3)
            parameter_controls[parameter] = combobox
            
        else:
            raise Exception("Control type '" + parameter_control + "' not supported!")

    user_interface_controls.parameter_controls = parameter_controls
    
    last_row_parameters = len(script.parameters)//column_number+1
        
    validation_label = tk.Label(top_frame, background='white', foreground='red', justify='left', wraplength=760)
    validation_label.grid(column=0, columnspan=6, row=last_row_parameters+1, sticky="NWS", padx=(pad_left,pad_right), pady=3)
    user_interface_controls.validation_label = validation_label

    middle_frame = tk.Frame(root, background='white')
    middle_frame.grid(column=0, row=1, sticky=("NWSE"), padx=(9,9), pady=(0,0))
    middle_frame.columnconfigure(0, weight=0)
    middle_frame.columnconfigure((1, 2), weight=1)
    middle_frame.rowconfigure((0, 1, 2), weight=0)

    label_script_controls = tk.Label(middle_frame, text='Script controls:', background='white', foreground='black')
    label_script_controls.grid(column=0, row=0, sticky="W", padx=(pad_left,0), pady=(15,3))
    
    buttons_frame = tk.Frame(middle_frame, background='white')
    buttons_frame['relief']='flat'
    buttons_frame.grid(column=0, columnspan=1, row=1, sticky=("NWES"))

    icon_start = ImageTk.PhotoImage(file=os.path.join(sys.path[0], 'icon_start.png'))
    icon_stop = ImageTk.PhotoImage(file=os.path.join(sys.path[0], 'icon_stop.png'))
    icon_abort = ImageTk.PhotoImage(file=os.path.join(sys.path[0], 'icon_abort.png'))
    icon_save = ImageTk.PhotoImage(file=os.path.join(sys.path[0], 'icon_save.png'))
    icon_load = ImageTk.PhotoImage(file=os.path.join(sys.path[0], 'icon_open.png'))

    start_button = tk.Button(buttons_frame, text='Start', image=icon_start, compound='top', background='white', foreground='black', width=30, height=40)
    start_button.image = icon_start
    start_button['borderwidth'] = 0.5
    start_button['relief'] = 'solid'
    start_button.grid(column=0, row=0, sticky="NE", padx=(pad_left+3,0))
    user_interface_controls.start_button = start_button
    
    stop_button = tk.Button(buttons_frame, text='Stop', image=icon_stop, compound='top', background='white', foreground='black', width=30, height=40)
    stop_button.image = icon_stop
    stop_button['borderwidth'] = 0.5
    stop_button['relief'] = 'solid'
    stop_button.grid(column=1, row=0, sticky="W", padx=2)
    user_interface_controls.stop_button = stop_button
    
    abort_button = tk.Button(buttons_frame, text='Abort', image=icon_abort, compound='top', background='white', foreground='black', width=30, height=40)
    abort_button.image = icon_abort
    abort_button['borderwidth'] = 0.5
    abort_button['relief'] = 'solid'
    abort_button.grid(column=4, row=0, sticky="W", padx=2)
    user_interface_controls.abort_button = abort_button
    
    save_button = tk.Button(buttons_frame, text='Save', image=icon_save, compound='top', background='white', foreground='black', width=30, height=40)
    save_button.image = icon_save
    save_button['borderwidth'] = 0.5
    save_button['relief'] = 'solid'
    save_button.grid(column=2, row=0, sticky="W", padx=2)
    user_interface_controls.save_button = save_button 
    
    load_button = tk.Button(buttons_frame, text='Load', image=icon_load, compound='top', background='white', foreground='black', width=30, height=40)
    load_button.image = icon_load
    load_button['borderwidth'] = 0.5
    load_button['relief'] = 'solid'
    load_button.grid(column=3, row=0, sticky="W", padx=2)
    user_interface_controls.load_button = load_button 
    
    status_frame = tk.Frame(middle_frame, background='white')
    status_frame['relief']='flat'
    status_frame.grid(column=0, columnspan=2, row=2, sticky=("NWES"), pady=(10,0))
    
    label_status = tk.Label(status_frame, background='white', foreground='black', text='Status: ')
    label_status.grid(column=0, row=0, sticky="WN", padx=(pad_left,0))
    
    script_status = tk.Label(status_frame, text='Stopped', background='white', foreground='black')
    script_status.grid(column=1, row=0, sticky="NW") 
    user_interface_controls.script_status = script_status

    help_canvas = tk.Canvas(middle_frame, height=130, background='white', highlightthickness=0, relief='ridge', width=600)
    help_canvas.create_text(3, 0, text='Parameter help:', anchor='nw', width=250, fill='black')
    help_canvas.create_text(3, 18, text='Select a field to display help.', anchor='nw', width=250, fill='black')
    help_canvas.grid(column=1, columnspan=2, row=0, rowspan=3, sticky="WE", padx=(30,pad_right), pady=(10,0))
    user_interface_controls.help_canvas = help_canvas
    
    plotting_frame = tk.Frame(root, background='white')
    plotting_frame['relief'] = 'ridge'
    plotting_frame.grid(column=0, row=2, sticky=("NWSE"), padx=(9,9))
    plotting_frame.columnconfigure(0, weight=1)
    plotting_frame.rowconfigure(0, weight=1)
    user_interface_controls.matplotlib_frame = plotting_frame
    
    output_frame = tk.Frame(root, background='white')
    output_frame.grid(column=0, row=3, sticky=("NWES"), padx=(9,9), pady=(0,9))
    output_frame.columnconfigure(0, weight=1)
    output_frame.columnconfigure(1, weight=0, minsize=20)
    
    output_text = tk.Text(output_frame, width=70, height=10, background='black', foreground='white')
    ys = ttk.Scrollbar(output_frame, orient='vertical', command=output_text.yview)
    output_text['yscrollcommand'] = ys.set
    output_text.grid(column=0, row=0, sticky=("NWES"), padx=(pad_left+3,0), pady=(10,15))
    ys.grid(column=1, row=0, sticky=("NSW"), padx=(0,pad_right), pady=(10,15))
    user_interface_controls.output_text = output_text

    user_interface_controls.root.bind("<Alt_L>", "Nothing", add=True)
    user_interface_controls.root.bind("<Alt_R>", "Nothing", add=True)
    
    return user_interface_controls