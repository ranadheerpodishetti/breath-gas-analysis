import tkinter as tk
from tkinter import ttk
from tkinter import *
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
from matplotlib.figure import Figure
from tkinter import filedialog
from tkinter import messagebox
from functools import partial



import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from scipy.signal import find_peaks

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

scaler = StandardScaler()

from sklearn import svm, datasets
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
import xgboost
from xgboost import XGBClassifier, XGBRFClassifier
from xgboost import plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import classification_report
from sklearn import preprocessing



LARGE_FONT = ("Verdana", 12)
class PTR_MS_analysis(tk.Tk):


    def __init__(self,*args,**kwargs):
        tk.Tk.__init__(self,*args,**kwargs)
        container = tk.Frame(self)

        container.pack(side ='top',fill = 'both', expand = True)

        container.grid_rowconfigure(0,minsize=2000,weight=1)
        container.grid_columnconfigure(0,minsize=2000,weight =1)

        menubar = tk.Menu(container)
        filemenu_file =tk.Menu(menubar , tearoff = 0)
        #filemenu.add_command(label= "save sttings",command = lambda: popupmsg("not supported just yet"))
        #filemenu.add_seperator()
        filemenu_file.add_command(label="New",command= self.getcsv)

        self.scrollbar = Scrollbar(self ,orient=HORIZONTAL)
        self.lbox = tk.Listbox(self, height=100, width=50, selectmode=MULTIPLE)
        self.scrollbar.config(command=self.lbox.yview)
        self.lbox.config(xscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side=RIGHT, fill=X)
        self.scrollbar.config(command=self.lbox.xview)

        self.lbox.place(x=20, y=50)





        filemenu_file.add_command(label="Open",command = self.preprocessing_peak_pick_pop_up)
        clear_sub_menu = tk.Menu(menubar, tearoff=0)
        clear_sub_menu.add_command(label="clear GUI",command = self.get_size_lb)
        clear_sub_menu.add_command(label="clear all imported data",command = self.clear_listbox)
        clear_sub_menu.add_command(label="clear",command = self.clear_selected_item)

        filemenu_file.add_cascade(label= "Clear",menu = clear_sub_menu)
        filemenu_file.add_command(label="EXIT",command =quit)

        menubar.add_cascade(label ="FILE",menu =filemenu_file)

        filemenu_View = tk.Menu(menubar, tearoff=0)
        filemenu_View.add_command(label="Spectrum",command = self.compare_spectrum)
        menubar.add_cascade(label="View",menu =filemenu_View)

        filemenu_Preprocessing = tk.Menu(menubar, tearoff=0)
        filemenu_Preprocessing.add_command(label="Peak picking",command = self.peak_pick_spectrum)
        filemenu_Preprocessing.add_command(label="Binning" ,command =self.view_binning)
        filemenu_Preprocessing.add_command(label="VOC Correlation" ,command = self.Feature_Correlation)
        filemenu_Preprocessing.add_command(label="cluster Heatmap",command = self.cluster_heat_map)
        menubar.add_cascade(label="Preprocessing",menu =filemenu_Preprocessing)

        filemenu_Analysis = tk.Menu(menubar, tearoff=0)
        Analysis_Classification_sub_menu = tk.Menu(filemenu_Analysis, tearoff=0)
        Analysis_PCA_sub_menu = tk.Menu(filemenu_Analysis, tearoff=0)
        Analysis_PCOA_sub_menu = tk.Menu(filemenu_Analysis, tearoff=0)

        Analysis_PCA_sub_menu.add_command(label="PCA Plotting" ,command =self.PCA_plotting)
        Analysis_PCA_sub_menu.add_command(label="Component Heat Map" )
        Analysis_PCA_sub_menu.add_command(label="PCA Loading plot")

        Analysis_PCOA_sub_menu.add_command(label="PCOA" )
        Analysis_PCOA_sub_menu.add_command(label="Peak picking" )

        Analysis_Classification_sub_menu.add_command(label="XGBOOST" )
        Analysis_Classification_sub_menu.add_command(label="KNN" )
        Analysis_Classification_sub_menu.add_command(label="Decision Trees" )
        Analysis_Classification_sub_menu.add_command(label="Support vector Machines" )

        filemenu_Analysis.add_cascade(label="PCA Analysis",menu =  Analysis_PCA_sub_menu)
        filemenu_Analysis.add_cascade(label="PCoA Analysis", menu= Analysis_PCOA_sub_menu)
        filemenu_Analysis.add_cascade(label="Classification", menu =Analysis_Classification_sub_menu)
        menubar.add_cascade(label="Anaylsis", menu=filemenu_Analysis)

        menubar.add_cascade(label="EDIT" )
        # default canvas
        self.default_canvas()



        tk.Tk.config(self,menu = menubar)
        self.frames = {}
        for F in (StartPage, PageOne):
            frame = F(container,self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")


        self.show_frame(StartPage)

    def default_canvas(self):
        # default frame in the starting page
        self.default_frame = tk.Frame(self, background="red")
        self.default_frame.place(x=400, y=50, height=900, width=1500)
        self.canvas = tk.Canvas(self.default_frame, background='white')
        self.canvas.place(x=0, y=0, height=900, width=1500)

        myscrollbar = ttk.Scrollbar(self.default_frame, orient="vertical", command=self.canvas.yview)
        myscrollbarx = ttk.Scrollbar(self.default_frame, orient="horizontal", command=self.canvas.xview)
        myscrollbar.pack(side="right", fill="y")
        myscrollbarx.pack(side="bottom", fill="x")
        self.canvas.configure(yscrollcommand=myscrollbar.set, xscrollcommand=myscrollbarx.set)
        self.canvas.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.default_frame2 = tk.Frame(self.canvas, background="red")
        # self.frame2.place(x=0, y=0, height=600, width=1200)
        self.canvas.create_window((0, 0), window=self.default_frame2, anchor="nw")

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

    def clear_listbox(self):
        self.lbox.delete(0,tk.END)

    def clear_selected_item(self):
        sel = self.lbox.curselection()
        for index in sel[::-1]:
            self.lbox.delete(index)

    def getcsv(self):
            try:
                # import_file_path = filedialog.askopenfilename()
                answer = messagebox.askyesnocancel("Do you want to select a folder", "click yes for selecting a folder or No for selecting a file?")
                if answer == 1 :
                    import_folder_path = filedialog.askdirectory()
                    print(import_folder_path)
                    file_names = os.listdir(import_folder_path)
                    print(file_names)
                    for file in file_names:
                        self.lbox.insert(tk.END, import_folder_path + '/' + file)
                elif answer == 0:
                    my_filetypes = [('csv files', '.csv')]
                    import_file_path = filedialog.askopenfilenames(
                                         title="Please select one or more files:",
                                         filetypes=my_filetypes)
                    print(import_file_path)
                    #file_names = os.listdir(import_file_path)
                    #print(list(import_file_path))
                    for file in list(import_file_path):
                        self.lbox.insert(tk.END,  file)
            except:
                print('Something went wrong while reading files')


    def readfile(self,file,lower_limit,upper_limit):
        file = pd.read_csv(file, sep=";")
        file = file.loc[(file['Average_Mass'] >= lower_limit) & (file['Average_Mass'] < upper_limit)]
        file = file.iloc[:, 0:2]
        return file

    def compare_spectrum(self):
        self.default_canvas()
        num  = len(self.lbox.curselection())
        lower_limit = 20
        upper_limit = 200
        fig = Figure(figsize=(15, 8), constrained_layout=True)
        for i in range(num):
            items = self.lbox.curselection()
            result = self.lbox.get(items[i])
            file = result[1]
            filename =  os.path.basename(file)
            file = self.readfile(file,lower_limit,upper_limit)

            # Create the figure and the line that we will manipulate
            if num == 1:
                print('first')

                ax = fig.add_subplot(1,1,1)
                ax.plot(file.iloc[:, 0], file.iloc[:, 1], c='r')
                ax.set_ylabel('intensity')
                ax.set_xlabel('m/z')
                ax.set_title(f'{filename}')

            elif num == 2:
                print('second')

                ax = fig.add_subplot(2, 1, i+1)
                ax.plot(file.iloc[:, 0], file.iloc[:, 1], c='r')
                ax.set_ylabel('intensity')
                ax.set_xlabel('m/z')
                ax.set_title(f'{filename}')
            else:
                print('you have selected 0 or more than 2 files')
        canvas = FigureCanvasTkAgg(fig, self.default_frame2)
        canvas.draw()
        canvas.get_tk_widget().place(x=0, y=0, height=600, width=1000)
        toolbar = NavigationToolbar2Tk(self.default_frame2, self.default_frame2)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)



    def preprocessing_peak_pick_pop_up(self):


        # import_file_path = filedialog.askopenfilename()
        answer = messagebox.askyesnocancel("Do you want to select a folder",
                                           "click yes for selecting a folder or No for selecting a file?")
        if answer == 1:
            import_folder_path = filedialog.askdirectory()
            print(import_folder_path)
            file_names = os.listdir(import_folder_path)
            print(file_names)
            self.peak_pick_pop_up_label(import_folder_path,file_names)


        elif answer == 0:
            my_filetypes = [('csv files', '.csv')]
            import_file_path = filedialog.askopenfilenames(
                title="Please select one or more files:",
                filetypes=my_filetypes)
            print(import_file_path)
            files_list = []
            for files in list(import_file_path):
                dirname = os.path.dirname(files)
                files_list.append(os.path.basename(files))
            self.peak_pick_pop_up_label(dirname, files_list)


    def get_size_lb(self):
        print(self.lbox.size())


        #except:
         #   print('Something went wrong while reading files')

    def add_files_to_list_box(self,path,file_list,label):
        for file in file_list:
            self.lbox.insert(tk.END, [str(label.get()),path + '/' + file])
        self.pop_up_peak_pick.destroy()
        self.preprocessing_peak_pick_pop_up()

    def proceed_to_peak_pick(self,path,file_list,label):
        for file in file_list:
            self.lbox.insert(tk.END, [str(label.get()),path + '/' + file])
        self.pop_up_peak_pick.destroy()
        self.peak_picking()


    def peak_pick_pop_up_label(self,path,file_list):
        global pop_up_peak_pick
        self.pop_up_peak_pick = Toplevel()

        self.path = path
        self.file_list =file_list

        self.canvas_pop_up_peak_pick = tk.Canvas(self.pop_up_peak_pick, width=400, height=350, relief='raised')
        self.canvas_pop_up_peak_pick.pack(fill='both', expand=True)

        self.label_bins = tk.Label(self.pop_up_peak_pick, text='Enter minimum peak', relief='flat')
        self.label_bins.config(font=('helvetica', 10))
        self.canvas_pop_up_peak_pick.create_window(200, 20, window=self.label_bins)

        self.peak_bins_entry = tk.Entry(self.pop_up_peak_pick, width=25, relief='raised')
        self.canvas_pop_up_peak_pick.create_window(200, 50, window=self.peak_bins_entry)

        self.lower_mz_bound_label = tk.Label(self.pop_up_peak_pick, text='Enter lower m/z bound', relief='flat')
        self.lower_mz_bound_label.config(font=('helvetica', 10))
        self.canvas_pop_up_peak_pick.create_window(200, 80, window=self.lower_mz_bound_label)

        self.mz_lower_entry = tk.Entry(self.pop_up_peak_pick, width=25, relief='raised')
        self.canvas_pop_up_peak_pick.create_window(200, 110, window=self.mz_lower_entry)

        self.upper_mz_bound_label = tk.Label(self.pop_up_peak_pick, text='Enter upper m/z bound', relief='flat')
        self.upper_mz_bound_label.config(font=('helvetica', 10))
        self.canvas_pop_up_peak_pick.create_window(200, 140, window=self.upper_mz_bound_label)

        self.mz_upper_entry = tk.Entry(self.pop_up_peak_pick, width=25, relief='raised')
        self.canvas_pop_up_peak_pick.create_window(200, 170, window=self.mz_upper_entry)

        self.label_class = tk.Label(self.pop_up_peak_pick, text='Enter label for the data', relief='flat')
        self.label_class.config(font=('helvetica', 10))
        self.canvas_pop_up_peak_pick.create_window(200, 200, window=self.label_class)

        self.class_entry = tk.Entry(self.pop_up_peak_pick, width=25, relief='raised')
        self.canvas_pop_up_peak_pick.create_window(200, 230, window=self.class_entry)

        self.peak_pick_Add_more = tk.Button(self.pop_up_peak_pick, text=" Add more data ", command=partial(self.add_files_to_list_box,path,file_list,self.class_entry),
                                          bg='green', fg='white',
                                          font=('helvetica', 10, 'bold'))
        self.canvas_pop_up_peak_pick.create_window(200, 270, window=self.peak_pick_Add_more)

        self.peak_pick_button = tk.Button(self.pop_up_peak_pick, text=" Proceed ", command=partial(self.proceed_to_peak_pick,path,file_list,self.class_entry),
                                          bg='green', fg='white',
                                          font=('helvetica', 10, 'bold'))
        self.canvas_pop_up_peak_pick.create_window(200, 310, window=self.peak_pick_button)

    def peak_picking(self):

        items = self.lbox.get(0, tk.END)
        print(items[0][0])
        df_list = []
        for files in  items:
            print(files[1])
            file = pd.read_csv( files[1], sep=';')
            file['label'] = files[0]
            file = file.drop(columns=['Current_Mass', 'Current'])
            file = file.loc[(file['Average_Mass'] > 20) & (file['Average_Mass'] < 200)].reset_index(drop=True)
            df_list.append(file)

        total_df = pd.concat(df_list).reset_index(drop=True)
        print('######')
        print(total_df)
        nom_labels = total_df['label'].unique()
        labels_unique = total_df['label'].nunique()
        num_labels = [i for i in range(0,labels_unique)]
        print(nom_labels)
        print(num_labels)
        for i in range(0,len(nom_labels)):
            total_df["label"].replace({nom_labels[i]: num_labels[i] }, inplace=True)

        #le = preprocessing.LabelEncoder()
        #total_df['label'] = le.fit_transform(total_df['label'])
        print('######')
        print(total_df)

        mass_per_charge = total_df.iloc[:, 0]
        ion_intensity = total_df.iloc[:, 1]
        peaks, indices = find_peaks(ion_intensity, height=250)
        mass_per_charge = mass_per_charge.take(peaks).to_frame()
        mass_per_charge = mass_per_charge.sort_values(by=['Average_Mass'])
        mass_per_charge = mass_per_charge.reset_index(drop=True)

        mass_per_charge_list = mass_per_charge['Average_Mass'].values.tolist()
        # print(mass_per_charge_list)
        attribute = []
        pop_count = 0
        duplicate_1 = mass_per_charge_list.copy()
        duplicate_2 = mass_per_charge_list.copy()
        for item_1 in mass_per_charge_list:

            for item_2 in duplicate_1:

                if abs(item_1 - item_2) <= 0.2 and abs(item_1 - item_2) != 0:
                    if item_2 in duplicate_2:
                        duplicate_2.remove(item_2)

            duplicate_1.pop(0)

        self.final_mz = np.unique(duplicate_2).tolist()
        # print(final_mz)

        final_data_list = []

        for files in df_list:
            mass_per_charge_files = files.iloc[:, 0]
            ion_intensity_files = files.iloc[:, 1]
            peaks_files, indices_files = find_peaks(ion_intensity_files, height=250)
            mass_per_charge_files = mass_per_charge_files.take(peaks_files).to_frame()
            # mass_per_charge_files = mass_per_charge_files.sort_values(by=['Average_Mass'])
            # mass_per_charge_files = mass_per_charge_files.reset_index(drop=True)

            mass_per_charge_list_files = mass_per_charge_files['Average_Mass'].values.tolist()

            peak_ion_intensities_files = pd.DataFrame.from_dict(indices_files)

            file = pd.concat([mass_per_charge_files.reset_index(drop=True), peak_ion_intensities_files], axis=1)
            # print(file)
            mz_1 = list(file['Average_Mass'])
            peak_heights = list(file['peak_heights'])
            mz_df_1 = pd.DataFrame(self.final_mz, columns=['Average_Mass'])

            mz_df_1['ion_intensity'] = 0

            for i in self.final_mz:
                for j in mz_1:
                    if i <= j + 0.2:
                        if abs(i - j) <= 0.2:
                            mz_df_1.loc[mz_df_1.Average_Mass == i, 'ion_intensity'] = file.loc[
                                file['Average_Mass'] == j, 'peak_heights'].item()

            mz_df_1 = list(mz_df_1.iloc[:, 1])
            # print(mz_df_1)
            mz_df_1 = pd.DataFrame(mz_df_1).transpose()
            mz_df_1.columns = self.final_mz
            # print(mz_df_1)
            mz_df_1['label'] = files.iat[0, 2]
            # print(mz_df_1)
            final_data_list.append(mz_df_1)


        self.peak_pick_spectrum_data = pd.concat(final_data_list).reset_index(drop=True)
        print(self.peak_pick_spectrum_data)

        self.binning_spectrum()
    def peak_pick_spectrum(self):
        self.default_canvas()
        num = len(self.lbox.curselection())

        fig = Figure(figsize=(15, 8), constrained_layout=True)
        for i in range(num):
            items = self.lbox.curselection()
            result = self.lbox.get(items[i])
            file = result[1]
            filename = os.path.basename(file)
            #file = self.readfile(file, lower_limit, upper_limit)

            # Create the figure and the line that we will manipulate
            if num == 1:
                print('first')

                ax = fig.add_subplot(1, 1, 1)
                x = self.peak_pick_spectrum_data.columns
                ax.bar(x[0:-1], self.peak_pick_spectrum_data.iloc[items[i]+1,0:-1] )
                ax.set_ylabel('intensity')
                ax.set_xlabel('m/z')
                ax.set_title(f'{filename}')

            elif num == 2:
                print('second')

                ax = fig.add_subplot(2, 1, i + 1)
                x = self.peak_pick_spectrum_data.columns
                ax.bar(x[0:-1], self.peak_pick_spectrum_data.iloc[items[i] + 1, 0:-1] )
                ax.set_ylabel('intensity')
                ax.set_xlabel('m/z')
                ax.set_title(f'{filename}')
            else:
                print('you have selected 0 or more than 2 files')
        canvas = FigureCanvasTkAgg(fig, self.default_frame2)
        canvas.draw()
        canvas.get_tk_widget().place(x=0, y=0, height=600, width=1000)
        toolbar = NavigationToolbar2Tk(canvas, self.default_frame2)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def binning_spectrum(self):
        items = self.lbox.get(0, tk.END)
        print(items[0][0])
        df_list = []
        for files in items:
            print(files[1])
            file = pd.read_csv(files[1], sep=';')
            file['label'] = files[0]
            file = file.drop(columns=['Current_Mass', 'Current'])
            file = file.loc[(file['Average_Mass'] > 20) & (file['Average_Mass'] < 200)].reset_index(drop=True)


            from scipy.stats import binned_statistic

            x_bins, bin_edges, misc = binned_statistic(file['Average_Mass'], file['Average'], statistic="max", bins=100)

            bin_intervals = pd.IntervalIndex.from_arrays(bin_edges[:-1], bin_edges[1:])
            x_bins = pd.DataFrame(np.transpose(x_bins))
            x_bins = x_bins.transpose()
            x_bins['label'] = file.iat[0, 2]
            df_list.append(x_bins)

        total_df = pd.concat(df_list).reset_index(drop=True)
        print('######')
        print(total_df)
        nom_labels = total_df['label'].unique()
        labels_unique = total_df['label'].nunique()
        num_labels = [i for i in range(0, labels_unique)]
        print(nom_labels)
        print(num_labels)
        for i in range(0, len(nom_labels)):
            total_df["label"].replace({nom_labels[i]: num_labels[i]}, inplace=True)

        # le = preprocessing.LabelEncoder()
        # total_df['label'] = le.fit_transform(total_df['label'])
        print('######')
        self.binning_spectrum_data = total_df

    def view_binning(self):
        self.default_canvas()
        num = len(self.lbox.curselection())

        fig = Figure(figsize=(15, 8), constrained_layout=True)
        for i in range(num):
            items = self.lbox.curselection()
            result = self.lbox.get(items[i])
            file = result[1]
            filename = os.path.basename(file)
            #file = self.readfile(file, lower_limit, upper_limit)

            # Create the figure and the line that we will manipulate
            if num == 1:
                print('first')

                ax = fig.add_subplot(1, 1, 1)
                x = self.binning_spectrum_data.columns
                ax.bar(x[0:-1], self.binning_spectrum_data.iloc[items[i]+1,0:-1] )
                ax.set_ylabel('intensity')
                ax.set_xlabel('m/z')
                ax.set_title(f'{filename}')

            elif num == 2:
                print('second')

                ax = fig.add_subplot(2, 1, i + 1)
                x = self.binning_spectrum_data.columns
                ax.bar(x[0:-1], self.binning_spectrum_data.iloc[items[i] + 1, 0:-1] )
                ax.set_ylabel('intensity')
                ax.set_xlabel('m/z')
                ax.set_title(f'{filename}')
            else:
                print('you have selected 0 or more than 2 files')
        canvas = FigureCanvasTkAgg(fig, self.default_frame2)
        canvas.draw()
        canvas.get_tk_widget().place(x=0, y=0, height=600, width=1000)
        toolbar = NavigationToolbar2Tk(canvas, self.default_frame2)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def Feature_Correlation(self):
        PTR = pd.read_csv('PTR_Library.csv').dropna()
        compound_names = {}
        for value in self.final_mz:
            PT2 = PTR.loc[PTR['m/z A'].astype(int) == int(value)]
            if PT2.empty:
                empty_list = ['unknown']
                compound_names.update({int(value): empty_list})
            else:
                compounds = PT2['Compound Name'].to_list()
                compound_names.update({int(value): compounds})

        print(compound_names)


        col = [compound_names[key][0] for key in compound_names]
        col.append('label')
        l1 = []
        l2 = []
        for key in compound_names:
            l1.append(str(key))
            l2.append(compound_names[key][0])
        res = [i + '_' + j for i, j in zip(l1, l2)]
        res.append('label')
        data = self.peak_pick_spectrum_data
        data.columns = res
        print(data)

        data_heatmap = data.drop(['label'], axis=1)
        correlations = data_heatmap.corr()

        fig = Figure(figsize=(8, 8), constrained_layout=True)
        ax = fig.add_subplot(1, 1, 1)
        #ax.plot(file.iloc[:, 0], file.iloc[:, 1], c='r')
        #ax.set_ylabel('intensity')
        #ax.set_xlabel('m/z')
        #ax.set_title(f'{filename}')

        # Set up the matplotlib figure
        #fig, ax = plt.subplots(figsize=(15, 15))

        sns.heatmap(correlations,cmap='RdBu', ax=ax)
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
        for tick in ax.get_yticklabels():
            tick.set_rotation(0)

        #sns.set(style="white")
        # Generate a custom diverging colormap
        #cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        #sns.heatmap(correlations,   cmap=cmap, vmax=.3, center=0,
        #            square=True, linewidths=.5, cbar_kws={"shrink": .5})

        canvas = FigureCanvasTkAgg(fig, self.default_frame2)
        canvas.draw()
        canvas.get_tk_widget().place(x=0, y=0, height=600, width=1000)
        toolbar = NavigationToolbar2Tk(canvas, self.default_frame2)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def cluster_heat_map(self):
        PTR = pd.read_csv('PTR_Library.csv').dropna()
        compound_names = {}
        for value in self.final_mz:
            PT2 = PTR.loc[PTR['m/z A'].astype(int) == int(value)]
            if PT2.empty:
                empty_list = ['unknown']
                compound_names.update({int(value): empty_list})
            else:
                compounds = PT2['Compound Name'].to_list()
                compound_names.update({int(value): compounds})

        print(compound_names)

        col = [compound_names[key][0] for key in compound_names]
        col.append('label')
        l1 = []
        l2 = []
        for key in compound_names:
            l1.append(str(key))
            l2.append(compound_names[key][0])
        res = [i + '_' + j for i, j in zip(l1, l2)]
        res.append('label')
        data = self.peak_pick_spectrum_data
        data.columns = res
        print(data)



        import seaborn as sns;
        sns.set_theme(color_codes=True)

        colors = ['r','b','g','c','m','y','k','w']
        n = len(np.unique(data['label']))
        color_dict = dict(zip(np.unique(data['label']), np.array(colors[0:n])))
        target_df = pd.DataFrame({"label": data.label})
        row_colors = target_df.label.map(color_dict)
        # species = iris.pop("species")

        fig = Figure(figsize=(8, 8), constrained_layout=True)
        ax = fig.add_subplot(1, 1, 1)
        data = data.drop(['label'], axis=1)
        sns.clustermap(data, standard_scale=1, row_colors=row_colors, cmap='RdBu', annot=True )
        plt.savefig('clustermap.png')
        import imageio


        from PIL import ImageTk, Image
        img = ImageTk.PhotoImage(Image.open('clustermap.png'))
        #self.canvas.create_image(20, 20, anchor=NW, image=self.img)
        #self.canvas.image = self.img

        # for tick in ax.get_xticklabels():
        #     tick.set_rotation(90)
        # for tick in ax.get_yticklabels():
        #     tick.set_rotation(0)
        label = Label(self.default_frame2, image=img)
        label.pack()

    def PCA_plotting(self):
        PTR = pd.read_csv('PTR_Library.csv').dropna()
        compound_names = {}
        for value in self.final_mz:
            PT2 = PTR.loc[PTR['m/z A'].astype(int) == int(value)]
            if PT2.empty:
                empty_list = ['unknown']
                compound_names.update({int(value): empty_list})
            else:
                compounds = PT2['Compound Name'].to_list()
                compound_names.update({int(value): compounds})

        print(compound_names)

        col = [compound_names[key][0] for key in compound_names]
        col.append('label')
        l1 = []
        l2 = []
        for key in compound_names:
            l1.append(str(key))
            l2.append(compound_names[key][0])
        res = [i + '_' + j for i, j in zip(l1, l2)]
        res.append('label')
        data = self.peak_pick_spectrum_data
        data.columns = res
        print(data)



        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        data1 = data.drop(['label'],axis = 1)
        principalComponents = pca.fit_transform(data1)
        principalDf = pd.DataFrame(data=principalComponents
                                   , columns=['principal component 1', 'principal component 2'])

        finalDf = pd.concat([principalDf, data[['label']]], axis=1)
        print(finalDf)
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
        ax.set_title('2 component PCA', fontsize=20)
        colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'w']
        n = len(np.unique(data['label']))
        color_dict = dict(zip(np.unique(data['label']), np.array(colors[0:n])))
        print(color_dict)
        target = np.unique(data['label'])
        targets = {0: 'A549', 1: 'MDA', 2: 'FB'}
        #

        for i in range(0,n):
            df = finalDf.loc[finalDf['label'] == target[i]]
            print(df)
            ax.scatter(df.iloc[:,0],df.iloc[:,1],color = colors[i])
        ax.legend(target)
        ax.grid()
        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
        ax.set_title('2 component PCA', fontsize=20)

        canvas = FigureCanvasTkAgg(fig, self.default_frame2)
        canvas.draw()
        canvas.get_tk_widget().place(x=0, y=0, height=600, width=1000)
        toolbar = NavigationToolbar2Tk(canvas, self.default_frame2)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)




class StartPage(tk.Frame):
    def __init__(self,parent,controller):
        tk.Frame.__init__(self,parent)

        self.scrollbar = Scrollbar(self, orient=tk.VERTICAL)
        self.lbox = tk.Listbox(self, height=50, width=50, yscrollcommand=self.scrollbar.set, selectmode=MULTIPLE)
        self.scrollbar.config(command=self.lbox.yview)
        self.scrollbar.pack(side=RIGHT, fill=tk.Y)
        # .pack(side = LEFT)
        self.lbox.place(x=20, y=200)


class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text='PCA Analysis', font=LARGE_FONT)

app = PTR_MS_analysis()
app.mainloop()