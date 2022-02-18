import tkinter as tk
from tkinter import ttk
from tkinter import *
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
from matplotlib.figure import Figure
from tkinter import filedialog
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
        filemenu_file.add_command(label="New")

        self.scrollbar = Scrollbar(self, orient=tk.VERTICAL)
        self.lbox = tk.Listbox(self, height=50, width=50, yscrollcommand=self.scrollbar.set, selectmode=MULTIPLE)
        self.scrollbar.config(command=self.lbox.yview)
        self.scrollbar.pack(side=RIGHT, fill=tk.Y)
        # .pack(side = LEFT)
        self.lbox.place(x=20, y=200)


        filemenu_file.add_command(label="Open", command = self.getcsv)
        filemenu_file.add_command(label="EXIT",command =quit)
        menubar.add_cascade(label ="FILE",menu =filemenu_file)

        filemenu_View = tk.Menu(menubar, tearoff=0)
        filemenu_View.add_command(label="Spectrum",command = self.comparespectra)
        menubar.add_cascade(label="View",menu =filemenu_View)

        filemenu_Preprocessing = tk.Menu(menubar, tearoff=0)
        filemenu_Preprocessing.add_command(label="Peak picking", command=self.peak_pick_pop_up)
        filemenu_Preprocessing.add_command(label="Binning", command=self.binning_pop_up)
        menubar.add_cascade(label="Preprocessing",menu =filemenu_Preprocessing)

        filemenu_Analysis = tk.Menu(menubar, tearoff=0)
        Analysis_Classification_sub_menu = tk.Menu(filemenu_Analysis, tearoff=0)
        Analysis_PCA_sub_menu = tk.Menu(filemenu_Analysis, tearoff=0)
        Analysis_PCOA_sub_menu = tk.Menu(filemenu_Analysis, tearoff=0)

        Analysis_PCA_sub_menu.add_command(label="Binning", command=self.comparespectra)
        Analysis_PCA_sub_menu.add_command(label="Peak picking", command=self.comparespectra)

        Analysis_PCOA_sub_menu.add_command(label="Binning", command=self.comparespectra)
        Analysis_PCOA_sub_menu.add_command(label="Peak picking", command=self.comparespectra)

        Analysis_Classification_sub_menu.add_command(label="XGBOOST", command=self.comparespectra)
        Analysis_Classification_sub_menu.add_command(label="KNN", command=self.comparespectra)
        Analysis_Classification_sub_menu.add_command(label="Decision Trees", command=self.comparespectra)
        Analysis_Classification_sub_menu.add_command(label="Support vector Machines", command=self.comparespectra)

        filemenu_Analysis.add_cascade(label="PCA Analysis",menu =  Analysis_PCA_sub_menu)
        filemenu_Analysis.add_cascade(label="PCoA Analysis", menu= Analysis_PCOA_sub_menu)
        filemenu_Analysis.add_cascade(label="Classification", menu =Analysis_Classification_sub_menu)
        menubar.add_cascade(label="Anaylsis", menu=filemenu_Analysis)






        tk.Tk.config(self,menu = menubar)
        self.frames = {}
        for F in (StartPage, PageOne,PageTwo,PageThree):
            frame = F(container,self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")


        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()



    def getcsv(self):
            global df
            # import_file_path = filedialog.askopenfilename()
            import_folder_path = filedialog.askdirectory()
            print(import_folder_path)
            file_names = os.listdir(import_folder_path)
            print(file_names)
            for file in file_names:
                self.lbox.insert(tk.END, import_folder_path + '/' + file)

    def binning_pop_up(self):
        global pop_up_binning
        pop_up_binning = Toplevel()

        canvas_pop_up_binning = tk.Canvas(pop_up_binning, width=400, height=300, relief='raised')
        canvas_pop_up_binning.pack(fill='both', expand=True)

        label_bins = tk.Label(pop_up_binning, text='Enter min peak voltage', relief='flat')
        label_bins.config(font=('helvetica', 10))
        canvas_pop_up_binning.create_window(200, 50, window=label_bins)

        peak_bins_entry = tk.Entry(pop_up_binning, width=25, relief='raised')
        canvas_pop_up_binning.create_window(200, 80, window=peak_bins_entry)

        lower_mz_bound_label = tk.Label(pop_up_binning, text='Enter lower m/z bound', relief='flat')
        lower_mz_bound_label.config(font=('helvetica', 10))
        canvas_pop_up_binning.create_window(200, 110, window=lower_mz_bound_label)

        mz_lower_entry = tk.Entry(pop_up_binning, width=25, relief='raised')
        canvas_pop_up_binning.create_window(200, 140, window=mz_lower_entry)

        upper_mz_bound_label = tk.Label(pop_up_binning, text='Enter upper m/z bound', relief='flat')
        upper_mz_bound_label.config(font=('helvetica', 10))
        canvas_pop_up_binning.create_window(200, 170, window=upper_mz_bound_label)

        mz_upper_entry = tk.Entry(pop_up_binning, width=25, relief='raised')
        canvas_pop_up_binning.create_window(200, 200, window=mz_upper_entry)

        binning_button = tk.Button(pop_up_binning,text=" Proceed ", command=self.binning, bg='green', fg='white',
                                      font=('helvetica', 10, 'bold'))
        canvas_pop_up_binning.create_window(200, 230, window=binning_button)

    def peak_pick_pop_up(self):
        global pop_up_peak_pick
        pop_up_peak_pick = Toplevel()

        canvas_pop_up_peak_pick = tk.Canvas(pop_up_peak_pick, width=400, height=300, relief='raised')
        canvas_pop_up_peak_pick.pack(fill='both', expand=True)

        label_bins = tk.Label(pop_up_peak_pick, text='Enter number of bins', relief='flat')
        label_bins.config(font=('helvetica', 10))
        canvas_pop_up_peak_pick.create_window(200, 50, window=label_bins)

        peak_bins_entry = tk.Entry(pop_up_peak_pick, width=25, relief='raised')
        canvas_pop_up_peak_pick.create_window(200, 80, window=peak_bins_entry)

        lower_mz_bound_label = tk.Label(pop_up_peak_pick, text='Enter lower m/z bound', relief='flat')
        lower_mz_bound_label.config(font=('helvetica', 10))
        canvas_pop_up_peak_pick.create_window(200, 110, window=lower_mz_bound_label)

        mz_lower_entry = tk.Entry(pop_up_peak_pick, width=25, relief='raised')
        canvas_pop_up_peak_pick.create_window(200, 140, window=mz_lower_entry)

        upper_mz_bound_label = tk.Label(pop_up_peak_pick, text='Enter upper m/z bound', relief='flat')
        upper_mz_bound_label.config(font=('helvetica', 10))
        canvas_pop_up_peak_pick.create_window(200, 170, window=upper_mz_bound_label)

        mz_upper_entry = tk.Entry(pop_up_peak_pick, width=25, relief='raised')
        canvas_pop_up_peak_pick.create_window(200, 200, window=mz_upper_entry)

        peak_pick_button = tk.Button(pop_up_peak_pick,text=" Proceed ", command=self.binning, bg='green', fg='white',
                                      font=('helvetica', 10, 'bold'))
        canvas_pop_up_peak_pick.create_window(200, 230, window=peak_pick_button)




    def comparespectra(self):

        # self.canvas = tk.Canvas(self, width=300, height=200, background='white')
        # self.canvas.place(x=400, y=200, height=600, width=1000)

        self.frame1 = tk.Frame(self, background="red")
        self.frame1.place(x=400, y=200, height=600, width=1200)
        self.canvas = tk.Canvas(self.frame1, background='white')
        self.canvas.place(x=0, y=0, height=600, width=1200)

        myscrollbar = ttk.Scrollbar(self.frame1, orient="vertical", command=self.canvas.yview)
        myscrollbarx = ttk.Scrollbar(self.frame1, orient="horizontal", command=self.canvas.xview)
        myscrollbar.pack(side="right", fill="y")
        myscrollbarx.pack(side="bottom", fill="x")
        self.canvas.configure(yscrollcommand=myscrollbar.set, xscrollcommand=myscrollbarx.set)
        self.canvas.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.frame2 = tk.Frame(self.canvas, background="red")
        # self.frame2.place(x=0, y=0, height=600, width=1200)
        self.canvas.create_window((0, 0), window=self.frame2, anchor="nw")

        rows = len(self.lbox.curselection())
        print(rows)

        fig = Figure(figsize=( 10,10))
        spectrum = 1
        for i in range(len(self.lbox.curselection())):
            items = self.lbox.curselection()
            result = self.lbox.get(items[i])

            file = pd.read_csv(result, sep=";")
            file = file.loc[(file['Average_Mass'] >= 20) & (file['Average_Mass'] < 100)]
            file = file.iloc[:, 0:2]
            #fig = plt.Figure(figsize=(4, 3), dpi=100)
            #ax1 = fig.add_subplot(111)
            #rows= len(self.lbox.curselection())
            #print(rows)

            #fig = Figure(figsize=(10,rows*5))
            fig.add_subplot(2,2,i+1).plot(file.iloc[:, 0], file.iloc[:, 1], c='r')
            #ax.set_xlabel('M/Z ', fontsize=15)
            #ax.set_ylabel('intensity', fontsize=15)
            #ax.set_title('Breathe Gas spectrum', fontsize=20)
            #for i in range(0, rows):
            #ax.plot(file.iloc[:, 0], file.iloc[:, 1], c='r')


        canvas = FigureCanvasTkAgg(fig, self.frame2)
        canvas.draw()
        canvas.get_tk_widget().place(x=0, y=0, height=600, width=1000)
        toolbar = NavigationToolbar2Tk(canvas, self.frame2)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)



class StartPage(tk.Frame):
    def __init__(self,parent,controller):
        tk.Frame.__init__(self,parent)
        label =tk.Label(self, text = 'Home Page', font = LARGE_FONT)
        label.pack(padx = 10, pady = 10)
        #label.grid(row=0, column = 0)

        button1 =ttk.Button(self,text= 'PCA Analysis' , command= lambda:controller.show_frame(PageOne))
        button1.pack()
        button2 = ttk.Button(self, text=' PCOA Analysis', command=lambda: controller.show_frame(PageTwo))
        button2.pack()
        button3 = ttk.Button(self, text='Classification', command=lambda: controller.show_frame(PageThree))
        button3.pack()
        upload_data_button = ttk.Button(self, text='upload Data', command=self.getcsv)
        upload_data_button.pack(pady=10, padx=10)

        compare_spectra_button = Button(self, text='Compare spectra', command=self.comparespectra, bg='green', fg='white',
                                        font=('helvetica', 10, 'bold'))
        compare_spectra_button.pack(pady=10, padx=10)

        # self.canvas = tk.Canvas(self, width=300, height=200, background='white')
        # self.canvas.place(x=400, y=200, height=600, width=1000)

        self.scrollbar = Scrollbar(self, orient=tk.VERTICAL)
        self.lbox = tk.Listbox(self, height=50, width=50, yscrollcommand=self.scrollbar.set, selectmode=MULTIPLE)
        self.scrollbar.config(command=self.lbox.yview)
        self.scrollbar.pack(side=RIGHT, fill=tk.Y)
        # .pack(side = LEFT)
        self.lbox.place(x=20, y=200)

    def getcsv(self):
        global df
        # import_file_path = filedialog.askopenfilename()
        import_folder_path = filedialog.askdirectory()
        print(import_folder_path)
        file_names = os.listdir(import_folder_path)
        print(file_names)
        for file in file_names:
            self.lbox.insert(tk.END, import_folder_path+'/'+ file)

    def comparespectra(self):

        # self.canvas = tk.Canvas(self, width=300, height=200, background='white')
        # self.canvas.place(x=400, y=200, height=600, width=1000)

        self.frame1 = tk.Frame(self, background="red")
        self.frame1.place(x=400, y=200, height=600, width=1200)
        self.canvas = tk.Canvas(self.frame1, background='white')
        self.canvas.place(x=0, y=0, height=600, width=1200)

        myscrollbar = ttk.Scrollbar(self.frame1, orient="vertical", command=self.canvas.yview)
        myscrollbarx = ttk.Scrollbar(self.frame1, orient="horizontal", command=self.canvas.xview)
        myscrollbar.pack(side="right", fill="y")
        myscrollbarx.pack(side="bottom", fill="x")
        self.canvas.configure(yscrollcommand=myscrollbar.set, xscrollcommand=myscrollbarx.set)
        self.canvas.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.frame2 = tk.Frame(self.canvas, background="red")
        # self.frame2.place(x=0, y=0, height=600, width=1200)
        self.canvas.create_window((0, 0), window=self.frame2, anchor="nw")
        row = 0
        for items in self.lbox.curselection():
            result = self.lbox.get(items)
            file = pd.read_csv(result, sep=";")
            file = file.loc[(file['Average_Mass'] >= 20) & (file['Average_Mass'] < 100)]
            file = file.iloc[:, 0:2]
            #fig = plt.Figure(figsize=(4, 3), dpi=100)
            #ax1 = fig.add_subplot(111)
            rows= len(self.lbox.curselection())
            print(rows)

            fig = Figure(figsize=(10,6))
            ax = fig.add_subplot(111)
            ax.set_xlabel('M/Z ', fontsize=15)
            ax.set_ylabel('intensity', fontsize=15)
            ax.set_title('Breathe Gas spectrum', fontsize=20)
            #for i in range(0, rows):
            ax.plot(file.iloc[:, 0], file.iloc[:, 1], c='r')
            canvas = FigureCanvasTkAgg(fig, self.frame2)
            canvas.draw()
            canvas.get_tk_widget().place(x=0+row, y=0, height=600, width=1000)
            toolbar = NavigationToolbar2Tk(canvas, self.frame2)
            toolbar.update()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            row =+1

class PageOne(tk.Frame):
    def __init__(self,parent,controller):
        tk.Frame.__init__(self,parent)
        label =tk.Label(self, text = 'PCA Analysis', font = LARGE_FONT)
        label.pack(padx = 10, pady = 10)
        #label.grid(row=0, column = 0)

        button2 =ttk.Button(self,text= 'Back to Home' , command= lambda : controller.show_frame(StartPage))
        button2.pack()
        button2 = ttk.Button(self, text='PCA Analysis', command=lambda: controller.show_frame(PageTwo))
        button2.pack()
        button3 = ttk.Button(self, text='Classification', command=lambda: controller.show_frame(PageThree))
        button3.pack()

        button_binning_spectrum = ttk.Button(self,text= 'binning' , command= self.binning)
        button_binning_spectrum.place(x=10, y=10, height=25, width=100)
        button_peak_picking_spectrum = ttk.Button(self, text='peak_picking', command=self.peak_picking)
        button_peak_picking_spectrum.place(x=10, y=50, height=25, width=100)
        #
        # distances = ('jaccard', 'braycurtis', 'cityblock', 'cosine', 'euclidean', 'mahalanobis',
        #           'minkowski', 'dice', 'hamming')
        #
        # label = ttk.Label(text="Please select a distance measure:")
        # label.place(x=10, y=100, height=25, width=100)
        # def month_changed(event):
        #      distance =selected_month.get()
        #      print(distance)
        #
        # # create a combobox
        # selected_month = tk.StringVar()
        #
        # month_cb = ttk.Combobox(self, value = distances )
        # month_cb.current(0)
        # month_cb.place(x=10, y=250, height=25, width=100)
        # month_cb.bind('<<ComboboxSelected>>', month_changed)





    def peak_picking(self):
        path_A549 = 'D:/MASTERS/Melanie_6CP/Dataset_PT_MSR/A549/'
        path_MDA = 'D:/MASTERS/Melanie_6CP/Dataset_PT_MSR/MDA/'
        path_FB = 'D:/MASTERS/Melanie_6CP/Dataset_PT_MSR/FB/'
        path_A549_stress = 'D:/MASTERS/Melanie_6CP/Dataset_PT_MSR/A549_stress/'
        path_MDA_stress = 'D:/MASTERS/Melanie_6CP/Dataset_PT_MSR/MDA_stress/'
        path_FB_stress = 'D:/MASTERS/Melanie_6CP/Dataset_PT_MSR/FB_stress/'

        path_list = [path_A549, path_MDA, path_FB,path_A549_stress,path_MDA_stress,path_FB_stress]
        A549 = os.listdir(path_A549)
        MDA = os.listdir(path_MDA)
        FB = os.listdir(path_FB)
        A549_stress = os.listdir(path_A549_stress)
        MDA_stress = os.listdir(path_MDA_stress)
        FB_stress = os.listdir(path_FB_stress)

        label = [0, 1, 2,3,4,5]
        list_1 = [A549, MDA, FB,A549_stress,MDA_stress,FB_stress]
        df_list = []
        for i in range(0, 6):
            path = path_list[i]
            print(path)
            j = 0
            for file in list_1[i]:
                print(file)
                file = pd.read_csv(path + str(file), sep=';')
                print(i)
                file['class'] = label[i]
                file = file.drop(columns=['Current_Mass', 'Current'])
                file = file.loc[(file['Average_Mass'] > 20) & (file['Average_Mass'] < 200)].reset_index(drop=True)
                j += 1
                df_list.append(file)

        total_df = pd.concat(df_list).reset_index(drop=True)
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

        final_mz = np.unique(duplicate_2).tolist()
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
            mz_df_1 = pd.DataFrame(final_mz, columns=['Average_Mass'])

            mz_df_1['ion_intensity'] = 0

            for i in final_mz:
                for j in mz_1:
                    if i <= j + 0.2:
                        if abs(i - j) <= 0.2:
                            mz_df_1.loc[mz_df_1.Average_Mass == i, 'ion_intensity'] = file.loc[
                                file['Average_Mass'] == j, 'peak_heights'].item()

            mz_df_1 = list(mz_df_1.iloc[:, 1])
            # print(mz_df_1)
            mz_df_1 = pd.DataFrame(mz_df_1).transpose()
            mz_df_1['label'] = files.iat[0, 2]
            # print(mz_df_1)
            final_data_list.append(mz_df_1)

        data = pd.concat(final_data_list).reset_index(drop=True)
        print(data)

        data1 = data.loc[:, data.columns != 'label']

        data1 = pd.DataFrame(scaler.fit_transform(data1))

        kmeans = KMeans(n_clusters=3)

        kmeans.fit(data1)

        # Find which cluster each data-point belongs to
        clusters = kmeans.predict(data1)

        data["Cluster"] = clusters

        print(data)

        comparison_column = np.where(data["label"] == data["Cluster"], 1, 0)

        print(sum(comparison_column))

        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(data1)
        principalDf = pd.DataFrame(data=principalComponents
                                   , columns=['principal component 1', 'principal component 2'])

        finalDf = pd.concat([principalDf, data[['label']]], axis=1)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
        ax.set_title('2 component PCA', fontsize=20)
        targets = [0, 1, 2,3,4,5]
        legends = ['A549', 'MDA', 'FB', 'A549_stress', 'MDA_stress', 'FB_stress']
        colors = ['r', 'g', 'b','c','m','y']
        for target, color in zip(targets, colors):
            indicesToKeep = finalDf['label'] == target
            ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                       , finalDf.loc[indicesToKeep, 'principal component 2']
                       , c=color
                       , s=50)
        fig.suptitle('PCA using Peak Picking', fontsize=16)
        ax.legend(legends)
        ax.grid()

        self.frame1 = tk.Frame(self, background="red")
        self.frame1.place(x=400, y=150, height=800, width=1200)
        self.canvas = tk.Canvas(self.frame1, background='white')
        self.canvas.place(x=0, y=0, height=800, width=1200)

        myscrollbar = ttk.Scrollbar(self.frame1, orient="vertical", command=self.canvas.yview)
        myscrollbarx = ttk.Scrollbar(self.frame1, orient="horizontal", command=self.canvas.xview)
        myscrollbar.pack(side="right", fill="y")
        myscrollbarx.pack(side="bottom", fill="x")
        self.canvas.configure(yscrollcommand=myscrollbar.set, xscrollcommand=myscrollbarx.set)
        self.canvas.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.frame2 = tk.Frame(self.canvas, background="red")
        # self.frame2.place(x=0, y=0, height=600, width=1200)
        self.canvas.create_window((0, 0), window=self.frame2, anchor="nw")

        canvas = FigureCanvasTkAgg(fig, self.frame2)
        canvas.draw()
        canvas.get_tk_widget().place(x=0, y=0, height=800, width=1200)
        toolbar = NavigationToolbar2Tk(canvas, self.frame2)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)



        #plt.show()
    def binning(self):
        path_A549 = 'D:/MASTERS/Melanie_6CP/Dataset_PT_MSR/A549/'
        path_MDA = 'D:/MASTERS/Melanie_6CP/Dataset_PT_MSR/MDA/'
        path_FB = 'D:/MASTERS/Melanie_6CP/Dataset_PT_MSR/FB/'
        path_A549_stress = 'D:/MASTERS/Melanie_6CP/Dataset_PT_MSR/A549_stress/'
        path_MDA_stress = 'D:/MASTERS/Melanie_6CP/Dataset_PT_MSR/MDA_stress/'
        path_FB_stress = 'D:/MASTERS/Melanie_6CP/Dataset_PT_MSR/FB_stress/'

        path_list = [path_A549, path_MDA,path_FB, path_A549_stress, path_MDA_stress, path_FB_stress]
        A549 = os.listdir(path_A549)
        MDA = os.listdir(path_MDA)
        FB = os.listdir(path_FB)
        A549_stress = os.listdir(path_A549_stress)
        MDA_stress = os.listdir(path_MDA_stress)
        FB_stress = os.listdir(path_FB_stress)

        label = [0, 1, 2, 3, 4,5]
        list_1 = [A549, MDA, FB, A549_stress, MDA_stress, FB_stress]
        df_list = []
        for i in range(0, 6):
            path = path_list[i]
            j = 0
            for file in list_1[i]:
                file = pd.read_csv(path + str(file), sep=';')
                file['class'] = label[i]
                file = file.drop(columns=['Current_Mass', 'Current'])
                file = file.loc[(file['Average_Mass'] > 20) & (file['Average_Mass'] < 100)].reset_index(drop=True)
                # print(file)
                x_data = file['Average_Mass']
                y_data = file['Average']

                from scipy.stats import binned_statistic

                x_bins, bin_edges, misc = binned_statistic(x_data, y_data, statistic="max", bins=100)

                bin_intervals = pd.IntervalIndex.from_arrays(bin_edges[:-1], bin_edges[1:])
                x_bins = pd.DataFrame(np.transpose(x_bins))
                x_bins = x_bins.transpose()
                x_bins['label'] = file.iat[0, 2]
                df_list.append(x_bins)

        data = pd.concat(df_list).reset_index(drop=True)

        def set_to_median(x, bin_intervals):
            for interval in bin_intervals:
                if x in interval:
                    return interval.mid

        data1 = data.loc[:, data.columns != 'label']

        data1 = pd.DataFrame(scaler.fit_transform(data1))

        kmeans = KMeans(n_clusters=3)

        kmeans.fit(data1)
        #
        # Find which cluster each data-point belongs to

        clusters = kmeans.predict(data1)

        data["Cluster"] = clusters

        print(data)

        comparison_column = np.where(data["label"] == data["Cluster"], 1, 0)

        print(sum(comparison_column))

        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(data1)
        principalDf = pd.DataFrame(data=principalComponents
                                   , columns=['principal component 1', 'principal component 2'])

        finalDf = pd.concat([principalDf, data[['label']]], axis=1)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
        ax.set_title('2 component PCA', fontsize=20)
        fig.suptitle('PCA using Binning Method', fontsize=16)
        targets = [0, 1, 2, 3, 4, 5]
        legends = ['A549', 'MDA', 'FB', 'A549_stress', 'MDA_stress', 'FB_stress']
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        for target, color in zip(targets, colors):
            indicesToKeep = finalDf['label'] == target
            ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                       , finalDf.loc[indicesToKeep, 'principal component 2']
                       , c=color
                       , s=50)
        ax.legend(legends)
        ax.grid()
        self.frame1 = tk.Frame(self, background="red")
        self.frame1.place(x=400, y=150, height=800, width=1200)
        self.canvas = tk.Canvas(self.frame1, background='white')
        self.canvas.place(x=0, y=0, height=800, width=1200)

        myscrollbar = ttk.Scrollbar(self.frame1, orient="vertical", command=self.canvas.yview)
        myscrollbarx = ttk.Scrollbar(self.frame1, orient="horizontal", command=self.canvas.xview)
        myscrollbar.pack(side="right", fill="y")
        myscrollbarx.pack(side="bottom", fill="x")
        self.canvas.configure(yscrollcommand=myscrollbar.set, xscrollcommand=myscrollbarx.set)
        self.canvas.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.frame2 = tk.Frame(self.canvas, background="red")
        # self.frame2.place(x=0, y=0, height=600, width=1200)
        self.canvas.create_window((0, 0), window=self.frame2, anchor="nw")

        canvas = FigureCanvasTkAgg(fig, self.frame2)
        canvas.draw()
        canvas.get_tk_widget().place(x=0, y=0, height=800, width=1200)
        toolbar = NavigationToolbar2Tk(canvas, self.frame2)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        #plt.show()

class PageTwo(tk.Frame):
    def __init__(self,parent,controller):
        tk.Frame.__init__(self,parent)
        label =tk.Label(self, text = 'PCOA Analysis', font = LARGE_FONT)
        label.pack(padx = 10, pady = 10)
        #label.grid(row=0, column = 0)

        button1 =ttk.Button(self,text= 'Back to Home' , command= lambda : controller.show_frame(StartPage))
        button1.pack()
        button2 = ttk.Button(self, text='PCA Analysis', command=lambda: controller.show_frame(PageOne))
        button2.pack()
        button3 = ttk.Button(self, text='Classification', command=lambda: controller.show_frame(PageThree))
        button3.pack()
        button_PCOA = ttk.Button(self, text='PCOA with peaking', command=self.PCOA)
        button_PCOA.place(x=10, y=50, height=25, width=150)

    def PCOA(self):
        path_A549 = 'D:/MASTERS/Melanie_6CP/Dataset_PT_MSR/A549/'
        path_MDA = 'D:/MASTERS/Melanie_6CP/Dataset_PT_MSR/MDA/'
        path_FB = 'D:/MASTERS/Melanie_6CP/Dataset_PT_MSR/FB/'
        path_A549_stress = 'D:/MASTERS/Melanie_6CP/Dataset_PT_MSR/A549_stress/'
        path_MDA_stress = 'D:/MASTERS/Melanie_6CP/Dataset_PT_MSR/MDA_stress/'
        path_FB_stress = 'D:/MASTERS/Melanie_6CP/Dataset_PT_MSR/FB_stress/'
        path_list = [path_A549, path_MDA, path_FB, path_A549_stress, path_MDA_stress,path_FB_stress]
        A549 = os.listdir(path_A549)
        MDA = os.listdir(path_MDA)
        FB = os.listdir(path_FB)
        A549_stress = os.listdir(path_A549_stress)
        MDA_stress = os.listdir(path_MDA_stress)
        FB_stress = os.listdir(path_FB_stress)

        label = ['A549', 'MDA', 'FB', 'A549_stress', 'MDA_stress','FB_stress']
        list_1 = [A549, MDA, FB, A549_stress, MDA_stress,FB_stress]
        df_list = []
        for i in range(0, 6):
            path = path_list[i]
            j = 0
            for file in list_1[i]:
                file = pd.read_csv(path + str(file), sep=';')
                file['class'] = label[i]
                file = file.drop(columns=['Current_Mass', 'Current'])
                file = file.loc[(file['Average_Mass'] > 20) & (file['Average_Mass'] < 100)].reset_index(drop=True)
                j += 1
                df_list.append(file)

        def pcoa_plot(DF_data):

            from scipy.spatial import distance

            Ar_MxMdistance = distance.squareform(distance.pdist(DF_data.T, metric="cityblock"))
            DF_dism = pd.DataFrame(Ar_MxMdistance, index=DF_data.columns, columns=DF_data.columns)
            print(DF_dism.to_string())

            DF_dism = DF_dism.fillna(0)
            import skbio

            # Compute the Principal Coordinates Analysis
            my_pcoa = skbio.stats.ordination.pcoa(DF_dism.values)

            # Show the new coordinates for our cities
            return my_pcoa

        plot_list = []
        for peak in range(100, 1000, 500):
            total_df = pd.concat(df_list).reset_index(drop=True)
            mass_per_charge = total_df.iloc[:, 0]
            ion_intensity = total_df.iloc[:, 1]
            peaks, indices = find_peaks(ion_intensity, height=peak)
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

            final_mz = np.unique(duplicate_2).tolist()
            # print(final_mz)

            final_data_list = []

            for files in df_list:
                mass_per_charge_files = files.iloc[:, 0]
                ion_intensity_files = files.iloc[:, 1]
                peaks_files, indices_files = find_peaks(ion_intensity_files, height=peak)
                mass_per_charge_files = mass_per_charge_files.take(peaks_files).to_frame()
                # mass_per_charge_files = mass_per_charge_files.sort_values(by=['Average_Mass'])
                # mass_per_charge_files = mass_per_charge_files.reset_index(drop=True)

                mass_per_charge_list_files = mass_per_charge_files['Average_Mass'].values.tolist()

                peak_ion_intensities_files = pd.DataFrame.from_dict(indices_files)

                file = pd.concat([mass_per_charge_files.reset_index(drop=True), peak_ion_intensities_files], axis=1)
                # print(file)
                mz_1 = list(file['Average_Mass'])
                peak_heights = list(file['peak_heights'])
                mz_df_1 = pd.DataFrame(final_mz, columns=['Average_Mass'])

                mz_df_1['ion_intensity'] = 0

                for i in final_mz:
                    for j in mz_1:
                        if i <= j + 0.2:
                            if abs(i - j) <= 0.2:
                                mz_df_1.loc[mz_df_1.Average_Mass == i, 'ion_intensity'] = file.loc[
                                    file['Average_Mass'] == j, 'peak_heights'].item()

                mz_df_1 = list(mz_df_1.iloc[:, 1])
                # print(mz_df_1)
                mz_df_1 = pd.DataFrame(mz_df_1).transpose()
                mz_df_1['label'] = files.iat[0, 2]
                # print(mz_df_1)
                final_data_list.append(mz_df_1)

            DF_data = pd.concat(final_data_list).reset_index(drop=True)
            #print(DF_data)
            DF_data = DF_data.drop(['label'], axis=1)
            print(DF_data)
            #data = DF_data.iloc[0:24, :]
            #print(data)
            A549_pcoa = pcoa_plot(DF_data.iloc[0:24, :])
            MDA_pcoa = pcoa_plot(DF_data.iloc[24:53, :])
            FB_pcoa = pcoa_plot(DF_data.iloc[53:71, :])
            A549_stress_pcoa = pcoa_plot(DF_data.iloc[71:89, :])
            MDA_stress_pcoa = pcoa_plot(DF_data.iloc[89:107, :])
            FB_stress_pcoa = pcoa_plot(DF_data.iloc[107:, :])
            # print(my_pcoa.samples[['PC1', 'PC2']])

            import matplotlib.pyplot as plt

            self.frame1 = tk.Frame(self, background="red")
            self.frame1.place(x=400, y=150, height=800, width=1200)
            self.canvas = tk.Canvas(self.frame1, background='white')
            self.canvas.place(x=0, y=0, height=800, width=1200)

            myscrollbar = ttk.Scrollbar(self.frame1, orient="vertical", command=self.canvas.yview)
            myscrollbarx = ttk.Scrollbar(self.frame1, orient="horizontal", command=self.canvas.xview)
            myscrollbar.pack(side="right", fill="y")
            myscrollbarx.pack(side="bottom", fill="x")
            self.canvas.configure(yscrollcommand=myscrollbar.set, xscrollcommand=myscrollbarx.set)
            self.canvas.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

            self.frame2 = tk.Frame(self.canvas, background="red")
            # self.frame2.place(x=0, y=0, height=600, width=1200)
            self.canvas.create_window((0, 0), window=self.frame2, anchor="nw")

            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xlabel('Principal Cordinate 1', fontsize=15)
            ax.set_ylabel('Principal Cordinate 2', fontsize=15)
            fig.suptitle('PCoA using Peak picking', fontsize=16)

            ax.scatter(A549_pcoa.samples['PC1'], A549_pcoa.samples['PC2'], marker='o')
            ax.scatter(MDA_pcoa.samples['PC1'], MDA_pcoa.samples['PC2'], marker='v')
            ax.scatter(FB_pcoa.samples['PC1'], FB_pcoa.samples['PC2'], marker='s')
            ax.scatter(A549_stress_pcoa.samples['PC1'], A549_stress_pcoa.samples['PC2'], marker='+')
            ax.scatter(MDA_stress_pcoa.samples['PC1'], MDA_stress_pcoa.samples['PC2'], marker='*')
            ax.scatter(FB_stress_pcoa.samples['PC1'], FB_stress_pcoa.samples['PC2'], marker='x')
            ax.legend(label)
            #ax.show()


            canvas = FigureCanvasTkAgg(fig, self.frame2)
            canvas.draw()
            canvas.get_tk_widget().place(x=0, y=0, height=800, width=1200)
            toolbar = NavigationToolbar2Tk(canvas, self.frame2)
            toolbar.update()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)




class PageThree(tk.Frame):
    def __init__(self,parent,controller):
        tk.Frame.__init__(self,parent)
        label =tk.Label(self, text = 'Classification', font = LARGE_FONT)
        label.pack(padx = 10, pady = 10)
        #label.grid(row=0, column = 0)

        button1 =ttk.Button(self,text= 'Back to Home' , command= lambda : controller.show_frame(StartPage))
        button1.pack()
        button2 = ttk.Button(self, text='PCA analysis', command=lambda: controller.show_frame(PageOne))
        button2.pack()
        button3 = ttk.Button(self, text='PCOA Analysis', command=lambda: controller.show_frame(PageTwo))
        button3.pack()

        button_SVM = ttk.Button(self, text='SVM', command=self.SVM)
        button_SVM.place(x=10, y=50, height=25, width=150)
        button_XgBoost = ttk.Button(self, text='XgBoost', command=self.XgBoost)
        button_XgBoost.place(x=10, y=100, height=25, width=150)
        button_KNN = ttk.Button(self, text='KNN', command=self.KNN)
        button_KNN.place(x=10, y=150, height=25, width=150)
        button_Decision_trees = ttk.Button(self, text='Decision_trees', command=self.XgBoost)
        button_Decision_trees.place(x=10, y=200, height=25, width=150)

        self.frame1 = tk.Frame(self, background="red")
        self.frame1.place(x=400, y=150, height=800, width=1200)
        self.canvas = tk.Canvas(self.frame1, background='white')
        self.canvas.place(x=0, y=0, height=800, width=1200)

        self.myscrollbar = ttk.Scrollbar(self.frame1, orient="vertical", command=self.canvas.yview)
        self.myscrollbarx = ttk.Scrollbar(self.frame1, orient="horizontal", command=self.canvas.xview)
        self.myscrollbar.pack(side="right", fill="y")
        self.myscrollbarx.pack(side="bottom", fill="x")
        self.canvas.configure(yscrollcommand=self.myscrollbar.set, xscrollcommand=self.myscrollbarx.set)
        self.canvas.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.frame2 = tk.Frame(self.canvas, background="red")
        # self.frame2.place(x=0, y=0, height=600, width=1200)
        self.canvas.create_window((0, 0), window=self.frame2, anchor="nw")

        # f = Figure(figsize =(5,5), dpi = 100)
        # a = f.add_subplot(111)
        # a.plot([1,2,3,4,5,6,7,8], [5,6,1,3,8,9,3,5])
        # canvas  = FigureCanvasTkAgg(f,self)
        # canvas.draw()
        # canvas.get_tk_widget().pack(side =tk.TOP,fill =tk.BOTH,expand = True)
        # toolbar = NavigationToolbar2Tk(canvas,self)
        # toolbar.update()
        # canvas.get_tk_widget().pack(side =tk.TOP,fill =tk.BOTH,expand =True)


    def XgBoost(self):
        # evaluate a give model using cross-validation
        def evaluate_model(model, X, y):
            # define the evaluation procedure
            cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=3, random_state=1)
            # evaluate the model
            scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
            print(scores)

        import os
        import pandas as pd
        import numpy as np

        import matplotlib.pyplot as plt

        import seaborn as sns
        sns.set()
        from scipy.signal import find_peaks

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

        from sklearn.preprocessing import LabelEncoder, StandardScaler

        import sklearn.model_selection as model_selection

        path_A549 = 'D:/MASTERS/Melanie_6CP/Dataset_PT_MSR/A549/'
        path_MDA = 'D:/MASTERS/Melanie_6CP/Dataset_PT_MSR/MDA/'
        path_FB = 'D:/MASTERS/Melanie_6CP/Dataset_PT_MSR/FB/'
        path_list = [path_A549, path_MDA, path_FB]
        A549 = os.listdir(path_A549)
        MDA = os.listdir(path_MDA)
        FB = os.listdir(path_FB)
        label = ['A549', 'MDA', 'FB']
        list_1 = [A549, MDA, FB]
        df_list = []
        for i in range(0, 3):
            path = path_list[i]
            j = 0
            for file in list_1[i]:
                file = pd.read_csv(path + str(file), sep=';')
                file['class'] = label[i]
                file = file.drop(columns=['Current_Mass', 'Current'])
                file = file.loc[(file['Average_Mass'] > 20) & (file['Average_Mass'] < 100)].reset_index(drop=True)
                j += 1
                df_list.append(file)

        plot_list = []
        for peak in range(100, 1000, 500):
            total_df = pd.concat(df_list).reset_index(drop=True)
            mass_per_charge = total_df.iloc[:, 0]
            ion_intensity = total_df.iloc[:, 1]
            peaks, indices = find_peaks(ion_intensity, height=peak)
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

            final_mz = np.unique(duplicate_2).tolist()
            # print(final_mz)

            final_data_list = []

            for files in df_list:
                mass_per_charge_files = files.iloc[:, 0]
                ion_intensity_files = files.iloc[:, 1]
                peaks_files, indices_files = find_peaks(ion_intensity_files, height=peak)
                mass_per_charge_files = mass_per_charge_files.take(peaks_files).to_frame()
                # mass_per_charge_files = mass_per_charge_files.sort_values(by=['Average_Mass'])
                # mass_per_charge_files = mass_per_charge_files.reset_index(drop=True)

                mass_per_charge_list_files = mass_per_charge_files['Average_Mass'].values.tolist()

                peak_ion_intensities_files = pd.DataFrame.from_dict(indices_files)

                file = pd.concat([mass_per_charge_files.reset_index(drop=True), peak_ion_intensities_files], axis=1)
                # print(file)
                mz_1 = list(file['Average_Mass'])
                peak_heights = list(file['peak_heights'])
                mz_df_1 = pd.DataFrame(final_mz, columns=['Average_Mass'])

                mz_df_1['ion_intensity'] = 0

                for i in final_mz:
                    for j in mz_1:
                        if i <= j + 0.2:
                            if abs(i - j) <= 0.2:
                                mz_df_1.loc[mz_df_1.Average_Mass == i, 'ion_intensity'] = file.loc[
                                    file['Average_Mass'] == j, 'peak_heights'].item()

                mz_df_1 = list(mz_df_1.iloc[:, 1])
                # print(mz_df_1)
                mz_df_1 = pd.DataFrame(mz_df_1).transpose()
                mz_df_1['label'] = files.iat[0, 2]
                # print(mz_df_1)
                final_data_list.append(mz_df_1)

            data = pd.concat(final_data_list).reset_index(drop=True)
            print(data)

            label_encoder = LabelEncoder()

            columns = list(data.columns.values)
            X_columns = columns[0:-1]
            Y_columns = [columns[-1]]
            # print(Y_columns)

            X = data[X_columns].values

            Y = data[Y_columns].values

            label_encoder = LabelEncoder()
            Y = label_encoder.fit_transform(Y)
            X = StandardScaler().fit_transform(X)

            X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, train_size=0.80, test_size=0.20,
                                                                                random_state=101)
            print('training......................', X_train.shape)
            print(f"for peak {peak}")
            # # get the models to evaluate
            model = XGBRFClassifier()
            #visualizer = ClassificationReport(model, classes=label, support=True)

            # visualizer.fit(X_train, y_train)  # Fit the visualizer and the model
            # visualizer.score(X_test, y_test)  # Evaluate the model on the test data
            # fig = visualizer.show()
            # fig = plt.imshow(visualizer)
            # plt.savefig(visualizer)
            model = model.fit(X_train, y_train)

            y_predicted = model.predict(X_test)

            target_names = ['A549', 'MDA', 'FB']

            report = classification_report(y_test, y_predicted, target_names=target_names,output_dict=True)
            df = pd.DataFrame(report).transpose()

            df.to_excel('XG_boost_classification_report.xlsx')
            conf_matrix = confusion_matrix(y_test, y_predicted)

            fig, ax = plt.subplots(figsize=(8,8))
            ax.matshow(conf_matrix, cmap=plt.cm.viridis, alpha=0.3)
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

            plt.xlabel('Predictions', fontsize=18)
            plt.ylabel('Actuals', fontsize=18)
            plt.title('Confusion Matrix for XGBoost', fontsize=18)
            #plt.show()

            #fig = plt.figure(figsize=(12, 8))
            #ax = fig.add_subplot(1, 1, 1)
            #ax.set_xlabel('Principal Cordinate 1', fontsize=15)
            #ax.set_ylabel('Principal Cordinate 2', fontsize=15)
            #fig.suptitle('PCoA using Peak picking', fontsize=16)

            # fig, ax = plot_confusion_matrix(conf_mat=conf_matrix, figsize=(6, 6), cmap=plt.cm.Greens)
            # plt.xlabel('Predictions', fontsize=18)
            # plt.ylabel('Actuals', fontsize=18)
            # plt.title('Confusion Matrix', fontsize=18)
            # plt.show()


            # canvas = FigureCanvasTkAgg(visualizer, self.frame2)
            # canvas.draw()
            # canvas.get_tk_widget().place(x=0, y=0, height=800, width=1200)
            # toolbar = NavigationToolbar2Tk(canvas, self.frame2)
            # toolbar.update()
            # canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            canvas = FigureCanvasTkAgg(fig, self.frame2)
            canvas.draw()
            canvas.get_tk_widget().place(x=0, y=0, height=800, width=1200)
            toolbar = NavigationToolbar2Tk(canvas, self.frame2)
            toolbar.update()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def SVM(self):
        path_A549 = 'D:/MASTERS/Melanie_6CP/Dataset_PT_MSR/A549/'
        path_MDA = 'D:/MASTERS/Melanie_6CP/Dataset_PT_MSR/MDA/'
        path_FB = 'D:/MASTERS/Melanie_6CP/Dataset_PT_MSR/FB/'
        path_list = [path_A549, path_MDA, path_FB]
        A549 = os.listdir(path_A549)
        MDA = os.listdir(path_MDA)
        FB = os.listdir(path_FB)
        label = ['A549', 'MDA', 'FB']
        list_1 = [A549, MDA, FB]
        df_list = []
        for i in range(0, 3):
            path = path_list[i]
            j = 0
            for file in list_1[i]:
                file = pd.read_csv(path + str(file), sep=';')
                file['class'] = label[i]
                file = file.drop(columns=['Current_Mass', 'Current'])
                file = file.loc[(file['Average_Mass'] > 20) & (file['Average_Mass'] < 100)].reset_index(drop=True)
                j += 1
                df_list.append(file)

        for peak in range(100, 10000, 500):
            total_df = pd.concat(df_list).reset_index(drop=True)
            mass_per_charge = total_df.iloc[:, 0]
            ion_intensity = total_df.iloc[:, 1]
            peaks, indices = find_peaks(ion_intensity, height=peak)
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

            final_mz = np.unique(duplicate_2).tolist()
            # print(final_mz)

            final_data_list = []

            for files in df_list:
                mass_per_charge_files = files.iloc[:, 0]
                ion_intensity_files = files.iloc[:, 1]
                peaks_files, indices_files = find_peaks(ion_intensity_files, height=peak)
                mass_per_charge_files = mass_per_charge_files.take(peaks_files).to_frame()
                # mass_per_charge_files = mass_per_charge_files.sort_values(by=['Average_Mass'])
                # mass_per_charge_files = mass_per_charge_files.reset_index(drop=True)

                mass_per_charge_list_files = mass_per_charge_files['Average_Mass'].values.tolist()

                peak_ion_intensities_files = pd.DataFrame.from_dict(indices_files)

                file = pd.concat([mass_per_charge_files.reset_index(drop=True), peak_ion_intensities_files], axis=1)
                # print(file)
                mz_1 = list(file['Average_Mass'])
                peak_heights = list(file['peak_heights'])
                mz_df_1 = pd.DataFrame(final_mz, columns=['Average_Mass'])

                mz_df_1['ion_intensity'] = 0

                for i in final_mz:
                    for j in mz_1:
                        if i <= j + 0.2:
                            if abs(i - j) <= 0.2:
                                mz_df_1.loc[mz_df_1.Average_Mass == i, 'ion_intensity'] = file.loc[
                                    file['Average_Mass'] == j, 'peak_heights'].item()

                mz_df_1 = list(mz_df_1.iloc[:, 1])
                # print(mz_df_1)
                mz_df_1 = pd.DataFrame(mz_df_1).transpose()
                mz_df_1['label'] = files.iat[0, 2]
                # print(mz_df_1)
                final_data_list.append(mz_df_1)

            data = pd.concat(final_data_list).reset_index(drop=True)
            print(data)

            #label_encoder = LabelEncoder()

            columns = list(data.columns.values)
            X_columns = columns[0:-1]
            Y_columns = [columns[-1]]
            print(Y_columns)

            X = data[X_columns].values
            Y = data[Y_columns].values

            label_encoder = LabelEncoder()
            Y = label_encoder.fit_transform(Y)

            X = StandardScaler().fit_transform(X)

            X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, train_size=0.80, test_size=0.20,
                                                                                random_state=101)
            #
            rbf = svm.SVC(kernel='rbf', gamma=1, C=0.1).fit(X_train, y_train)
            poly = svm.SVC(kernel='poly', degree=3, C=1).fit(X_train, y_train)

            poly_pred = poly.predict(X_test)
            rbf_pred = rbf.predict(X_test)
            target_names = ['A549', 'MDA', 'FB']

            report = classification_report(y_test, poly_pred, target_names=target_names, output_dict=True)
            df = pd.DataFrame(report).transpose()

            df.to_excel('SVM_classification_report.xlsx')
            conf_matrix = confusion_matrix(y_test, poly_pred)

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.matshow(conf_matrix, cmap=plt.cm.viridis, alpha=0.3)
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

            plt.xlabel('Predictions', fontsize=18)
            plt.ylabel('Actuals', fontsize=18)
            plt.title('Confusion Matrix for SVM', fontsize=18)


            canvas = FigureCanvasTkAgg(fig, self.frame2)
            canvas.draw()
            canvas.get_tk_widget().place(x=0, y=0, height=800, width=1200)
            toolbar = NavigationToolbar2Tk(canvas, self.frame2)
            toolbar.update()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def KNN(self):
        from sklearn.neighbors import KNeighborsClassifier

        path_A549 = 'D:/MASTERS/Melanie_6CP/Dataset_PT_MSR/A549/'
        path_MDA = 'D:/MASTERS/Melanie_6CP/Dataset_PT_MSR/MDA/'
        path_FB = 'D:/MASTERS/Melanie_6CP/Dataset_PT_MSR/FB/'
        path_list = [path_A549, path_MDA, path_FB]
        A549 = os.listdir(path_A549)
        MDA = os.listdir(path_MDA)
        FB = os.listdir(path_FB)
        label = ['A549', 'MDA', 'FB']
        list_1 = [A549, MDA, FB]
        df_list = []
        for i in range(0, 3):
            path = path_list[i]
            j = 0
            for file in list_1[i]:
                file = pd.read_csv(path + str(file), sep=';')
                file['class'] = label[i]
                file = file.drop(columns=['Current_Mass', 'Current'])
                file = file.loc[(file['Average_Mass'] > 20) & (file['Average_Mass'] < 100)].reset_index(drop=True)
                j += 1
                df_list.append(file)

        for peak in range(100, 10000, 500):
            total_df = pd.concat(df_list).reset_index(drop=True)
            mass_per_charge = total_df.iloc[:, 0]
            ion_intensity = total_df.iloc[:, 1]
            peaks, indices = find_peaks(ion_intensity, height=peak)
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

            final_mz = np.unique(duplicate_2).tolist()
            # print(final_mz)

            final_data_list = []

            for files in df_list:
                mass_per_charge_files = files.iloc[:, 0]
                ion_intensity_files = files.iloc[:, 1]
                peaks_files, indices_files = find_peaks(ion_intensity_files, height=peak)
                mass_per_charge_files = mass_per_charge_files.take(peaks_files).to_frame()
                # mass_per_charge_files = mass_per_charge_files.sort_values(by=['Average_Mass'])
                # mass_per_charge_files = mass_per_charge_files.reset_index(drop=True)

                mass_per_charge_list_files = mass_per_charge_files['Average_Mass'].values.tolist()

                peak_ion_intensities_files = pd.DataFrame.from_dict(indices_files)

                file = pd.concat([mass_per_charge_files.reset_index(drop=True), peak_ion_intensities_files], axis=1)
                # print(file)
                mz_1 = list(file['Average_Mass'])
                peak_heights = list(file['peak_heights'])
                mz_df_1 = pd.DataFrame(final_mz, columns=['Average_Mass'])

                mz_df_1['ion_intensity'] = 0

                for i in final_mz:
                    for j in mz_1:
                        if i <= j + 0.2:
                            if abs(i - j) <= 0.2:
                                mz_df_1.loc[mz_df_1.Average_Mass == i, 'ion_intensity'] = file.loc[
                                    file['Average_Mass'] == j, 'peak_heights'].item()

                mz_df_1 = list(mz_df_1.iloc[:, 1])
                # print(mz_df_1)
                mz_df_1 = pd.DataFrame(mz_df_1).transpose()
                mz_df_1['label'] = files.iat[0, 2]
                # print(mz_df_1)
                final_data_list.append(mz_df_1)

            data = pd.concat(final_data_list).reset_index(drop=True)
            print(data)

            label_encoder = LabelEncoder()

            columns = list(data.columns.values)
            X_columns = columns[0:-1]
            Y_columns = [columns[-1]]
            print(Y_columns)

            X = data[X_columns].values
            Y = data[Y_columns].values

            label_encoder = LabelEncoder()
            Y = label_encoder.fit_transform(Y)
            X = StandardScaler().fit_transform(X)

            X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, train_size=0.80, test_size=0.20,
                                                                                random_state=101)
            print(f"for peak {peak}")
            # get the models to evaluate
            # knn = KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)

            model = KNeighborsClassifier(n_neighbors=7)

            model = model.fit(X_train, y_train)

            y_predicted = model.predict(X_test)

            target_names = ['A549', 'MDA', 'FB']

            report = classification_report(y_test, y_predicted, target_names=target_names, output_dict=True)
            df = pd.DataFrame(report).transpose()

            df.to_excel('KNN_classification_report.xlsx')
            conf_matrix = confusion_matrix(y_test, y_predicted)

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.matshow(conf_matrix, cmap=plt.cm.viridis, alpha=0.3)
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

            plt.xlabel('Predictions', fontsize=18)
            plt.ylabel('Actuals', fontsize=18)
            plt.title('Confusion Matrix for KNN', fontsize=18)

            canvas = FigureCanvasTkAgg(fig, self.frame2)
            canvas.draw()
            canvas.get_tk_widget().place(x=0, y=0, height=800, width=1200)
            toolbar = NavigationToolbar2Tk(canvas, self.frame2)
            toolbar.update()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


app = PTR_MS_analysis()
app.mainloop()