import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# see here: https://stackoverflow.com/questions/40659212/futurewarning-elementwise-comparison-failed-returning-scalar-but-in-the-futur
import os.path
import sys
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class dataLoader:
    """
    this class imports all the data (graph+feature+label) from the files located in dataset/object/gt/
    """

    def __init__(self, clf, verbosity=1):
        self.n_nodes = 0
        self.id = ''
        self.gt = []
        self.features = []
        self.edge_features = []
        self.edge_lists = []
        self.clf = clf
        self.verbosity = verbosity

        self.read_edge_features = clf.model.edge_convs

    def __len__(self):
        return len(self.gt)

    def reset(self):
        self.n_nodes = 0
        self.gt = []
        self.features = []
        self.edge_features = []
        self.edge_lists = []
        self.id = ''

    def getInfo(self):

        fs = self.features.size()[1] - bool(self.clf.regularization.cell_type)
        self.clf.temp.num_node_features = fs

        if(self.read_edge_features):
            fse = self.edge_features.size()[1] - bool(self.clf.regularization.edge_type)
            self.clf.temp.num_edge_features = fse
        else:
            self.clf.temp.num_edge_features = None


        if(self.verbosity):
            print("\t-{} nodes".format(self.n_nodes))
            print("\t-{} node features:".format(fs))
            if (self.clf.features.node_normalization_feature):
                print("\t-", self.node_feature_names[1:])
            else:
                print("\t-", self.node_feature_names)

            if (self.read_edge_features):
                print("\t-{} edge features:".format(fse))
                if (self.clf.features.edge_normalization_feature):
                    print("\t-", self.edge_feature_names[1:])
                else:
                    print("\t-", self.edge_feature_names)

            if (self.clf.features.node_normalization_feature):
                print("\t-shape weight feature for loss: ", self.node_feature_names[0])
            if('sum' in self.clf.features.scaling):
                print("\t-divide all features by their sum per object")
            if('s' in self.clf.features.scaling):
                print("\t-standardize features to 0 mean and 1 std")
            elif('n' in self.clf.features.scaling):
                print("\t-normalize features to interval [0...1]")
            elif('r' in self.clf.features.scaling):
                print("\t-normalize features r")
            else:
                print("\t-Warning: Features are neither standardized nor normalized")

        return self.n_nodes

    def run(self, d):

        self.path = d["path"]
        self.filename = d["filename"]
        self.category = d["category"]
        self.id = d["id"]
        self.scan_conf = d["scan_conf"]
        self.gtfile = d["gtfile"]
        self.ioufile = d["ioufile"]

        self.basefilename = os.path.join(self.path, self.gtfile)


        # read vertex features and labels
        # self.readNodeData()
        self.readNodeData_bin()

        # read adjacencies and edge features
        # self.readAdjacencies()
        self.readAdjacencies_bin()

        # read edge features
        if(self.read_edge_features):
            # self.readEdgeData()
            self.readEdgeData_bin()

        # standardize all features
        if(self.clf.features.scaling):
            self.standardizeFeatures()

        self.toTorch()
        self.n_nodes+=len(self.features)


    def readNodeData(self):

        ### print file for debugging
        # print(self.basefilename)

        self.gt = pd.read_csv(filepath_or_buffer=self.basefilename+"_labels.txt", sep = ' ', header = 0,
                        index_col = 0, dtype = np.float32)
        self.gt = torch.tensor(self.gt.values, dtype=torch.float)
        self.infinite = self.gt[:,4].type(torch.bool)


        self.features = pd.DataFrame()

        temp = pd.read_csv(filepath_or_buffer=self.basefilename+"_cgeom.txt", sep = ' ', header = 0, dtype = np.float32, index_col=False)
        self.mean_edge = (temp["longest_edge"].sum() + temp["shortest_edge"].sum()) / (2*len(temp))
        if('shape' in self.clf.features.node_features):
            self.features = pd.concat((self.features,temp),axis=1)

        if('vertex' in self.clf.features.node_features):
            temp = pd.read_csv(filepath_or_buffer=self.basefilename+"_cbvf.txt", sep = ' ', header = 0, dtype = np.float32, index_col=False)
            if('count' not in self.clf.features.node_features):
                temp.drop(labels=['cb_vertex_inside_count','cb_vertex_outside_count','cb_vertex_last_count'], axis='columns', inplace=True)
            if('min' not in self.clf.features.node_features):
                temp.drop(labels=['cb_vertex_inside_dist_min','cb_vertex_outside_dist_min','cb_vertex_last_dist_min'], axis='columns', inplace=True)
            if('max' not in self.clf.features.node_features):
                temp.drop(labels=['cb_vertex_inside_dist_max','cb_vertex_outside_dist_max','cb_vertex_last_dist_max'], axis='columns', inplace=True)
            if('sum' not in self.clf.features.node_features):
                temp.drop(labels=['cb_vertex_inside_dist_sum','cb_vertex_outside_dist_sum','cb_vertex_last_dist_sum'], axis='columns', inplace=True)
            self.features = pd.concat((self.features,temp),axis=1)

        if('facet' in self.clf.features.node_features):
            temp = pd.read_csv(filepath_or_buffer=self.basefilename+"_cbff.txt", sep = ' ', header = 0, dtype = np.float32, index_col=False)
            if('count' not in self.clf.features.node_features):
                temp.drop(labels=['cb_facet_inside_first_count','cb_facet_outside_first_count','cb_facet_last_first_count',
                                  'cb_facet_inside_second_count','cb_facet_outside_second_count','cb_facet_last_second_count'], axis='columns', inplace=True)
            if('min' not in self.clf.features.node_features):
                temp.drop(labels=['cb_facet_inside_first_dist_min','cb_facet_outside_first_dist_min','cb_facet_last_first_dist_min',
                                  'cb_facet_inside_second_dist_min','cb_facet_outside_second_dist_min', 'cb_facet_last_second_dist_min'], axis='columns', inplace=True)
            if('max' not in self.clf.features.node_features):
                temp.drop(labels=['cb_facet_inside_first_dist_max','cb_facet_outside_first_dist_max', 'cb_facet_last_first_dist_max',
                                  'cb_facet_inside_second_dist_max','cb_facet_outside_second_dist_max', 'cb_facet_last_second_dist_max'], axis='columns', inplace=True)
            if('sum' not in self.clf.features.node_features):
                temp.drop(labels=['cb_facet_inside_first_dist_sum','cb_facet_outside_first_dist_sum','cb_facet_last_first_dist_sum',
                                  'cb_facet_inside_second_dist_sum','cb_facet_outside_second_dist_sum','cb_facet_last_second_dist_sum'], axis='columns', inplace=True)
            self.features = pd.concat((self.features,temp),axis=1)

        if('last' not in self.clf.features.node_features):
            if ('vertex' in self.clf.features.node_features):
                if ('count' in self.clf.features.node_features):
                    self.features.drop(labels=['cb_vertex_last_count'], axis='columns', inplace=True)
                if ('min' in self.clf.features.node_features):
                    self.features.drop(labels=['cb_vertex_last_dist_min'], axis='columns', inplace=True)
                if ('max' in self.clf.features.node_features):
                    self.features.drop(labels=['cb_vertex_last_dist_max'], axis='columns', inplace=True)
                if ('sum' in self.clf.features.node_features):
                    self.features.drop(labels=['cb_vertex_last_dist_sum'], axis='columns', inplace=True)
            if ('facet' in self.clf.features.node_features):
                if('count' in self.clf.features.node_features):
                    self.features.drop(labels=['cb_facet_last_first_count','cb_facet_last_second_count'], axis='columns', inplace=True)
                if('min' in self.clf.features.node_features):
                    self.features.drop(labels=['cb_facet_last_first_dist_min','cb_facet_last_second_dist_min'], axis='columns', inplace=True)
                if('max' in self.clf.features.node_features):
                    self.features.drop(labels=['cb_facet_last_first_dist_max','cb_facet_last_second_dist_max'], axis='columns', inplace=True)
                if('sum' in self.clf.features.node_features):
                    self.features.drop(labels=['cb_facet_last_first_dist_sum','cb_facet_last_second_dist_sum'], axis='columns', inplace=True)

        # put a copy of the loss normalization feature at the beginning of the feature dataframe
        if(self.clf.regularization.cell_type):
            self.features.insert(0, "reg_"+self.clf.regularization.cell_type, self.features[self.clf.regularization.cell_type],allow_duplicates=False)

        self.node_feature_names = list(self.features.columns.values)

        assert(len(self.gt) == len(self.features))
        assert(not self.features.isnull().values.any())
        assert(not self.gt.isnan().any())

        return self.features


    def readNodeData_bin(self):

        ### print file for debugging
        # print(self.basefilename)

        ####################################################
        ################### ground truth ###################
        ####################################################
        temp = np.load(self.basefilename+"_labels.npz")
        if(self.clf.inference.has_label):
            self.gt = torch.Tensor([temp["inside_perc"],temp["outside_perc"]]).type(torch.float)
            self.gt = torch.transpose(self.gt, 1, 0)
        else:
            self.gt = torch.zeros(size=temp["infinite"].shape)
        self.infinite = torch.from_numpy(temp["infinite"]).type(torch.bool)

        ####################################################
        ##################### features #####################
        ####################################################
        self.features = pd.DataFrame()

        temp = np.load(self.basefilename+"_cgeom.npz")
        self.mean_edge = (temp["longest_edge"].sum() + temp["shortest_edge"].sum()) / (2*len(temp["longest_edge"]))

        if('shape' in self.clf.features.node_features):
            for key,value in temp.items():
                self.features[key] = value


        if('vertex' in self.clf.features.node_features):
            temp = np.load(self.basefilename + "_cbvf.npz")
            for key,value in temp.items():
                self.features[key] = value
            if('count' not in self.clf.features.node_features):
                self.features.drop(labels=['cb_vertex_inside_count','cb_vertex_outside_count','cb_vertex_last_count'], axis='columns', inplace=True)
            if('min' not in self.clf.features.node_features):
                self.features.drop(labels=['cb_vertex_inside_dist_min','cb_vertex_outside_dist_min','cb_vertex_last_dist_min'], axis='columns', inplace=True)
            if('max' not in self.clf.features.node_features):
                self.features.drop(labels=['cb_vertex_inside_dist_max','cb_vertex_outside_dist_max','cb_vertex_last_dist_max'], axis='columns', inplace=True)
            if('sum' not in self.clf.features.node_features):
                self.features.drop(labels=['cb_vertex_inside_dist_sum','cb_vertex_outside_dist_sum','cb_vertex_last_dist_sum'], axis='columns', inplace=True)

        if('facet' in self.clf.features.node_features):
            temp = np.load(self.basefilename+"_cbff.npz")
            for key,value in temp.items():
                self.features[key] = value
            if('count' not in self.clf.features.node_features):
                temp.drop(labels=['cb_facet_inside_first_count','cb_facet_outside_first_count','cb_facet_last_first_count',
                                  'cb_facet_inside_second_count','cb_facet_outside_second_count','cb_facet_last_second_count'], axis='columns', inplace=True)
            if('min' not in self.clf.features.node_features):
                temp.drop(labels=['cb_facet_inside_first_dist_min','cb_facet_outside_first_dist_min','cb_facet_last_first_dist_min',
                                  'cb_facet_inside_second_dist_min','cb_facet_outside_second_dist_min', 'cb_facet_last_second_dist_min'], axis='columns', inplace=True)
            if('max' not in self.clf.features.node_features):
                temp.drop(labels=['cb_facet_inside_first_dist_max','cb_facet_outside_first_dist_max', 'cb_facet_last_first_dist_max',
                                  'cb_facet_inside_second_dist_max','cb_facet_outside_second_dist_max', 'cb_facet_last_second_dist_max'], axis='columns', inplace=True)
            if('sum' not in self.clf.features.node_features):
                temp.drop(labels=['cb_facet_inside_first_dist_sum','cb_facet_outside_first_dist_sum','cb_facet_last_first_dist_sum',
                                  'cb_facet_inside_second_dist_sum','cb_facet_outside_second_dist_sum','cb_facet_last_second_dist_sum'], axis='columns', inplace=True)

        if('last' not in self.clf.features.node_features):
            if ('vertex' in self.clf.features.node_features):
                if ('count' in self.clf.features.node_features):
                    self.features.drop(labels=['cb_vertex_last_count'], axis='columns', inplace=True)
                if ('min' in self.clf.features.node_features):
                    self.features.drop(labels=['cb_vertex_last_dist_min'], axis='columns', inplace=True)
                if ('max' in self.clf.features.node_features):
                    self.features.drop(labels=['cb_vertex_last_dist_max'], axis='columns', inplace=True)
                if ('sum' in self.clf.features.node_features):
                    self.features.drop(labels=['cb_vertex_last_dist_sum'], axis='columns', inplace=True)
            if ('facet' in self.clf.features.node_features):
                if('count' in self.clf.features.node_features):
                    self.features.drop(labels=['cb_facet_last_first_count','cb_facet_last_second_count'], axis='columns', inplace=True)
                if('min' in self.clf.features.node_features):
                    self.features.drop(labels=['cb_facet_last_first_dist_min','cb_facet_last_second_dist_min'], axis='columns', inplace=True)
                if('max' in self.clf.features.node_features):
                    self.features.drop(labels=['cb_facet_last_first_dist_max','cb_facet_last_second_dist_max'], axis='columns', inplace=True)
                if('sum' in self.clf.features.node_features):
                    self.features.drop(labels=['cb_facet_last_first_dist_sum','cb_facet_last_second_dist_sum'], axis='columns', inplace=True)

        # put a copy of the loss normalization feature at the beginning of the feature dataframe
        if(self.clf.regularization.cell_type):
            self.features.insert(0, "reg_"+self.clf.regularization.cell_type, self.features[self.clf.regularization.cell_type],allow_duplicates=False)

        self.node_feature_names = list(self.features.columns.values)


        assert(self.gt.shape[0] == self.features.shape[0])
        assert(not self.features.isnull().values.any())
        assert(not self.gt.isnan().any())

        return self.features


    def readEdgeData(self):

        self.edge_features = pd.DataFrame()
        if('shape' in self.clf.features.edge_features):
            temp = pd.read_csv(filepath_or_buffer=self.basefilename+"_fgeom.txt", sep = ' ', header = 0, index_col = False, dtype = np.float32)
            self.edge_features = pd.concat((self.edge_features,temp),axis=1)

        if('vertex' in self.clf.features.edge_features):
            temp = pd.read_csv(filepath_or_buffer=self.basefilename+"_fbvf.txt", sep = ' ', header = 0, index_col=False, dtype = np.float32)
                               # skiprows=[1], usecols = [i for i in range(8)]) # because exporting was messed up, should be deleted at some point
            if('count' not in self.clf.features.edge_features):
                temp.drop(labels=['fb_vertex_inside_count','fb_vertex_outside_count'], axis='columns', inplace=True)
            if('min' not in self.clf.features.edge_features):
                temp.drop(labels=['fb_vertex_inside_dist_min','fb_vertex_outside_dist_min'], axis='columns', inplace=True)
            if('max' not in self.clf.features.edge_features):
                temp.drop(labels=['fb_vertex_inside_dist_max','fb_vertex_outside_dist_max','fb_vertex_last_dist_max'], axis='columns', inplace=True)
            if('sum' not in self.clf.features.edge_features):
                temp.drop(labels=['fb_vertex_inside_dist_sum','fb_vertex_outside_dist_sum'], axis='columns', inplace=True)
            self.edge_features = pd.concat((self.edge_features,temp),axis=1)

        if('facet' in self.clf.features.edge_features):
            temp = pd.read_csv(filepath_or_buffer=self.basefilename+"_fbff.txt", sep = ' ', header = 0, dtype = np.float32, index_col=False)
                               # skiprows=[1], usecols=[i for i in range(8)])  # because exporting was messed up, should be deleted at some point
            if('count' not in self.clf.features.edge_features):
                temp.drop(labels=['fb_facet_inside_count','fb_facet_outside_count'], axis='columns', inplace=True)
            if('min' not in self.clf.features.edge_features):
                temp.drop(labels=['fb_facet_inside_dist_min','fb_facet_outside_dist_min'], axis='columns', inplace=True)
            if('max' not in self.clf.features.edge_features):
                temp.drop(labels=['fb_facet_inside_dist_max','fb_facet_outside_dist_max','fb_facet_last_dist_max'], axis='columns', inplace=True)
            if('sum' not in self.clf.features.edge_features):
                temp.drop(labels=['fb_facet_inside_dist_sum','fb_facet_outside_dist_sum'], axis='columns', inplace=True)
            self.edge_features = pd.concat((self.edge_features,temp),axis=1)

        # reactivate following, once fbvf and fbff are reexported

        if('last' not in self.clf.features.edge_features):
            if ('vertex' in self.clf.features.edge_features):
                if ('count' in self.clf.features.edge_features):
                    self.edge_features.drop(labels=['fb_vertex_last_count'], axis='columns', inplace=True)
                if ('min' in self.clf.features.edge_features):
                    self.edge_features.drop(labels=['fb_vertex_last_dist_min'], axis='columns', inplace=True)
                if ('max' in self.clf.features.edge_features):
                    self.edge_features.drop(labels=['fb_vertex_last_dist_max'], axis='columns', inplace=True)
                if ('sum' in self.clf.features.edge_features):
                    self.edge_features.drop(labels=['fb_vertex_last_dist_sum'], axis='columns', inplace=True)
            if ('facet' in self.clf.features.edge_features):
                if('count' in self.clf.features.edge_features):
                    self.edge_features.drop(labels=['fb_facet_last_count'], axis='columns', inplace=True)
                if('min' in self.clf.features.edge_features):
                    self.edge_features.drop(labels=['fb_facet_last_dist_min'], axis='columns', inplace=True)
                if('max' in self.clf.features.edge_features):
                    self.edge_features.drop(labels=['fb_facet_last_dist_max'], axis='columns', inplace=True)
                if('sum' in self.clf.features.edge_features):
                    self.edge_features.drop(labels=['fb_facet_last_dist_sum'], axis='columns', inplace=True)

        # put a copy of the normalization feature at the beginning of the feature dataframe
        if(self.clf.regularization.reg_type):
            self.edge_features.insert(0, "reg_"+self.clf.regularization.reg_type, self.edge_features[self.clf.regularization.reg_type],allow_duplicates=False)

        self.edge_feature_names = list(self.edge_features.columns.values)

        assert (self.edge_lists.shape[1] == len(self.edge_features))
        assert(not self.edge_features.isnull().values.any())



    def readEdgeData_bin(self):

        self.edge_features = pd.DataFrame()
        if('shape' in self.clf.features.edge_features):
            temp = np.load(self.basefilename+"_fgeom.npz")
            for key,value in temp.items():
                self.edge_features[key] = value

        if('vertex' in self.clf.features.edge_features):
            temp = np.load(self.basefilename+"_fbvf.npz")
            for key,value in temp.items():
                self.edge_features[key] = value
            if('count' not in self.clf.features.edge_features):
                temp.drop(labels=['fb_vertex_inside_count','fb_vertex_outside_count'], axis='columns', inplace=True)
            if('min' not in self.clf.features.edge_features):
                temp.drop(labels=['fb_vertex_inside_dist_min','fb_vertex_outside_dist_min'], axis='columns', inplace=True)
            if('max' not in self.clf.features.edge_features):
                temp.drop(labels=['fb_vertex_inside_dist_max','fb_vertex_outside_dist_max','fb_vertex_last_dist_max'], axis='columns', inplace=True)
            if('sum' not in self.clf.features.edge_features):
                temp.drop(labels=['fb_vertex_inside_dist_sum','fb_vertex_outside_dist_sum'], axis='columns', inplace=True)

        if('facet' in self.clf.features.edge_features):
            temp = np.load(self.basefilename+"_fbff.npz")
            for key,value in temp.items():
                self.edge_features[key] = value
            if('count' not in self.clf.features.edge_features):
                temp.drop(labels=['fb_facet_inside_count','fb_facet_outside_count'], axis='columns', inplace=True)
            if('min' not in self.clf.features.edge_features):
                temp.drop(labels=['fb_facet_inside_dist_min','fb_facet_outside_dist_min'], axis='columns', inplace=True)
            if('max' not in self.clf.features.edge_features):
                temp.drop(labels=['fb_facet_inside_dist_max','fb_facet_outside_dist_max','fb_facet_last_dist_max'], axis='columns', inplace=True)
            if('sum' not in self.clf.features.edge_features):
                temp.drop(labels=['fb_facet_inside_dist_sum','fb_facet_outside_dist_sum'], axis='columns', inplace=True)

        if('last' not in self.clf.features.edge_features):
            if ('vertex' in self.clf.features.edge_features):
                if ('count' in self.clf.features.edge_features):
                    self.edge_features.drop(labels=['fb_vertex_last_count'], axis='columns', inplace=True)
                if ('min' in self.clf.features.edge_features):
                    self.edge_features.drop(labels=['fb_vertex_last_dist_min'], axis='columns', inplace=True)
                if ('max' in self.clf.features.edge_features):
                    self.edge_features.drop(labels=['fb_vertex_last_dist_max'], axis='columns', inplace=True)
                if ('sum' in self.clf.features.edge_features):
                    self.edge_features.drop(labels=['fb_vertex_last_dist_sum'], axis='columns', inplace=True)
            if ('facet' in self.clf.features.edge_features):
                if('count' in self.clf.features.edge_features):
                    self.edge_features.drop(labels=['fb_facet_last_count'], axis='columns', inplace=True)
                if('min' in self.clf.features.edge_features):
                    self.edge_features.drop(labels=['fb_facet_last_dist_min'], axis='columns', inplace=True)
                if('max' in self.clf.features.edge_features):
                    self.edge_features.drop(labels=['fb_facet_last_dist_max'], axis='columns', inplace=True)
                if('sum' in self.clf.features.edge_features):
                    self.edge_features.drop(labels=['fb_facet_last_dist_sum'], axis='columns', inplace=True)

        # put a copy of the normalization feature at the beginning of the feature dataframe
        if(self.clf.regularization.edge_type):
            self.edge_features.insert(0, "reg_"+self.clf.regularization.edge_type, self.edge_features[self.clf.regularization.edge_type],allow_duplicates=False)

        self.edge_feature_names = list(self.edge_features.columns.values)

        assert (self.edge_lists.shape[1] == len(self.edge_features))
        assert(not self.edge_features.isnull().values.any())

    def readAdjacencies(self):

        ## read adjacency
        files = [open(self.basefilename+"_adjacency_ij.txt", 'r')]
        files.append(open(self.basefilename+"_adjacency_ji.txt", 'r'))

        # TODO: this could be sped up, because I'm still reading as string and converting to int later
        # furthermore the _3dt.npz file could be used instead, but would need to be updated to also include infinite cells
        self.edge_lists = []
        for f in files:
            # skip header line
            next(iter(f))
            self.edge_lists.append(np.asarray(next(iter(f)).split(),dtype=np.int32))
            f.close()

        self.edge_lists = torch.Tensor([self.edge_lists[0],self.edge_lists[1]]).type(torch.LongTensor)
        a=5

    def readAdjacencies_bin(self):

        temp = np.load(self.basefilename+"_adjacencies.npz")
        self.edge_lists = torch.from_numpy(temp["adjacencies"]).type(torch.LongTensor)
        self.edge_lists = torch.transpose(self.edge_lists,1,0)
        a=5




    def standardizeFeatures(self):

        if('sum' in self.clf.features.scaling):
            if (self.clf.features.node_normalization_feature):
                self.features.iloc[:, 1:] = self.features.iloc[:, 1:]*10**3/self.features.iloc[:, 1:].sum()
            else:
                self.features = self.features * 10 ** 3 / self.features.sum()
            if(self.read_edge_features):
                self.edge_features=self.edge_features*10**3/self.edge_features.sum()

        if('vol' in self.clf.features.scaling):
            if (self.clf.features.node_normalization_feature):
                self.features.iloc[:, 1:] = (self.features.iloc[:, 1:]).div(self.features.norm+0.0001, axis=0)
            else:
                print("NOT IMPLEMENTED ERROR: cannot scale features by vol if shape features are turned off.")
                sys.exit(1)

        if(self.clf.features.node_normalization_feature is not None):
            if(self.clf.regularization.cell_type is not None):
                self.features.iloc[:, 1:] = (self.features.iloc[:, 1:]).div(self.features[self.clf.regularization.cell_type]+0.0001, axis=0)
            else:
                self.features = self.features.div(self.features[self.clf.regularization.cell_type] + 0.0001, axis=0)

        if('edge' in self.clf.features.scaling):
            self.features = self.features.div(self.mean_edge, axis=0)


        if('s' in self.clf.features.scaling):
            scaler = StandardScaler() # scales features to zero mean and unit variance
        elif('n' in self.clf.features.scaling):
            n = tuple(self.clf.features.normalization_range)
            scaler = MinMaxScaler(feature_range=n)  # scales features into the range between 0 and 1
        elif('r' in self.clf.features.scaling):
            scaler = RobustScaler() # scales features to zero mean and unit variance
        else:
            if(not 'sum' in self.clf.features.scaling):
                print("{} are no valid scalers. choose either 'sum', 's', 'n' or 'r'".format(self.clf.feature_scaling))
                sys.exit(1)
            else:
                return

        if(self.clf.regularization.cell_type is not None):
            # starting at col 1 because col 0 is the regularization feature
            scaler.fit(self.features.iloc[:, 1:])
            self.features.iloc[:, 1:] = scaler.transform(self.features.iloc[:, 1:])
        else:
            scaler.fit(self.features)
            self.features.iloc[:] = scaler.transform(self.features.iloc[:])

        if(self.read_edge_features):

            if (self.clf.features.edge_normalization_feature is not None):
                if (self.clf.regularization.edge_type is not None):
                    self.edge_features.iloc[:, 1:] = (self.edge_features.iloc[:, 1:]).div(self.edge_features[self.clf.regularization.edge_type] + 0.0001, axis=0)
                else:
                    self.edge_features = self.edge_features.div(self.edge_features[self.clf.regularization.edge_type] + 0.0001, axis=0)

            if (self.clf.regularization.edge_type is not None):
                scaler.fit(self.edge_features.iloc[:, 1:])
                self.edge_features.iloc[:, 1:] = scaler.transform(self.edge_features.iloc[:, 1:])
            else:
                scaler.fit(self.edge_features)
                self.edge_features.iloc[:] = scaler.transform(self.edge_features.iloc[:])





    def toTorch(self):

        # self.gt = torch.tensor(self.gt.values, dtype=torch.float)
        self.features = torch.tensor(self.features.values, dtype=torch.float)
        if(self.read_edge_features):
            self.edge_features = torch.tensor(self.edge_features.values, dtype=torch.float)
        else:
            self.edge_features = torch.empty(1,1, dtype=torch.float)

    def exportScore(self, prediction):

        outpath =  os.path.join(self.clf.paths.out,"prediction")
        if(self.verbosity):
            print("Export predictions to: ", outpath)

        # export predictions
        file = os.path.join(outpath,self.filename+".npz")
        f = open(file, 'wb')
        np.savez(f,
                 number_of_cells=int(len(prediction)),
                 sigmoid=prediction.sigmoid().numpy(),
                 logits=prediction.numpy(),
                 softmax=prediction.softmax(dim=-1).numpy())
        f.close()