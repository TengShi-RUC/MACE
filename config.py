import os

# max_seq_len = 300
# trainDataPath = 'data/trainSet.npy'
# testDataPath = 'data/testSet.npy'

# DMF_trainDataPath = 'data/DMF_trainSet.npy'
# DMF_testDataPath = 'data/DMF_testSet.npy'
# DMF_embeddingPath = 'data/DMF_embedding.npy'
# randomType 0(极端分布) 1(均匀分布)
# groupFeature = 'occupation'
# randomFeature = None
# randomType = -1
# testAlgo = 'din_base'

# ml-1m
# userNum:6040 itemNum:3952
# seqTrainSize:  988129
# seqTestSize:  6040
# interactTrainSize:  994169
# interactTestSize:  6040

# insurance
# userNum:29132 itemNum:21
# interactTrainSize:  550434
# interactTestSize:  61338


class BaseConfig:
    def __init__(
        self,
        args,
        GF_mixup=False,
        CGF_mixup=False,
        mixup_method=None,
        use_iv=False,
        use_MI=False,
        AdvLearning=False,
        random_iv=False,
        direct_use_gexo=False
    ):
        self.analysis = args.analysis
        if args.analysis:
            self.train_h1 = args.train_h1
            self.train_h2 = args.train_h2
            self.test_h1 = args.test_h1
            self.test_h2 = args.test_h2

        self.epochs = args.epochs
        self.groupFeature = args.groupFeature
        self.train = args.train
        self.device = args.device
        self.useSensitiveFeature = args.useSensitiveFeature
        self.description = args.algos

        self.learning_rate = 0.001
        self.interval = 100
        self.l2_penalty = 0.001
        self.earlyStop = 5
        self.train_batch_size = 128
        self.test_batch_size = 128

        self.mixup_group_size = 32
        self.GF_mixup = GF_mixup  # 只在原始数据Mixup
        self.CGF_mixup = CGF_mixup  # 在多个分布上Mixup
        self.mixup_method = mixup_method
        self.use_iv = use_iv  # 工具变量重构
        self.ridge_lambd = 0.9
        self.use_MI = use_MI  # 互信息优化
        self.AdvLearning = AdvLearning
        self.Adv_discriminator_iter = 10

        # 消融实验
        self.random_iv = random_iv
        self.direct_use_gexo = direct_use_gexo

        self.dataset = args.dataset

        if self.dataset == "ml-1m":
            self.itemNum = 3952
            self.userNum = 6040
            self.sensitiveFeatureNum = 3
            self.sensitiveFeatureClass = {
                "gender": 2, "age": 7, "occupation": 21}
            self.interactMatrix = "data/ml-1m/interactMatrix.npy"
        elif self.dataset == "insurance":
            self.itemNum = 21
            self.userNum = 29132
            self.sensitiveFeatureNum = 3
            self.interactMatrix = "data/insurance/interactMatrix.npy"
            self.sensitiveFeatureClass = {
                "gender": 2,
                "marital_status": 8,
                "occupation": 6,
            }
        else:
            raise ValueError("dataset error!")

        self.base_model_dir = "./{}_model/".format(self.dataset)
        if not os.path.exists(self.base_model_dir):
            os.mkdir(self.base_model_dir)

        if self.GF_mixup or self.CGF_mixup:
            self.fair_loss_weight = args.fairWeight
            self.modelInfo = "fairWeight:{}".format(self.fair_loss_weight)
        elif self.AdvLearning:
            self.discriminator_weight = args.discriminatorWeight
            self.modelInfo = "discriminatorWeight:{}".format(
                self.discriminator_weight)
        else:
            self.modelInfo = None

        self.modelPath = self.createSaveDir(self.modelInfo)

        if self.use_MI:
            self.MI_estimator_name = "CLUBSample"
            self.MI_figure_save_path = self.createMIPath(
                base_path="{}MI_figure/".format(self.base_model_dir), fileType=".png"
            )
            self.csvPath = self.createMIPath(
                base_path="{}csv_result/".format(self.base_model_dir), fileType=".csv"
            )

    def createSaveDir(self, otherInfo=None):
        if self.analysis:
            model_dir = self.base_model_dir + "analysis_checkpoints/"
        else:
            model_dir = self.base_model_dir + "checkpoints/"
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        if not os.path.exists(model_dir + self.description):
            os.mkdir(model_dir + self.description)

        if self.useSensitiveFeature:
            finalPath = model_dir + self.description + "/" + "with_sensitive/"
        else:
            finalPath = model_dir + self.description + "/" + "without_sensitive/"
        if not os.path.exists(finalPath):
            os.mkdir(finalPath)
        if otherInfo:
            finalPath = finalPath + otherInfo + "/"
            if not os.path.exists(finalPath):
                os.mkdir(finalPath)
        if self.analysis:
            finalPath += "train_h1={}train_h2={}test_h1={}test_h2={}/".format(
                self.train_h1, self.train_h2, self.test_h1, self.test_h2)
            if not os.path.exists(finalPath):
                os.mkdir(finalPath)
        # for file in os.listdir(finalPath):
        #     os.remove(finalPath + file)
        return finalPath

    def createMIPath(self, base_path, fileType):
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        if not os.path.exists(base_path + self.description):
            os.mkdir(base_path + self.description)
        if self.useSensitiveFeature:
            MI_final_path = base_path + self.description + "/with_sensitive/"
        else:
            MI_final_path = base_path + self.description + "/without_sensitive/"
        if not os.path.exists(MI_final_path):
            os.mkdir(MI_final_path)
        return MI_final_path + self.modelInfo + fileType


class DIN_Config(BaseConfig):
    def __init__(
        self,
        args,
        GF_mixup=False,
        CGF_mixup=False,
        mixup_method=None,
        use_iv=False,
        use_MI=False,
        AdvLearning=False,
        random_iv=False,
        direct_use_gexo=False
    ):
        super().__init__(
            args, GF_mixup, CGF_mixup, mixup_method, use_iv, use_MI, AdvLearning, random_iv, direct_use_gexo
        )
        self.itemEmbeddingSize = 128
        if self.useSensitiveFeature:
            self.userEmbeddingSize = self.itemEmbeddingSize + self.sensitiveFeatureNum
        else:
            self.userEmbeddingSize = self.itemEmbeddingSize

        if self.use_iv:
            self.d_exo = 32
            self.Gexo = {
                "hidden_dims": [self.userEmbeddingSize, 64, self.d_exo],
                "dropout": [0.1, 0.1],
                "is_dropout": True,
            }

            if not self.direct_use_gexo:
                self.d_endo = 32
                self.Gendo = {
                    "hidden_dims": [self.userEmbeddingSize, 64, self.d_endo],
                    "dropout": [0.1, 0.1],
                    "is_dropout": True,
                }
                self.h1 = {
                    "hidden_dims": [64, 32, 16, 1],
                    "dropout": [],
                    "is_dropout": False,
                }
                self.h2 = {
                    "hidden_dims": [64, 32, 16, 1],
                    "dropout": [],
                    "is_dropout": False,
                }
        if self.use_MI:
            self.MI_estimator_hiddenSize = 32
            # self.MI_train_epochs = 2
            self.MI_eachBatchIter = 5


class DMF_Config(BaseConfig):
    def __init__(
        self,
        args,
        GF_mixup=False,
        CGF_mixup=False,
        mixup_method=None,
        use_iv=False,
        use_MI=False,
        AdvLearning=False,
        random_iv=False,
        direct_use_gexo=False
    ):
        super().__init__(
            args, GF_mixup, CGF_mixup, mixup_method, use_iv, use_MI, AdvLearning, random_iv, direct_use_gexo
        )

        self.embeddingPath = self.interactMatrix
        self.userDim = self.itemNum
        self.userMLP = {
            "hidden_dims": [self.userDim, 128, 64],
            "dropout": [0.1, 0.1],
            "is_dropout": True,
        }

        if self.useSensitiveFeature:
            itemOutdim = 64 + self.sensitiveFeatureNum
        else:
            itemOutdim = 64

        self.userEmbeddingSize = itemOutdim

        if self.use_iv:
            self.d_exo = 32
            inputSize = itemOutdim
            self.Gexo = {
                "hidden_dims": [inputSize, 64, self.d_exo],
                "dropout": [0.1, 0.1],
                "is_dropout": True,
            }
            if not self.direct_use_gexo:
                self.d_endo = 32
                self.Gendo = {
                    "hidden_dims": [inputSize, 64, self.d_endo],
                    "dropout": [0.1, 0.1],
                    "is_dropout": True,
                }
                self.h1 = {
                    "hidden_dims": [64, 32, 16, 1],
                    "dropout": [],
                    "is_dropout": False,
                }
                self.h2 = {
                    "hidden_dims": [64, 32, 16, 1],
                    "dropout": [],
                    "is_dropout": False,
                }
            itemOutdim = self.d_exo if self.direct_use_gexo else self.d_endo

        self.itemMLP = {
            "hidden_dims": [self.userNum, 128, itemOutdim],
            "dropout": [0.1, 0.1],
            "is_dropout": True,
        }

        if self.use_MI:
            self.MI_estimator_hiddenSize = 32
            self.MI_eachBatchIter = 5


class DeepModel_Config(BaseConfig):
    def __init__(
        self,
        args,
        GF_mixup=False,
        CGF_mixup=False,
        mixup_method=None,
        use_iv=False,
        use_MI=False,
        AdvLearning=False,
        random_iv=False,
        direct_use_gexo=False
    ):
        super().__init__(
            args, GF_mixup, CGF_mixup, mixup_method, use_iv, use_MI, AdvLearning, random_iv, direct_use_gexo
        )
        self.user_vector_size = 256
        self.item_vector_size = 256
        if self.useSensitiveFeature:
            self.userDim = self.user_vector_size + self.sensitiveFeatureNum
        else:
            self.userDim = self.user_vector_size
        self.itemDim = self.item_vector_size
        self.userEmbeddingSize = self.userDim

        if self.use_iv:
            self.d_exo = 32
            self.Gexo = {
                "hidden_dims": [self.userDim, 64, self.d_exo],
                "dropout": [0.1, 0.1],
                "is_dropout": True,
            }
            if not self.direct_use_gexo:
                self.d_endo = 32
                self.Gendo = {
                    "hidden_dims": [self.userDim, 64, self.d_endo],
                    "dropout": [0.1, 0.1],
                    "is_dropout": True,
                }
                self.h1 = {
                    "hidden_dims": [64, 32, 16, 1],
                    "dropout": [],
                    "is_dropout": False,
                }
                self.h2 = {
                    "hidden_dims": [64, 32, 16, 1],
                    "dropout": [],
                    "is_dropout": False,
                }
            self.userDim = self.d_exo if self.direct_use_gexo else self.d_endo

        self.mainMLP = {
            "hidden_dims": [self.userDim + self.itemDim, 256, 256, 256, 256, 1],
            "dropout": [0.1, 0.1, 0.1, 0.1, 0.1],
            "is_dropout": True,
        }
        if self.use_MI:
            self.MI_estimator_hiddenSize = 32
            self.MI_eachBatchIter = 5
