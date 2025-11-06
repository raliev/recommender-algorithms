# algorithms/__init__.py
from .apriori import AprioriRecommender
from .bpr_adaptive import BPRAdaptiveRecommender
from .eclat import EclatRecommender
from .fpgrowth import FPGrowthRecommender
from .svd import SVDRecommender
from .als import ALSRecommender
from .bpr import BPRRecommender
from .als_improved import ALSImprovedRecommender
from .item_knn import ItemKNNRecommender
from .slope_one import SlopeOneRecommender
from .nmf import NMFRecommender
from .als_pyspark import ALSPySparkRecommender
from .funksvd import FunkSVDRecommender
from .puresvd import PureSVDRecommender
from .svdpp import SVDppRecommender
from .toppopular import TopPopularRecommender
from .wrmf import WRMFRecommender
from .cml import CMLRecommender
from .user_knn import UserKNNRecommender
from .ncf import NCFRecommender
from .sasrec import SASRecRecommender
from .slim import SLIMRecommender
from .autoencoder import VAERecommender
from .fism import FISMRecommender


__all__ = [
    'SVDRecommender',
    'ALSRecommender',
    'BPRRecommender',
    'BPRAdaptiveRecommender',
    'ALSImprovedRecommender',
    'ItemKNNRecommender',
    'SlopeOneRecommender',
    'NMFRecommender',
    'ALSPySparkRecommender',
    'FunkSVDRecommender',
    'PureSVDRecommender',
    'SVDppRecommender',
    'WRMFRecommender',
    'CMLRecommender',
    'UserKNNRecommender',
    'NCFRecommender',
    'SASRecRecommender',
    'SLIMRecommender',
    'VAERecommender',
    'FISMRecommender',
    'TopPopularRecommender',
    'AprioriRecommender',
    'FPGrowthRecommender',
    'EclatRecommender',
    'ASVDRecommender',
    'WMFBPRRecommender'
]