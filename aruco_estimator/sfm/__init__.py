
from aruco_estimator.sfm.colmap import read_model as crm


#TODO Modify with mode argument or SfmProjectBase
def read_model(*args, **kwargs):
    return crm(*args, **kwargs)